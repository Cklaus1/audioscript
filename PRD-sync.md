# AudioScript Sync Directory — Feature PRD

**Version:** 1.0
**Date:** 2026-03-24
**Status:** Draft
**Parent:** PRD.md (Phase 2, F-20 evolution)

---

## 1. Problem Statement

Users accumulate audio recordings across devices — phone voice memos synced via OneDrive, meeting recordings, lecture captures. Today, transcribing these requires manually running `audioscript transcribe --input "path/*.m4a"` each time new files appear. There's no persistent "point at a folder and keep it transcribed" workflow.

**Real example:** `/mnt/c/Users/cklau/OneDrive/Documents/Sound Recordings/` contains 219 recordings (213 m4a, 4 opus, 2 mp4) that grow over time via Windows Sound Recorder → OneDrive sync. A user on WSL wants these auto-transcribed and pushed to MiNotes without manual intervention.

---

## 2. Feature Overview

### `audioscript sync` — Persistent directory synchronization

A new command and config section that:
1. **Scans** a source directory for audio files
2. **Diffs** against the manifest to find new/changed files
3. **Transcribes** only what's new (content-addressed via SHA-256)
4. **Exports** results to configured targets (output dir, MiNotes, markdown)
5. **Supports WSL→Windows paths** transparently (`C:\Users\...` auto-translated)
6. **Runs as one-shot or watch mode** (poll-based, no inotify across WSL/Windows boundary)

### Config-driven (`.audioscript.yaml`)

```yaml
sync:
  # Source directories to watch for audio files
  sources:
    - path: "C:\\Users\\cklau\\OneDrive\\Documents\\Sound Recordings"
      # Per-source overrides (inherit global defaults)
      tier: balanced
      diarize: true
      export: minotes

    - path: "./local-recordings"
      tier: draft

  # File discovery
  extensions: [m4a, mp3, wav, flac, ogg, opus, webm, mp4, wma, aac]
  recursive: true           # Scan subdirectories
  ignore_patterns:           # Glob patterns to skip
    - "*.tmp"
    - ".*"                   # Hidden files
    - "_processed/*"

  # Sync behavior
  poll_interval: 300         # Seconds between scans in watch mode (default: 5 min)
  batch_size: 10             # Max files per sync cycle (0 = unlimited)
  delay_between: 2           # Seconds between files (resource management)
  skip_older_than: null      # Skip files older than N days (null = no limit)
  min_file_size: 1024        # Skip files smaller than N bytes (avoid empty/corrupt)
  max_file_size: null        # Skip files larger than N bytes (null = no limit)

  # Output
  output_dir: "./transcripts/sync"   # Where transcripts go
  output_format: markdown             # Default format for synced files

  # MiNotes integration
  minotes:
    enabled: true
    sync_dir: null            # Auto-detect or explicit path
    journal: true             # Add journal entry per sync cycle
    folder: "Transcripts"     # MiNotes folder for transcript pages

  # WSL
  wsl:
    auto_detect: true         # Auto-translate Windows paths when running in WSL
    mount_prefix: "/mnt/"     # WSL mount prefix (default, configurable via wsl.conf)
```

---

## 3. Requirements

### 3.1 Core: `audioscript sync` Command

**One-shot mode (default):**
```bash
# Sync all configured sources
audioscript sync

# Sync a specific directory (overrides config)
audioscript sync --source "C:\Users\cklau\OneDrive\Documents\Sound Recordings"

# Dry-run: show what would be transcribed
audioscript sync --dry-run

# Pipe mode: emit NDJSON per-file
audioscript sync --pipe
```

**Watch mode:**
```bash
# Continuous polling (Ctrl+C to stop)
audioscript sync --watch

# Custom poll interval
audioscript sync --watch --poll-interval 60
```

**CLI options:**
```
--source PATH          Override source directory
--watch                Continuous polling mode
--poll-interval N      Seconds between scans (default: 300)
--batch-size N         Max files per cycle (default: 10)
--skip-older-than N    Skip files older than N days
--force                Re-transcribe all files
--export TARGET        Override export target (minotes, none)
--output-dir PATH      Override output directory
--output-format FMT    Override output format
```

### 3.2 File Discovery & Diff

**Scanning:**
- Glob for configured extensions in source directories
- Respect `recursive`, `ignore_patterns`, `min_file_size`, `max_file_size`
- Sort by modification time (newest first) for most-relevant-first processing

**Diff logic (content-addressed):**
- Compute SHA-256 hash for each discovered file
- Check against per-project manifest (`{output_dir}/.audioscript_manifest.json`)
- File is "new" if hash not in manifest OR manifest status is "error"
- File is "changed" if path exists in manifest but hash differs (renamed/moved files handled by content addressing)
- File is "skipped" if hash exists with status "completed" at same tier+version

**Performance for large directories:**
- Hash only files whose (size, mtime) pair differs from cached values
- Cache (path, size, mtime, hash) in `.audioscript_sync_cache.json` alongside manifest
- Avoids re-hashing 219 files on every scan when nothing changed

### 3.3 WSL Path Translation

**Detection:**
- Check `/proc/version` for `microsoft` or `Microsoft` string
- Check `WSL_DISTRO_NAME` environment variable
- Cache result for process lifetime

**Translation rules:**
- Windows path (`C:\Users\...` or `C:/Users/...`) → WSL path (`/mnt/c/Users/...`)
- Use `wslpath -u` subprocess when available (handles edge cases, custom mount points)
- Fallback to regex translation: `^([A-Za-z]):[/\\](.*)` → `/mnt/{drive_lower}/{rest}`
- Respect custom mount prefix from `wsl.conf` `[automount]` section

**When to translate:**
- On config load: translate all `sync.sources[].path` values if they look like Windows paths
- On CLI `--source` argument: translate before validation
- Transparent to user: `audioscript sync --source "C:\Users\cklau\OneDrive\Documents\Sound Recordings"` just works
- Log the translation: `"Translated WSL path: C:\... → /mnt/c/..."` at INFO level

**Path validation:**
- After translation, verify the path exists and is readable
- If translation fails or path doesn't exist, emit clear error:
  ```
  Error: Source directory not found: /mnt/c/Users/cklau/OneDrive/Documents/Sound Recordings
  Hint: If running in WSL, ensure the Windows drive is mounted.
        Check that OneDrive has synced the files locally.
  ```

### 3.3.1 OneDrive Files On-Demand Handling

**The problem:** OneDrive Files On-Demand shows placeholder files in `/mnt/c/` that look like real files (correct filename, correct reported size) but are cloud-only. Attempting to read them from WSL fails with `OSError: [Errno 22] Invalid argument` because the 9P bridge can't trigger OneDrive's download mechanism.

**Tested behavior (2026-03-25):**
- `ls` shows all 219 files with correct sizes
- `cp` fails on cloud-only files with "Invalid argument"
- `cp` succeeds on locally-cached files
- Only ~5-10% of files were locally available without explicit download
- WSL file reads via 9P do NOT trigger OneDrive download (unlike native Windows reads)

**Solution: Multi-stage download pipeline**

```
Stage 1: Probe     — try to read first 1KB of each file
Stage 2: Trigger   — for failed files, call PowerShell to pin for download
Stage 3: Wait      — poll until files are available (OneDrive downloads in background)
Stage 4: Transcribe — process all available files
```

#### Stage 1: Probe (fast, no side effects)

```python
def probe_file(path: Path) -> str:
    """Test if a file is locally available.
    Returns: 'local' | 'cloud' | 'error'
    """
    try:
        with open(path, 'rb') as f:
            f.read(1024)
        return 'local'
    except OSError:
        return 'cloud'
    except PermissionError:
        return 'error'  # OneDrive lock or permissions
```

Run on all files first — fast (stat + 1KB read), separates local from cloud-only.

#### Stage 2: Trigger Download via PowerShell

```python
def trigger_onedrive_download(paths: list[Path]) -> None:
    """Pin files for local availability via OneDrive.

    Uses PowerShell's attrib.exe to remove the 'cloud-only' flag,
    which triggers OneDrive to download the file.
    """
    for path in paths:
        win_path = subprocess.run(
            ['wslpath', '-w', str(path)],
            capture_output=True, text=True,
        ).stdout.strip()

        # Method 1: attrib.exe (removes Unpinned flag, adds Pinned)
        subprocess.run(
            ['attrib.exe', '-U', '+P', win_path],
            capture_output=True, timeout=10,
        )
```

**Alternative methods (fallback chain):**

| Method | Command | Reliability | Notes |
|--------|---------|-------------|-------|
| **attrib.exe** | `attrib.exe -U +P "C:\path\file.m4a"` | High | Changes file attribute from "cloud" to "pinned" |
| **PowerShell** | `powershell.exe -c "(Get-Item 'C:\path\file.m4a').Attributes"` | Medium | Reading attributes can trigger download |
| **cmd.exe copy** | `cmd.exe /c copy "C:\path\file.m4a" NUL` | Medium | Reading file content triggers download |
| **Windows touch** | `powershell.exe -c "[IO.File]::ReadAllBytes('C:\path\file.m4a') | Out-Null"` | High | Forces full file read on Windows side |

**Recommended:** `attrib.exe -U +P` as primary (changes OneDrive pin status). Falls back to PowerShell ReadAllBytes if attrib fails.

#### Stage 3: Wait for Downloads

```python
def wait_for_downloads(
    paths: list[Path],
    timeout: int = 300,      # 5 min default
    poll_interval: int = 5,  # Check every 5 seconds
) -> tuple[list[Path], list[Path]]:
    """Wait for OneDrive to download triggered files.
    Returns (ready, still_pending).
    """
    deadline = time.time() + timeout
    pending = set(paths)
    ready = []

    while pending and time.time() < deadline:
        for path in list(pending):
            if probe_file(path) == 'local':
                pending.remove(path)
                ready.append(path)
        if pending:
            time.sleep(poll_interval)

    return ready, list(pending)
```

#### Stage 4: Copy to Local Staging (Performance)

Reading from `/mnt/c/` is 10-50x slower than native Linux filesystem. For transcription (which reads the full file), copying to a local staging directory first is faster overall:

```python
def stage_files(paths: list[Path], staging_dir: Path) -> list[Path]:
    """Copy OneDrive files to local Linux filesystem for fast processing."""
    staged = []
    for path in paths:
        dest = staging_dir / path.name
        shutil.copy2(path, dest)  # Copy with metadata
        staged.append(dest)
    return staged
```

**Performance math for 219 files:**
- Reading 219 files from `/mnt/c/` during transcription: ~50x I/O penalty per file
- Copying 219 files to local first: ~2-5 min one-time cost
- Transcription on local files: full GPU speed
- Net savings: significant for large files

#### CLI Integration

```bash
# Auto-handle OneDrive (default behavior)
audioscript sync --source "C:\Users\cklau\OneDrive\Documents\Sound Recordings"
# → Probes 219 files
# → 15 locally available, 204 cloud-only
# → Triggers download for 204 files via attrib.exe
# → Waits up to 5 min for downloads
# → Stages to local dir
# → Transcribes all available files
# → Reports: "Transcribed 180/219. 39 still downloading (will process on next sync)."

# Skip download trigger (only process what's already local)
audioscript sync --source "..." --no-download
# → Only transcribes the ~15 locally available files

# Force full download with longer timeout
audioscript sync --source "..." --download-timeout 1800
# → Waits up to 30 min for all files to download

# Explicit download-only step (no transcription)
audioscript sync --source "..." --download-only
# → Triggers downloads, waits, reports status. No transcription.
```

#### Config

```yaml
sync:
  onedrive:
    auto_download: true          # Trigger download for cloud-only files
    download_timeout: 300        # Seconds to wait for downloads (default: 5 min)
    download_poll_interval: 5    # Seconds between download checks
    staging_dir: null             # Local staging dir (default: /tmp/audioscript-stage/)
    cleanup_staging: true         # Delete staged files after transcription
    max_concurrent_downloads: 10  # Don't overwhelm OneDrive with 219 simultaneous pins

  # Graceful degradation
  on_cloud_only: "trigger"       # "trigger" | "skip" | "error"
  #   trigger: attempt to download via attrib.exe/PowerShell
  #   skip: log warning, process only local files
  #   error: fail with actionable error message
```

#### Error Messages

```
# When OneDrive files can't be downloaded
Warning: 204/219 files are cloud-only (OneDrive Files On-Demand).
Triggered download for 204 files via attrib.exe.
Waiting up to 5 min for OneDrive to sync...
  [30s] 45/204 downloaded...
  [60s] 120/204 downloaded...
  [90s] 180/204 downloaded...
  [120s] 204/204 downloaded ✓
Staging files to /tmp/audioscript-stage/...

# When downloads time out
Warning: 39/219 files still downloading after 5 min timeout.
Transcribed 180 files. Run `audioscript sync` again to process remaining files.
Hint: Use --download-timeout 1800 for slower connections.

# When attrib.exe is not available (not WSL or unusual setup)
Error: Cannot trigger OneDrive download — attrib.exe not found.
Hint: Run from Windows PowerShell:
  attrib -U +P "C:\Users\cklau\OneDrive\Documents\Sound Recordings\*"
Then re-run: audioscript sync
```

### 3.4 Sync State & Reporting

**Per-sync-cycle report:**
```json
{
  "ok": true,
  "command": "sync",
  "data": {
    "source": "/mnt/c/Users/cklau/OneDrive/Documents/Sound Recordings",
    "scanned": 219,
    "new": 5,
    "skipped": 214,
    "transcribed": 5,
    "failed": 0,
    "output_dir": "./transcripts/sync",
    "exported_to": "minotes",
    "results": [
      {"file": "Recording (215).m4a", "status": "completed", "duration": "2m 15s"},
      {"file": "Recording (216).m4a", "status": "completed", "duration": "0m 45s"}
    ]
  }
}
```

**Watch mode status:**
```
[14:30:00] Scan #1: 5 new files found, transcribing...
[14:30:45] Scan #1: 5/5 completed (3m 12s total)
[14:35:00] Scan #2: 0 new files, sleeping 300s...
[14:40:00] Scan #3: 1 new file found, transcribing...
```

### 3.5 MiNotes Integration (Sync-Specific)

When `sync.minotes.enabled: true`:
- All transcribed files auto-export to MiNotes (reuses F-01 MiNotesExporter)
- Transcripts placed in configured MiNotes folder (`sync.minotes.folder`)
- Per-cycle journal entry: `"Synced 5 recordings from Sound Recordings: [[Recording 215]], [[Recording 216]], ..."`
- Link related transcripts from same source directory

### 3.6 Config Integration

**New section in `.audioscript.yaml`:**

The `sync` block is a new top-level section. All transcription settings (tier, model, diarize, etc.) can be overridden per-source or inherited from the global config.

**Merge order:**
1. Global `.audioscript.yaml` defaults (tier, model, output_format, etc.)
2. `sync.*` section defaults (output_dir, output_format, extensions)
3. `sync.sources[].` per-source overrides (tier, diarize, export)
4. CLI flags (highest priority)

**New Pydantic models:**
```python
class SyncSourceConfig(BaseModel):
    path: str
    # All AudioScriptConfig fields as optional overrides
    tier: TranscriptionTier | None = None
    model: str | None = None
    diarize: bool | None = None
    export: str | None = None
    output_format: str | None = None

class SyncMiNotesConfig(BaseModel):
    enabled: bool = False
    sync_dir: str | None = None
    journal: bool = True
    folder: str = "Transcripts"

class SyncConfig(BaseModel):
    sources: list[SyncSourceConfig] = []
    extensions: list[str] = ["m4a", "mp3", "wav", "flac", "ogg", "opus", "webm", "mp4", "wma", "aac"]
    recursive: bool = True
    ignore_patterns: list[str] = ["*.tmp", ".*"]
    poll_interval: int = 300
    batch_size: int = 10
    delay_between: float = 2.0
    skip_older_than: int | None = None
    min_file_size: int = 1024
    max_file_size: int | None = None
    output_dir: str = "./transcripts/sync"
    output_format: str = "markdown"
    minotes: SyncMiNotesConfig = SyncMiNotesConfig()

class WSLConfig(BaseModel):
    auto_detect: bool = True
    mount_prefix: str = "/mnt/"
```

### 3.7 Meeting Intelligence — Structured Post-Processing

The transcript is raw material. The value is in **structured sections** that make recordings searchable, actionable, and classifiable. Every synced transcript produces a rich markdown page with these sections:

#### Full Page Structure

```markdown
---
title: "Transcript: Q1 Budget Review"
date: 2026-03-24
source: audioscript
duration: 2345.6
language: en
backend: faster-whisper
speakers: [Alice, Bob, Carol]
classification: business-meeting
topics: [budget, Q1 results, hiring plan]
project: "Work/Meetings"
tags: [transcript, audioscript, business-meeting]
---

# Q1 Budget Review

## Classification
- **Type:** Business Meeting
- **Confidence:** 0.92
- **Topics:** Budget, Q1 Results, Hiring Plan

## Metadata
| Property | Value |
|----------|-------|
| Duration | 39m 5s |
| Language | English |
| Speakers | 3 (Alice, Bob, Carol) |
| File Size | 45.2 MB |
| Recorded | 2026-03-24 14:00 |

## Summary
Alice led a review of Q1 financial results. Revenue exceeded targets by 12%.
Bob presented the hiring plan for Q2 — 3 engineering roles approved.
Carol flagged concerns about infrastructure costs trending above budget.

## Action Items
- [ ] Bob: Post engineering job descriptions by Friday (2026-03-28)
- [ ] Carol: Prepare infrastructure cost breakdown for next meeting
- [ ] Alice: Schedule follow-up review for April 7

## Key Decisions
- Approved Q2 hiring plan (3 engineering, 1 design)
- Deferred cloud migration to Q3 due to budget constraints
- Agreed to monthly cost reviews going forward

## Key Insights
- Revenue 12% above Q1 target — strongest quarter in 2 years
- Infrastructure costs growing 8% MoM — needs attention
- Team morale high based on tone analysis

## Follow-ups
- [[Q2 Hiring Plan]] — Bob to draft
- [[Infrastructure Cost Review]] — Carol, due April 7
- [[Monthly Cost Review Template]] — Alice to create

## Transcript

### Alice — 00:00:00
Welcome everyone. Let's start with the Q1 results...

### Bob — 00:05:23
Thanks Alice. I've prepared the hiring plan slides...

### Carol — 00:12:45
Before we move on, I want to flag the infrastructure costs...
```

#### Section Descriptions

| Section | Source | Required? |
|---------|--------|-----------|
| **Classification** | LLM or rule-based (see §3.8) | Optional (needs `--classify`) |
| **Metadata** | Extracted from audio file + transcription stats | Always |
| **Summary** | LLM-based (abstractive) or extractive fallback | When `summarize: true` |
| **Action Items** | LLM extraction or pattern matching ("I'll...", "by Friday", "TODO") | When `summarize: true` |
| **Key Decisions** | LLM extraction or pattern matching ("we agreed", "decision:", "approved") | When `summarize: true` |
| **Key Insights** | LLM-generated or notable quotes/data points | When `summarize: true` + LLM available |
| **Follow-ups** | Extracted from action items, linked as `[[wikilinks]]` | When `summarize: true` |
| **Transcript** | Whisper transcription + diarization | Always |

#### Implementation Tiers

**Tier 1 — Rule-based extraction (no LLM, default):**
- Summary: TextRank extractive (top 3-5 sentences)
- Action items: Regex patterns — future tense + person + deadline ("Bob will... by Friday", "TODO:", "Action:")
- Decisions: Regex patterns — "we decided", "agreed to", "approved", "resolution:"
- Follow-ups: Items extracted from action items with `[[wikilinks]]`
- Insights: Skipped (requires LLM)
- Classification: Keyword-based heuristics (see §3.8)

**Tier 2 — LLM-based (opt-in, `--summary-provider claude|openai`):**
- Summary: Abstractive, 2-3 paragraphs
- Action items: Structured extraction with assignee, deadline, description
- Decisions: Full decision log with context
- Insights: Generated from content + tone analysis
- Follow-ups: Intelligent linking suggestions
- Classification: LLM-based with confidence score

**Config:**
```yaml
sync:
  sources:
    - path: "C:\\Users\\cklau\\OneDrive\\Documents\\Sound Recordings"
      summarize: true
      classify: true
      # summary_provider: claude  # Optional LLM tier
      # summary_api_key: sk-...   # Or use env var ANTHROPIC_API_KEY
```

### 3.8 Meeting Classification

Auto-classify recordings by type to enable routing, filtering, and organization.

#### Classification Taxonomy

```yaml
classifications:
  business-meeting:
    keywords: [agenda, action item, quarterly, revenue, budget, deadline, stakeholder, KPI]

  family:
    keywords: [family, kids, school, dinner, vacation, birthday, holiday]

  brainstorm:
    keywords: [idea, brainstorm, what if, concept, prototype, explore, creative]

  one-on-one:
    keywords: [feedback, career, growth, performance, check-in, 1:1]

  interview:
    keywords: [candidate, resume, experience, tell me about, why do you want]

  lecture:
    keywords: [chapter, slide, homework, exam, textbook, professor, class]

  voice-memo:
    keywords: []  # Fallback for short, single-speaker recordings

  sales-call:
    keywords: [demo, pricing, proposal, contract, close, deal, pipeline, objection, competitor, ROI, discount, trial, procurement]

  discovery-call:
    keywords: [tell me about, pain point, how do you currently, what's your biggest challenge, workflow, frustration, workaround, use case]

  call:
    keywords: [hello, calling about, phone, voicemail]

  podcast:
    keywords: [episode, listener, welcome back, subscribe, sponsor, show notes]

  other:
    keywords: []  # Default fallback
```

Classification becomes a `select` property and a tag on the MiNotes page — not a folder.
Classification also **determines which analysis modules run** (see §3.12 and §3.13).

#### Classification Logic

**Rule-based (default):**
1. Count keyword matches in first 2 minutes of transcript
2. Factor in speaker count (1 speaker → voice-memo/lecture, 2 → call/1:1, 3+ → meeting)
3. Factor in duration (<2 min → voice-memo, 2-15 min → call, 15+ min → meeting)
4. Weighted score → pick highest classification
5. Confidence: score / max_possible_score

**LLM-based (opt-in):**
- Send first 500 words + speaker count + duration to LLM
- Ask for classification + confidence + detected topics
- Return structured result

**Configurable:**
```yaml
sync:
  classify: true
  classifications:
    # Override or extend the default taxonomy
    standup:
      keywords: [standup, blocker, yesterday, today, sprint]
      minotes_folder: "Work/Standups"
```

### 3.9 MiNotes: Single Folder + Property-Based Classification

All transcripts live in **one folder** ("Transcripts"). Classification, topics, and meeting type are baked into **MiNotes properties** on each page — not folder structure. This keeps things flat and queryable.

#### Why properties over folders

- No folder sprawl — all transcripts in one place
- MiNotes properties are typed and queryable via SQL: `minotes query "SELECT * FROM properties WHERE key = 'classification' AND value = 'business-meeting'"`
- `multi-select` properties act as tags — filter/search by topic, type, speaker
- Properties show in MiNotes page sidebar (like Notion database fields)
- Obsidian frontmatter tags serve the same role for Obsidian users

#### MiNotes Property Schema

```
Class: transcript
Properties:
  - source_file       (text)          "/mnt/c/.../Recording (215).m4a"
  - duration          (text)          "2m 15s"
  - language          (text)          "en"
  - speakers          (number)        3
  - classification    (select)        business-meeting | family | brainstorm | one-on-one | ...
  - classification_confidence (number) 0.85
  - topics            (multi-select)  "budget, Q1 results, hiring"
  - tags              (multi-select)  "transcript, audioscript, business-meeting"
  - action_item_count (number)        3
  - decision_count    (number)        2
  - transcribed_at    (datetime)      2026-03-24T14:30:00Z
  - confidence_avg    (number)        0.87
```

#### How It Works

1. All transcripts export to `sync.minotes.folder` (default: "Transcripts") — one flat folder
2. Classification runs → sets `classification` property (select) and `topics` property (multi-select)
3. YAML frontmatter `tags:` array mirrors MiNotes tags for Obsidian compatibility
4. User queries transcripts via MiNotes:
   - `minotes query "SELECT * FROM properties WHERE key='classification' AND value='business-meeting'"`
   - MiNotes UI: filter by classification, topics, speakers
   - Graph view: transcripts link to each other via shared topics/speakers

#### Config

```yaml
sync:
  minotes:
    enabled: true
    folder: "Transcripts"   # Single folder for all transcripts
    journal: true            # Add journal entries

  classify: true             # Enable classification → sets properties
  # classifications are tags, not folders
```

#### Frontmatter + Properties Example

```yaml
---
title: "Q1 Budget Review"
date: 2026-03-24
source: audioscript
classification: business-meeting
topics: [budget, Q1 results, hiring plan]
speakers: [Alice, Bob, Carol]
tags: [transcript, audioscript, business-meeting, budget, Q1 results, hiring plan]
action_items: 3
decisions: 2
duration: 2345.6
language: en
---
```

The `tags` array is the union of: `[transcript, audioscript]` + classification + topics. This makes every transcript discoverable via MiNotes search, Obsidian tag pane, or property queries.

### 3.10 Additional Transcript Sections

These sections are confirmed for implementation:

| Section | Description | When | Implementation |
|---------|-------------|------|----------------|
| **Questions Raised** | Unanswered questions from the meeting | LLM tier or regex (`?` + "does anyone know", "we need to find out", "can someone") | Rule-based: extract `?` sentences not followed by answers. LLM: semantic unanswered detection. |
| **Attendees** | Named participants with speaking stats | When diarization enabled | Speaker DB names + per-speaker stats table |
| **Timeline** | Topic shift markers with timestamps | Always (if timestamps) | Rule-based: detect speaker pattern changes + long pauses. LLM: semantic topic boundaries. |
| **Related** | Links to previous meetings on same topic/speakers | When MiNotes enabled | Query MiNotes: `WHERE key='topics' AND value LIKE '%budget%'` → `[[wikilinks]]` |
| **Raw Stats** | Word count, speaking time, silence %, WPM per speaker | Always | Computed directly from diarization segments |
| **Communication Insights** | Speaking balance, engagement, quality coaching | Always (rule-based) + LLM enhancement | See §3.11 |

### 3.11 Communication Insights — Analytics & Coaching

This is what transforms AudioScript from a transcription tool into a **communication intelligence platform**. Competitive products (Gong, Read.ai, Chorus, Fireflies) charge $15-30/user/month for these metrics. We compute them locally from the transcript.

#### Tier 1 — Rule-Based (ships with every transcript, no LLM needed)

**Speaking Balance Table:**
```markdown
## Communication Insights

### Speaking Balance
| Speaker | Talk Time | % | Words | WPM | Questions | Monologue Max |
|---------|-----------|---|-------|-----|-----------|---------------|
| Alice   | 17m 32s   | 45% | 2,540 | 145 | 35% | 1m 20s |
| Bob     | 13m 39s   | 35% | 2,457 | 180 | 8%  | 4m 12s |
| Carol   | 7m 48s    | 20% | 1,053 | 135 | 22% | 0m 45s |
```

**Metrics computed:**

| Metric | Formula | What It Reveals |
|--------|---------|-----------------|
| **Talk ratio** | `speaker_words / total_words` | Who dominated, who was silent. **#1 predictor of sales success** (Gong: 43:57 optimal) |
| **Question ratio** | `sentences_ending_? / total_sentences` per speaker | Curiosity vs. declarativeness. **Gong: 11-14 questions optimal for discovery** |
| **Longest monologue** | Max continuous speaker segment duration | **Gong: >76s loses attention. Top performers: <45s** |
| **Words per minute** | `word_count / speaking_seconds * 60` | Pace (ideal: 130-170 WPM) |
| **We/I ratio** | `count("we","our","us") / count("I","my","me")` per speaker | Collaborative vs. individual framing |
| **Interruption count** | Overlapping diarization segments < 3s after prior speaker | Turn-taking respect |
| **Speaker switches/min** | Transitions between speakers / duration | Interactivity (higher = more dialogue) |
| **Silence %** | Gaps between segments / total duration | Dead air, thinking time |
| **Participation balance** | Gini coefficient of talk percentages | 0.0 = perfectly equal, 1.0 = one person talked |
| **Response latency** | Avg gap between end of one speaker, start of next | **Top performers pause slightly longer (~1s) — signals active listening** |

**Engagement Summary (rule-based):**
```markdown
### Engagement Patterns
- **Interactivity:** 4.2 switches/min (conversational)
- **Longest monologue:** Bob, 4m 12s (budget presentation)
- **Interruptions:** 3 total (Alice→Bob: 2, Bob→Carol: 1)
- **Silence:** 12% (normal for 3-person meeting)
- **Participation:** 0.18 Gini (well-balanced)
- **We/I ratio:** Alice 3.2 (collaborative), Bob 0.8 (individual-focused)
```

#### Tier 2 — LLM-Enhanced (opt-in, high-value coaching)

**Communication Quality Score:**
```markdown
### Communication Quality (requires --summary-provider)
**Overall: 7.2/10**
- **Clarity:** 8/10 — agenda clear, decisions explicit
- **Efficiency:** 6/10 — 12 min off-topic (snack discussion at 22:00)
- **Inclusivity:** 7/10 — Carol spoke 20%, could be invited to contribute more
- **Tone:** Professional and constructive throughout
- **Active listening:** 6 instances detected (paraphrasing, building on ideas)
- **Constructiveness:** 72% solution-oriented language
```

**Growth Opportunities (personal coaching):**
```markdown
### Growth Opportunities
- **Bob:** Longest monologue was 4m 12s — try breaking into 45s segments with questions. Ask before presenting.
- **Bob:** Asked only 8% questions (team avg: 22%). Consider asking before presenting — "What are your thoughts on the budget?" before launching into slides.
- **Alice:** Strong facilitation. Improve by explicitly inviting Carol: "Carol, what's your take on this?"
- **Carol:** Clear and concise when speaking. Consider contributing earlier — your infrastructure insight at 12:45 was the most important point.
```

**Sentiment Arc:**
```markdown
### Sentiment Arc
- 00:00-05:00: Positive (0.6) — warm welcome, good energy
- 05:00-15:00: Neutral (0.1) — status updates, routine
- 15:00-22:00: Negative (-0.3) — budget tension, defensive responses
- 22:00-30:00: Positive (0.4) — brainstorm, collaborative energy
- 30:00-39:00: Neutral (0.2) — action items, wrap-up
```

**Meeting Efficiency:**
```markdown
### Meeting Efficiency
- **Outcome density:** 0.23 decisions/min (above team avg 0.15)
- **Action item clarity:** 3/5 have owners, 1/5 have deadlines → 53% clarity
- **Unresolved topics:** 2 topics ended without clear decision (flagged)
- **Recap presence:** No closing summary detected — consider adding 2-min recap
```

#### Tier 3 — Longitudinal (future, requires meeting history)

| Metric | What It Tracks | Requires |
|--------|---------------|----------|
| **Trend over time** | Is Bob's talk ratio improving? Is monologue length decreasing? | 5+ meetings per speaker |
| **Cross-meeting follow-up** | Were last meeting's action items addressed? | Linked meeting history in MiNotes |
| **Team communication score** | Team-level patterns: are meetings getting more efficient? | Aggregate across all synced meetings |
| **Personal coaching dashboard** | Per-person improvement areas tracked over weeks | MiNotes page per speaker with rolling stats |

These require MiNotes graph queries across multiple transcript pages and are a Phase 3+ feature.

### 3.12 Sales Call Intelligence (Classification: `sales-call`)

When a recording is classified as `sales-call`, AudioScript performs deep Gong-style analysis. This is the highest-value analysis tier — Gong charges $100+/user/month for this.

#### Sales Call Phase Detection

Every sales call follows a predictable structure. Detecting phases enables per-phase scoring and coaching.

```markdown
## Call Phases

| Phase | Time | Duration | Score |
|-------|------|----------|-------|
| Intro & Rapport | 00:00 - 02:30 | 2m 30s | 8/10 |
| Discovery | 02:30 - 15:00 | 12m 30s | 6/10 |
| Demo / Presentation | 15:00 - 32:00 | 17m 00s | 7/10 |
| Objection Handling | 32:00 - 38:00 | 6m 00s | 5/10 |
| Pricing | 38:00 - 43:00 | 5m 00s | 4/10 — ⚠️ discussed too early relative to value |
| Close / Next Steps | 43:00 - 45:00 | 2m 00s | 3/10 — ⚠️ no concrete next step set |
```

**Phase detection signals:**

| Phase | Detection Heuristics | LLM Enhancement |
|-------|---------------------|-----------------|
| **Intro/Rapport** | Start of call, small talk patterns ("how are you", "thanks for joining"), agenda setting ("today I'd like to") | Detect quality of rapport (genuine vs. scripted) |
| **Discovery** | Question-heavy section, prospect speaking more, "tell me about", "how do you currently", "what's your biggest challenge" | Score question quality (SPIN progression) |
| **Demo/Presentation** | Rep talk ratio spikes, feature language ("let me show you", "this feature", "what you'll see"), screen share indicators | Detect prospect engagement during demo (questions, affirmations) |
| **Objection Handling** | Prospect pushback ("I'm not sure", "what about", "our concern is", "competitor X does"), followed by rep response | Score handling quality (acknowledge → explore → address vs. steamroll) |
| **Pricing** | Money language ("cost", "price", "budget", "investment", "per seat", "annual"), numbers, discount discussion | Timing analysis (too early = risk), confidence in delivery |
| **Close/Next Steps** | End of call, "next steps", "follow up", "schedule", "send over", "let me check with my team" | Detect commitment quality (vague vs. specific date+action+owner) |

#### Sales Methodology Scoring

Score each call against established frameworks. Users pick their methodology in config:

```yaml
sync:
  sales:
    enabled: true
    methodology: meddic    # meddic | bant | spin | challenger
```

**MEDDIC Scorecard:**
```markdown
## MEDDIC Score: 7/12

| Element | Status | Evidence |
|---------|--------|----------|
| **M**etrics | ✅ Covered | "Looking to reduce onboarding time by 40%" (14:23) |
| **E**conomic Buyer | ⚠️ Identified, not engaged | "My VP makes the final call" — VP not on this call |
| **D**ecision Criteria | ✅ Covered | "We need SOC2, SSO, and API access" (22:15) |
| **D**ecision Process | ❌ Not discussed | No buying process mapped — ask next call |
| **I**dentify Pain | ✅ Strong | 3 pain points with impact quantified (see below) |
| **C**hampion | ⚠️ Possible | Prospect volunteered internal info, but no coaching behavior yet |
```

**BANT Scorecard:**
```markdown
## BANT Score: 3/4

| Element | Status | Evidence |
|---------|--------|----------|
| **B**udget | ✅ | "We've allocated $50K for this quarter" (38:10) |
| **A**uthority | ⚠️ Partial | Prospect is evaluator, not decision maker |
| **N**eed | ✅ | Strong — 3 pain points, active workarounds |
| **T**imeline | ✅ | "Need to decide by end of Q2" (41:00) |
```

**SPIN Selling Analysis:**
```markdown
## SPIN Analysis

| Question Type | Count | % | Target | Assessment |
|---------------|-------|---|--------|------------|
| **S**ituation | 8 | 42% | <25% | ⚠️ Too many fact-gathering questions |
| **P**roblem | 6 | 32% | 25-30% | ✅ Good problem exploration |
| **I**mplication | 3 | 16% | 20-25% | ⚠️ Missed opportunities to deepen pain |
| **N**eed-payoff | 2 | 10% | 20-25% | ❌ Rarely connected solution to outcomes |

**Coaching:** Reduce situation questions (you already know their stack from the previous call).
Increase implication questions — when they say "onboarding takes 3 weeks", ask
"What happens to your pipeline when a new rep can't sell for 3 weeks?"
```

#### Sales-Specific Metrics

```markdown
## Sales Signals

### Buying Signals Detected
- "When we implement this..." (ownership language, 28:15) ✅
- "Can you send the proposal to my VP too?" (multi-threading, 40:30) ✅
- "What does onboarding look like?" (implementation questions, 35:12) ✅

### Risk Signals Detected
- "We're also looking at [Competitor]" (competitor evaluation, 20:45) ⚠️
- "I need to think about it" (stall language, 42:00) ⚠️
- No concrete next step set (call ended vaguely) ❌

### Competitor Mentions
- Competitor X: mentioned 3x (prospect initiated 2x, rep 1x)
- Rep response: addressed 1/2 prospect mentions — missed opportunity at 25:30

### Objections Raised
1. "The price seems high compared to [Competitor]" (38:15)
   - Rep response: Acknowledged ✅, pivoted to ROI ✅, provided proof point ✅
   - Rating: Well handled
2. "We're not sure about the integration timeline" (33:00)
   - Rep response: Dismissed ("it's usually fine") ❌
   - Rating: Missed opportunity — should have explored the concern
```

#### Missed Opportunities to Close

```markdown
## Missed Opportunities

1. **Prospect showed high interest at 28:15** ("this is exactly what we need") — rep continued demoing instead of testing for close. Consider: "It sounds like this solves your problem — what would it take to move forward?"

2. **Budget was confirmed at 38:10** but rep didn't connect price to stated ROI metrics from discovery. Could have anchored: "You mentioned onboarding costs you $X/month — this pays for itself in 6 weeks."

3. **No next step set.** Call ended with "I'll follow up" (vague). Should have proposed: specific date, specific action, specific attendees (include the VP).
```

### 3.13 Customer Discovery Intelligence (Classification: `discovery-call`)

Different analysis for founders doing customer research. The goal isn't to close — it's to learn.

#### Discovery Call Phase Detection

```markdown
## Call Phases

| Phase | Time | Duration | Assessment |
|-------|------|----------|------------|
| Context Setting | 00:00 - 03:00 | 3m | ✅ Good framing |
| Current State | 03:00 - 12:00 | 9m | ✅ Deep exploration |
| Pain Deep Dive | 12:00 - 25:00 | 13m | ✅ Strong — 4 pain points surfaced |
| Impact Exploration | 25:00 - 30:00 | 5m | ⚠️ Could go deeper on quantification |
| Workarounds | 30:00 - 35:00 | 5m | ✅ Mapped 3 current solutions |
| Wrap-up | 35:00 - 38:00 | 3m | ⚠️ Didn't ask for referrals |
```

#### Mom Test Compliance Score

```markdown
## Mom Test Score: 7/10

### Good Questions Asked ✅
- "Tell me about the last time you had to onboard a new rep" (specific, past-tense)
- "Walk me through what happened after the deal fell through" (narrative, not hypothetical)
- "How much time did that cost your team last quarter?" (quantified impact)

### Violations ❌
- "Would you use a tool that automatically..." (hypothetical — Mom Test violation at 18:30)
- "Do you think this is a big problem?" (fishing for validation at 22:00)
- Showed product mockup at 28:00 before fully exploring their workflow (premature solutioning)

### Compliment Traps Detected ⚠️
- "That sounds amazing!" (26:15) — enthusiasm without commitment. Follow up: "What would need to be true for you to switch from your current approach?"

### Commitment Signals
- Offered to introduce their colleague (reputation commitment ✅)
- Asked "when can I try it?" (time commitment ✅)
- No financial commitment signal detected
```

#### Pain Point Extraction

```markdown
## Pain Points Discovered

### 1. New rep onboarding takes 3+ weeks (SEVERITY: HIGH)
- **Frequency:** Every new hire (8/year)
- **Impact:** "$15K/rep in lost pipeline per month of ramp"
- **Current solution:** "Senior rep shadows for 2 weeks, then we pray"
- **Emotional intensity:** High — mentioned 4 times, used word "painful"
- **Evidence:** 05:23, 14:10, 22:30, 34:15

### 2. No visibility into what reps say on calls (SEVERITY: HIGH)
- **Frequency:** Ongoing
- **Impact:** "We only find out when deals are already lost"
- **Current solution:** "Manager listens to random call recordings — maybe 2/week"
- **Emotional intensity:** Medium
- **Evidence:** 12:45, 19:00

### 3. Competitor battlecard info is outdated (SEVERITY: MEDIUM)
- **Frequency:** "Every competitive deal"
- **Impact:** "Reps make stuff up or just discount"
- **Current solution:** "Marketing updates a PDF quarterly"
- **Emotional intensity:** Low — accepted as normal (⚠️ normalized pain)
- **Evidence:** 24:30

### Hidden Opportunity
Prospect described manually tracking "what works" in a spreadsheet (30:15).
They don't realize this is a knowledge management problem — they frame it as a
"documentation problem." The real opportunity is automated best-practice capture
from winning calls.
```

#### Jobs to Be Done Extraction

```markdown
## Jobs to Be Done

### Primary Job
**When** we hire a new sales rep,
**I want to** get them productive in under 1 week,
**so I can** hit quarterly targets without the pipeline gap.

### Secondary Jobs
- **When** I review pipeline, **I want to** know which deals are at risk based on what reps actually said, **so I can** intervene before it's too late.
- **When** a rep faces a competitor, **I want to** give them current, specific counter-arguments, **so I can** win on value not discounting.

### Switching Triggers
- "The last straw was when our best rep lost a deal because they didn't know [Competitor] launched a new feature" (24:45)
```

#### Discovery vs Sales — Config

```yaml
sync:
  sources:
    - path: "C:\\Users\\cklau\\OneDrive\\Documents\\Sound Recordings"
      classify: true
      # Classification auto-detects and applies the right analysis:
      # sales-call → MEDDIC/BANT/SPIN scoring, pipeline signals, close opportunities
      # discovery-call → Mom Test, JTBD, pain extraction, insight mapping
      # business-meeting → action items, decisions, communication insights
      # voice-memo → simple transcript + summary

  # Sales-specific config (applies when classification = sales-call)
  sales:
    enabled: true
    methodology: meddic         # meddic | bant | spin | challenger
    track_competitors: [Gong, Chorus, Fireflies]  # Named competitors to watch
    close_coaching: true        # Flag missed opportunities to close

  # Discovery-specific config (applies when classification = discovery-call)
  discovery:
    enabled: true
    framework: mom_test         # mom_test | jtbd | problem_solution
    pain_extraction: true       # Extract and rank pain points
    insight_mapping: true       # Map insights across multiple calls
```

### 3.14 Deal Tracking Across Calls (Phase 3+)

Link related sales/discovery calls about the same deal/prospect for longitudinal analysis.

**Linking signals (no CRM needed):**
- Same speaker names (from speaker DB)
- Same company name mentioned (NER extraction)
- Explicit references: "as we discussed last time", "following up on"
- Same topics/pain points across calls

**Deal-level page in MiNotes:**
```markdown
---
title: "Deal: Acme Corp"
classification: deal-tracker
tags: [deal, acme-corp, sales-pipeline]
stage: negotiation
contacts: [Sarah (Champion), Mike (Economic Buyer)]
calls: 4
first_contact: 2026-02-15
last_contact: 2026-03-20
---

## Deal Timeline
- [[2026-02-15 Acme Discovery]] — Initial discovery, 3 pain points identified
- [[2026-03-01 Acme Demo]] — Demo, strong interest, competitor mentioned
- [[2026-03-10 Acme Technical]] — Technical deep dive with engineering team
- [[2026-03-20 Acme Negotiation]] — Pricing discussion, pending VP approval

## MEDDIC Coverage (Cumulative)
| Element | Call 1 | Call 2 | Call 3 | Call 4 | Status |
|---------|--------|--------|--------|--------|--------|
| Metrics | ✅ | | | | Covered |
| Econ Buyer | | | | ⚠️ | Named but not engaged |
| Decision Criteria | | ✅ | ✅ | | Covered |
| Decision Process | | | | ❌ | NOT DISCUSSED |
| Pain | ✅ | ✅ | | | Strong |
| Champion | | ⚠️ | ✅ | ✅ | Sarah confirmed |

## Deal Health: 7/10
- ✅ Multi-threaded (3 contacts across 4 calls)
- ✅ Champion identified (Sarah)
- ⚠️ Economic buyer not on any call yet
- ❌ Decision process not mapped — critical gap
- ⚠️ 10 days since last contact — momentum slowing
```

This is Phase 3+ because it requires cross-page linking in MiNotes and entity resolution across transcripts.

#### Config



```yaml
sync:
  # Communication insights (data-backed metrics only)
  insights:
    enabled: true              # Compute rule-based metrics (Tier 1)
    speaking_balance: true     # Per-speaker stats table (talk ratio, questions, monologue)
    engagement: true           # Engagement patterns section (switches/min, silence, balance)
    # LLM-enhanced (Tier 2, requires summary_provider)
    quality_score: false       # Communication quality rating
    growth_tips: false         # Personal coaching suggestions
    sentiment: false           # Sentiment arc across meeting
    efficiency: false          # Meeting efficiency metrics
```

#### Implementation Notes

**All Tier 1 metrics are computable from:**
- Diarization segments (start, end, speaker) — already produced by our pipeline
- Transcript text per segment — already in the result dict
- No external dependencies needed

All Tier 1 metrics are computed purely from diarization segments (start, end, speaker) and transcript text. ~0ms overhead per file.

---

## 4. Architecture

```
audioscript sync [--watch] [--source PATH]
       │
       ▼
  ┌──────────────────────────────────────────────────┐
  │ SyncEngine                                        │
  │                                                    │
  │  1. Resolve     ← WSL path translation             │
  │  2. Scan        ← Glob + extensions + filters      │
  │  3. Diff        ← SHA-256 vs manifest + mtime cache│
  │  4. Process     ← AudioProcessor (reuse existing)  │
  │  5. Classify    ← MeetingClassifier (keyword/LLM)  │
  │  6. Analyze     ← MeetingAnalyzer (sections)       │
  │  7. Format      ← MarkdownFormatter (full page)    │
  │  8. Tag         ← Properties: classification,      │
  │                    topics, tags (multi-select)      │
  │  9. Export      ← MiNotesExporter (reuse existing) │
  │ 10. Report      ← emit() / emit_ndjson()           │
  └──────────────────────────────────────────────────┘
       │
       ▼ (if --watch)
  sleep(poll_interval) → loop back to step 2
```

### New Files

| File | Purpose |
|------|---------|
| `audioscript/sync/__init__.py` | Sync package |
| `audioscript/sync/engine.py` | `SyncEngine` class — scan, diff, process, analyze loop |
| `audioscript/sync/wsl.py` | WSL detection and path translation |
| `audioscript/sync/discovery.py` | File discovery, filtering, mtime cache |
| `audioscript/intelligence/__init__.py` | Meeting intelligence package |
| `audioscript/intelligence/classifier.py` | `MeetingClassifier` — keyword/LLM classification |
| `audioscript/intelligence/analyzer.py` | `MeetingAnalyzer` — extract summaries, actions, decisions, questions |
| `audioscript/intelligence/tagger.py` | `PropertyTagger` — classification + topics → MiNotes properties & frontmatter tags |
| `audioscript/intelligence/communication.py` | `CommunicationAnalyzer` — talk ratio, questions, monologue length, engagement metrics |
| `audioscript/intelligence/sales.py` | `SalesAnalyzer` — phase detection, methodology scoring, buying/risk signals, missed opportunities |
| `audioscript/intelligence/discovery.py` | `DiscoveryAnalyzer` — Mom Test, JTBD extraction, pain points, insight mapping |
| `audioscript/intelligence/llm_insights.py` | LLM-based quality scoring, coaching, sentiment (opt-in Tier 2) |
| `audioscript/cli/commands/sync_cmd.py` | `audioscript sync` CLI command |
| `tests/test_sync_engine.py` | Sync engine tests |
| `tests/test_wsl.py` | WSL path translation tests |
| `tests/test_sync_discovery.py` | File discovery tests |
| `tests/test_classifier.py` | Meeting classification tests |
| `tests/test_analyzer.py` | Meeting analysis / section extraction tests |
| `tests/test_tagger.py` | Property tagging tests |
| `tests/test_communication.py` | Communication analytics tests (Tier 1 metrics) |
| `tests/test_sales.py` | Sales call analysis tests (phases, methodology, signals) |
| `tests/test_discovery.py` | Discovery call analysis tests (Mom Test, JTBD, pain points) |
| `tests/test_llm_insights.py` | LLM insights tests (mocked API calls) |

### Modified Files

| File | Changes |
|------|---------|
| `config/settings.py` | Add `SyncConfig`, `SyncSourceConfig`, `SyncMiNotesConfig`, `WSLConfig`, `ClassificationConfig` models |
| `cli/main.py` | Register `sync` subcommand |
| `formatters/markdown_formatter.py` | Extend `render_markdown` with classification, actions, decisions, insights sections |
| `.audioscript.yaml` | Add `sync:` section example |

---

## 5. WSL Path Module Specification

```python
# audioscript/sync/wsl.py

def is_wsl() -> bool:
    """Detect if running inside WSL. Cached after first call."""

def translate_path(path: str, mount_prefix: str = "/mnt/") -> str:
    """Translate a Windows path to WSL path if needed.

    - If not WSL or path is already a Unix path: return as-is
    - If wslpath available: use subprocess
    - Fallback: regex C:\\... → /mnt/c/...
    - Handles forward and backslashes
    - Preserves spaces and special characters
    """

def resolve_sync_path(path: str, wsl_config: WSLConfig) -> Path:
    """Full resolution: translate + validate + return Path.

    Raises PathValidationError with WSL-specific hints on failure.
    """
```

---

## 6. User Workflows

### Workflow 1: First-time setup
```bash
# User adds sync config to .audioscript.yaml:
#   sync:
#     sources:
#       - path: "C:\\Users\\cklau\\OneDrive\\Documents\\Sound Recordings"

# First sync — transcribes all 219 files
audioscript sync
# → Scanned 219 files, 219 new, transcribing...
# → [batch 1/22] Transcribing 10 files...
# → ...
# → Sync complete: 219 transcribed, 0 failed

# Second sync — nothing new
audioscript sync
# → Scanned 219 files, 0 new, 219 skipped. Up to date.
```

### Workflow 2: Ongoing sync with MiNotes
```bash
# Config has minotes.enabled: true
audioscript sync --watch
# → [14:30] Scan: 0 new (219 skipped)
# → [14:35] Scan: 2 new files found
# →   Transcribing Recording (220).m4a... done (1m 05s)
# →   Transcribing Recording (221).m4a... done (0m 32s)
# →   Exported to MiNotes: 2 transcripts
# →   Journal: "Synced 2 recordings from Sound Recordings"
# → [14:40] Scan: 0 new
```

### Workflow 3: Quick one-off sync with Windows path
```bash
audioscript sync --source "C:\Users\cklau\Downloads\meetings" --tier high_quality --diarize
# → Translated WSL path: C:\Users\cklau\Downloads\meetings → /mnt/c/Users/cklau/Downloads/meetings
# → Scanned 3 files, 3 new, transcribing...
```

---

## 7. Open Questions

### Resolved

| # | Question | Decision |
|---|----------|----------|
| 1 | **Initial bulk sync:** Transcribe all or require `--force`? | Transcribe all by default — manifest handles dedup. Ctrl+C resumes on next sync. |
| 2 | **Watch mode:** Poll-based or filesystem events? | Poll-based. `inotify` doesn't work across WSL/Windows boundary. |
| 3 | **Batch size default:** | 10 in watch mode, unlimited in one-shot. Configurable. |
| 4 | **OneDrive placeholders:** | Check `file_size > min_file_size` + catch `PermissionError`. Skip with warning. |
| 5 | **Multiple sources:** | Sequential per source, parallel within source (reuse `--concurrency`). |
| 6 | **Config reload in watch mode:** | Yes — re-read config each poll cycle. |
| 7 | **Local save + MiNotes:** | Both. Local markdown/JSON is source of truth, MiNotes is an export copy. |
| 8 | **Timestamps:** | Yes, per-speaker-turn timestamps in `### Speaker — HH:MM:SS` format. |
| 9 | **Structured sections:** | Full meeting intelligence page: summary, action items, decisions, insights, follow-ups, classification. |
| 10 | **Project folder routing:** | Classification-based → MiNotes folder mapping. Per-source override available. |
| 11 | **Meeting classification:** | Taxonomy: business-meeting, family, brainstorm, one-on-one, interview, lecture, voice-memo, call, other. Keyword-based default, LLM opt-in. |

### Still Open

| # | Question | Options |
|---|----------|---------|
| 12 | **Custom classification taxonomy:** Should users be able to add their own types beyond the defaults (e.g., "standup", "retro", "therapy session")? | A: Yes, fully extensible via config. B: Fixed taxonomy only. **Rec: A** |
| 13 | **Action item format:** Should extracted action items be MiNotes-native task blocks (`- [ ] text`) or plain text? | A: Checkbox markdown (portable). B: MiNotes task cards (richer, MiNotes-specific). **Rec: A for markdown, cards as future enhancement** |
| 14 | **Re-classification:** If classification was wrong, can user override? How persist? | A: Manual property edit in MiNotes (preserved via skip-if-exists). B: Override file. **Rec: A** |
| 15 | **LLM cost management:** 219 files bulk sync with LLM = significant API cost. | A: Prompt before bulk. B: `--llm-batch-limit N`. C: LLM only for new files in watch-mode, rule-based for bulk. **Rec: C** |
| 16 | **Title generation:** Auto-generate from content ("Q1 Budget Review") instead of "Recording (215)"? | A: LLM-generated. B: First sentence. C: Keep filename. **Rec: A with LLM, B as fallback** |
| 17 | **Topic extraction depth:** How many topics per transcript? Free-form or from a controlled vocabulary? | A: Free-form, LLM/keyword extracted (max 5). B: Controlled vocabulary from config. **Rec: A — multi-select properties support free-form** |
| 18 | **Podcast detection:** Added "podcast" to classification. Should podcast transcripts get a "show notes" section instead of "action items"? | A: Yes, classification-specific sections. B: Same sections for all types. **Rec: A for v2, B for now** |

---

## 8. Milestones

| Phase | Scope | Estimate |
|-------|-------|----------|
| **8a** | WSL path module + tests | Small |
| **8b** | File discovery + diff (mtime cache, extension filter) | Small |
| **8c** | Config models (SyncConfig, ClassificationConfig, etc.) | Small |
| **8d** | SyncEngine (one-shot mode) + CLI command | Medium |
| **8e** | Meeting classifier (rule-based) + property tagger | Medium |
| **8f** | Meeting analyzer — rule-based section extraction (summary, actions, decisions, questions) | Medium |
| **8g** | Communication analyzer — Tier 1 rule-based metrics (talk ratio, questions, monologue, engagement) | Medium |
| **8h** | Extended markdown formatter (all sections including communication insights) | Medium |
| **8i** | Watch mode (poll loop) | Small |
| **8j** | MiNotes sync integration with property-based classification | Small (reuses F-01) |
| **8k** | Sales call analyzer — phase detection, MEDDIC/BANT/SPIN scoring, buying/risk signals, missed close opportunities | Medium |
| **8l** | Discovery call analyzer — Mom Test scoring, JTBD extraction, pain point extraction & ranking, hidden opportunity detection | Medium |
| **8m** | LLM-based Tier 2 (quality score, coaching, sentiment, efficiency, Challenger scoring) | Medium |
| **8n** | Deal tracker — cross-call linking, cumulative MEDDIC, deal health page (Phase 3+) | Large |
| **8o** | Initial bulk sync for user's 219 recordings | Integration test |

---

## 9. Success Criteria

- `audioscript sync --source "C:\Users\cklau\OneDrive\Documents\Sound Recordings"` transcribes all 219 files on first run
- Second run: 0 new, 219 skipped, completes in <5 seconds (mtime cache)
- New files added to OneDrive auto-detected on next sync cycle
- `--watch` mode runs indefinitely, detects new files within poll_interval
- Each transcript page has: metadata, summary, action items, decisions, transcript with timestamps
- Classification correctly routes business meetings → "Work/Meetings", voice memos → "Notes/Voice Memos"
- MiNotes pages created in correct project folders with full properties
- WSL path translation works with spaces, OneDrive paths, and custom mount prefixes
- Rule-based analysis produces usable summaries and action items without LLM
- LLM tier (when configured) produces significantly richer output with insights and follow-up links
