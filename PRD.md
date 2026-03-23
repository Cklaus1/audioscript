# AudioScript — Product Requirements Document (PRD)

**Version:** 1.0
**Date:** 2026-03-23
**Author:** Product & Engineering
**Status:** Draft

---

## 1. Executive Summary

AudioScript is a local-first, CLI-driven audio transcription tool built on OpenAI Whisper with optional speaker diarization via pyannote-audio. It targets developers, knowledge workers, and AI agents who need batch audio-to-text pipelines with structured output, privacy guarantees, and integration into note-taking workflows.

**Vision:** The composable, open-source CLI pipeline that chains transcription + diarization + alignment + hallucination filtering + multi-format output — a gap no single OSS tool fills end-to-end today.

**Primary integration target:** [MiNotes](../MiNotes) — a local-first knowledge management engine (Rust/SQLite) with block-based outlining, bidirectional linking, and markdown import/export.

---

## 2. Target Users

| Persona | Description | Primary Need |
|---------|-------------|--------------|
| **Developer / Power User** | Uses CLI tools, scripts pipelines, values local-first | Batch transcription with structured JSON/NDJSON output |
| **Knowledge Worker** | Records meetings, lectures, interviews | Searchable, speaker-labeled transcripts in their note system |
| **AI Agent** | LLM-based automation (Claude Code, etc.) | Machine-readable output, validation, dry-run, pipe mode |
| **Content Creator** | Podcasters, YouTubers, journalists | Subtitle generation (SRT/VTT) with accurate timestamps |
| **Researcher** | Qualitative interviews, oral histories | Speaker diarization + searchable archives |

---

## 3. Current State Assessment

### 3.1 What Works

- Typer CLI with 7 subcommands (transcribe, diarize, detect-language, vad, schema, status, check)
- Pydantic config with YAML + env var + CLI arg merge hierarchy
- Agent-friendly design: `--pipe`, `--format`, `--fields`, `--dry-run`, path injection protection
- SHA-256 content-addressed dedup + atomic manifest + checkpoint retry
- 3-tier model selection (draft/balanced/high_quality)
- Lazy initialization of Whisper and pyannote models
- Structured output (JSON, YAML, table, NDJSON)
- Rich stderr UI with progress bars
- Metadata extraction via ffprobe + mutagen

### 3.2 Known Gaps & Placeholder Implementations

| ID | Gap | Severity | Current State |
|----|-----|----------|---------------|
| G-01 | `clean_audio()` is a no-op | High | Just copies the file; `--clean-audio` flag is misleading |
| G-02 | `generate_summary()` returns first 25 words | High | Extractive placeholder; not useful for real summaries |
| G-03 | `--concurrency` defined but not implemented | Medium | Always sequential; no parallelism |
| G-04 | Thread-based timeout can't kill worker | Medium | Resource leak risk on timeout |
| G-05 | No markdown output format | Medium | Only json/srt/vtt/tsv/txt supported |
| G-06 | No MiNotes integration | High | No export path to MiNotes or any note system |
| G-07 | No confidence scores exposed | Medium | Whisper produces token probabilities but they're discarded |
| G-08 | No hallucination detection beyond VAD | High | Single-layer defense; repetitions and phantom text not caught |
| G-09 | No real-time/streaming mode | Medium | Batch-only processing |
| G-10 | `mutagen` used but not in core dependencies | Low | Only in pyproject.toml indirectly; fails silently |
| G-11 | Manifest is single JSON file | Low | No archival/rotation; will bloat at scale |
| G-12 | No distinction between transient vs permanent errors | Medium | All errors retried equally |
| G-13 | Vanilla Whisper only — no faster-whisper/WhisperX backend | High | Misses 4-6x speed gains and better alignment |
| G-14 | No word-level forced alignment | Medium | Word timestamps from Whisper are approximate |
| G-15 | No translation support | Low | Whisper supports any→English but not exposed |

---

## 4. Feature Requirements

### 4.1 P0 — Must Have (v0.2)

#### F-01: MiNotes Export Integration
Export transcription results directly into MiNotes as structured pages with blocks.

**Requirements:**
- `--export minotes` flag on `transcribe` command
- Create a MiNotes page per audio file with structured content:
  - Page title: `Transcript: {filename} ({date})`
  - Properties: `source_file`, `duration`, `language`, `model`, `tier`, `speakers` (via MiNotes property system)
  - Blocks: One block per speaker turn (or per segment if no diarization)
  - Block format: `**[Speaker N]** (00:01:23 → 00:02:45): Transcribed text here`
  - Nested blocks for word-level detail (optional, via `--word-blocks`)
- Use MiNotes CLI (`minotes page create`, `minotes block create`) for data operations
- Markdown sync as bulk content path (`--output-format markdown` to sync directory)
- Support `--minotes-db` to specify database path (default: `~/.minotes/default.db`)
- **Plugin registration:** On first export, auto-register AudioScript as a MiNotes plugin:
  - `minotes plugin register --name audioscript --version 0.2.0 --description "Audio transcription integration"`
  - All events logged with actor `plugin:audioscript` for audit trail
  - Use plugin persistent storage for sync state (last exported file hash, export timestamp)
- **Property class:** Register `transcript` class in MiNotes with typed schema (source_file, duration, language, model, tier, speakers, confidence_avg, transcribed_at)
- Bidirectional linking: auto-link related transcripts (same speaker, same day, same project folder)
- Journal integration: if `--journal` flag, append transcript summary to today's MiNotes journal page
- **Event subscription (F-20):** Subscribe to MiNotes events for auto-transcription triggers

**Acceptance Criteria:**
- `audioscript transcribe --input meeting.mp3 --export minotes` creates a fully structured page in MiNotes
- AudioScript appears in `minotes plugin list` as registered plugin
- All created blocks/pages show `plugin:audioscript` as actor in `minotes events`
- Plugin storage contains sync state queryable via `minotes plugin storage get audioscript <key>`
- Page is searchable via MiNotes FTS
- Properties are queryable via MiNotes SQL interface
- Re-running with same file skips export (unless `--force`), checked via plugin storage state

---

#### F-02: Markdown Output Format
Native markdown export with speaker labels, timestamps, and structured headings.

**Requirements:**
- `--output-format markdown` (or `md`) added to transcribe command
- Output structure:
  ```markdown
  ---
  title: "Transcript: meeting-2026-03-23.mp3"
  date: 2026-03-23
  duration: "01:23:45"
  language: en
  model: large-v3
  speakers: 3
  ---

  # Transcript: meeting-2026-03-23.mp3

  ## Metadata
  | Field | Value |
  |-------|-------|
  | Duration | 01:23:45 |
  | Language | English |
  | Model | large-v3 (high_quality) |
  | Speakers | 3 identified |

  ## Transcript

  ### Speaker 1 — 00:00:00
  Welcome everyone to the meeting. Today we'll discuss the Q1 results.

  ### Speaker 2 — 00:00:15
  Thanks. I've prepared the slides for the revenue section.

  ## Summary
  Brief meeting discussing Q1 results...
  ```
- YAML frontmatter compatible with MiNotes markdown import
- Speaker names from speaker DB used when available
- Configurable: `--md-heading-level` (h2/h3 per speaker), `--md-timestamps` (inline/heading/none)
- Pipe-compatible: markdown streamed to stdout in `--pipe` mode

---

#### F-03: Faster-Whisper Backend
Replace or supplement vanilla Whisper with faster-whisper for 4-6x speed improvement and reduced VRAM.

**Requirements:**
- Add `faster-whisper` as preferred transcription backend
- Maintain vanilla Whisper as fallback (`--backend whisper|faster-whisper`, auto-detect default)
- Expose CTranslate2 optimizations: int8/float16 quantization, batched inference
- Expose per-token log probabilities for confidence scoring (F-06)
- Built-in Silero VAD integration (faster-whisper ships it)
- Preserve all existing Whisper options (temperature fallback, beam search, etc.)
- Config: `backend: faster-whisper` in `.audioscript.yaml`

**Acceptance Criteria:**
- Default backend is faster-whisper when installed
- `audioscript transcribe --input test.mp3 --backend faster-whisper` completes in <25% of vanilla Whisper time
- All existing tests pass with both backends
- `audioscript check` reports which backends are available

---

#### F-04: Implement Real Audio Cleaning
Replace the placeholder `clean_audio()` with actual noise reduction.

**Requirements:**
- Integrate `noisereduce` library (or `demucs` for higher quality)
- Pipeline: input → noise profile estimation → spectral gating → cleaned output
- Preserve original file; write cleaned version to temp directory
- Configurable aggressiveness: `--clean-level light|moderate|aggressive`
- Skip cleaning if audio SNR is above threshold (avoid degrading clean audio)
- Log before/after SNR metrics

**Acceptance Criteria:**
- `--clean-audio` measurably improves WER on noisy test samples
- Clean audio doesn't degrade WER on already-clean samples
- Processing time overhead <20% of transcription time

---

#### F-05: Implement Real Summarization
Replace the first-25-words placeholder with LLM-based or extractive summarization.

**Requirements:**
- **Tier 1 (local, no API):** Extractive summarization using sentence scoring (TextRank or similar)
  - Extract key sentences weighted by: position, speaker diversity, keyword density
  - Configurable length: `--summary-length short|medium|long` (1/3/5 paragraphs)
- **Tier 2 (optional, with API key):** LLM-based abstractive summarization
  - Support Claude API or OpenAI API via `--summary-provider claude|openai`
  - Generate: summary, key topics, action items, decisions
  - Structured output in JSON for agent consumption
- **Meeting mode** (`--shortcut +meeting`): Auto-extract action items, decisions, attendees
- Summary included in markdown and MiNotes export

---

#### F-06: Confidence Scores & Hallucination Detection
Multi-layer defense against Whisper hallucinations.

**Requirements:**
- **Layer 1 — VAD preprocessing:** Silero VAD (already partially implemented via faster-whisper)
  - Strip non-speech segments before inference
  - Configurable: `--vad-threshold 0.5` (speech probability cutoff)
- **Layer 2 — Confidence scoring:**
  - Expose per-segment and per-word confidence scores (from token log probabilities)
  - Add `confidence` field to JSON output (0.0–1.0)
  - Flag low-confidence segments: `--min-confidence 0.6`
  - In markdown output: low-confidence text marked with `[?]` prefix
- **Layer 3 — Repetition detection:**
  - Detect repeated phrases (n-gram overlap >80% across consecutive segments)
  - Detect looping (same text repeating >3 times)
  - Auto-filter or flag: `--hallucination-filter auto|flag|off`
- **Layer 4 — Energy validation:**
  - Cross-check transcript segments against audio energy
  - Flag text generated from near-silence segments
- Output: `hallucination_risk` field per segment (low/medium/high)

---

### 4.2 P1 — Should Have (v0.3)

#### F-07: WhisperX Integration
Pipeline integration for word-level forced alignment and improved diarization.

**Requirements:**
- Add WhisperX as optional backend: `--backend whisperx`
- Benefits: phoneme-level alignment, better word timestamps, integrated diarization
- Forced alignment post-processing: snap word boundaries to phoneme boundaries
- Improved speaker assignment accuracy via aligned word timestamps
- Dependency: `pip install audioscript[whisperx]`

---

#### F-08: Real-Time Streaming Mode
Live transcription from microphone or audio stream.

**Requirements:**
- `audioscript stream` command
- Input sources: microphone (`--device default`), audio file (simulated stream), pipe (`stdin`)
- Output: NDJSON to stdout (one segment per line, incrementally)
- Latency target: <2 seconds end-to-end
- Integration: `audioscript stream --export minotes` appends blocks in real-time to a MiNotes journal page
- Backend: faster-whisper with chunked inference or whisper.cpp streaming
- Graceful shutdown on Ctrl+C with final flush

---

#### F-09: Parallel / Concurrent Processing
Implement the `--concurrency` flag that's currently a no-op.

**Requirements:**
- Process multiple files in parallel using `multiprocessing` (not threads — for true GPU multiplexing or CPU parallelism)
- `--concurrency N` controls worker count (default: 1)
- GPU-aware scheduling: if single GPU, serialize model inference but parallelize I/O and post-processing
- Progress: per-file progress bars in Rich UI; per-file NDJSON events in pipe mode
- Manifest updates remain atomic (file-level locking)
- Memory-aware: auto-reduce concurrency if available memory is low

---

#### F-10: Translation Support
Expose Whisper's any→English translation capability and extend it.

**Requirements:**
- `--translate` flag on transcribe command
- `--translate-to en` (Whisper native: any→English)
- For other target languages: optional integration with external translation (e.g., `argos-translate` for offline, or cloud API)
- Dual output: original language + translated text side-by-side in markdown/JSON
- In MiNotes export: bilingual blocks with original as nested child block

---

#### F-11: Custom Vocabulary / Hot Words
Domain-specific term boosting for improved accuracy.

**Requirements:**
- `--vocabulary ./terms.txt` flag: one term per line
- `--hot-words "Kubernetes,gRPC,AudioScript"` inline
- Implementation: initial prompt injection (Whisper technique) + post-processing correction
- With faster-whisper: use `hotwords` parameter (native support)
- Vocabulary profiles: `--vocabulary-profile medical|legal|tech` for bundled term lists
- Persist in config: `vocabulary_file: ./terms.txt` in `.audioscript.yaml`

---

#### F-12: Export to Additional Targets
Beyond MiNotes and markdown — support common knowledge tools.

**Requirements:**
- `--export obsidian` — write markdown file to Obsidian vault path with wiki-link conventions
- `--export notion` — push via Notion API (requires `--notion-token`)
- `--export json-ld` — structured linked data for knowledge graphs
- `--export clipboard` — copy transcript to system clipboard
- Plugin architecture: `audioscript.exporters` entry point for third-party exporters
- Each exporter is optional dependency: `pip install audioscript[notion]`

---

### 4.3 P2 — Nice to Have (v0.4+)

#### F-13: Emotion / Tone Detection
Detect speaker sentiment and emotional tone per segment.

**Requirements:**
- Classify segments: neutral, positive, negative, excited, frustrated, questioning
- Audio-based (prosody, pitch, energy) + text-based (sentiment analysis)
- Output: `emotion` field per segment in JSON; emoji annotations in markdown (opt-in)
- Use case: meeting retrospectives, interview analysis, customer call QA

---

#### F-14: Accessibility Compliance Output
Auto-generate WCAG 2.1 AA compliant caption files.

**Requirements:**
- WebVTT with positioning, styling, and region metadata
- TTML output for broadcast compliance
- Caption quality metrics: reading speed (WPM), line length, display duration
- Auto-split long captions to meet accessibility guidelines
- `audioscript transcribe --output-format webvtt-accessible` with auto-validation

---

#### F-15: Swappable Model Backends
Support models beyond the Whisper family.

**Requirements:**
- Abstract transcription interface: `TranscriberBackend` protocol
- Implementations: `WhisperBackend`, `FasterWhisperBackend`, `WhisperXBackend`, `NemoBackend`
- Future: `CanaryBackend` (NVIDIA, best WER), `ParakeetBackend` (NVIDIA, fastest)
- Model registry: `audioscript schema models` lists all available backends and their installed models
- Config: `backend: nemo` or auto-select based on task (speed vs accuracy)

---

#### F-16: Intelligent Retry Classification
Distinguish transient from permanent errors for smarter retry behavior.

**Requirements:**
- Classify errors: `transient` (OOM, timeout, GPU busy) vs `permanent` (corrupt file, unsupported codec, too short)
- Only retry transient errors; immediately fail permanent errors with actionable message
- Adaptive backoff: shorter for OOM (may free after GC), longer for timeout
- `--retry-strategy smart|always|never` (default: smart)

---

#### F-17: Manifest Management
Address manifest bloat and lifecycle.

**Requirements:**
- `audioscript manifest prune --older-than 90d` — remove old entries
- `audioscript manifest export --format csv` — for analysis
- Manifest rotation: auto-archive when file exceeds 10MB
- Per-project manifests: store manifest alongside output (not global)
- Migration path from v1.0 format to future versions

---

#### F-18: MiNotes Bidirectional Sync
Full two-way sync between AudioScript outputs and MiNotes.

**Requirements:**
- Use MiNotes `sync-dir` capability to watch an AudioScript output directory
- Auto-import new transcripts as they complete
- Edits in MiNotes (speaker name corrections, text fixes) are preserved on re-transcription
- Conflict resolution: MiNotes edits take precedence over re-generated transcripts
- Event-driven: MiNotes fires `page.updated` events that AudioScript can subscribe to

---

## 5. Technical Architecture

### 5.1 Backend Abstraction

```
┌─────────────────────────────────────────────────┐
│                  CLI Layer (Typer)                │
│  transcribe | diarize | stream | export | ...    │
├─────────────────────────────────────────────────┤
│               Pipeline Orchestrator              │
│  input → validate → clean → VAD → transcribe →   │
│  align → diarize → hallucination_filter →         │
│  format → export                                  │
├────────────┬────────────┬───────────────────────┤
│ Backends   │ Exporters  │ Post-processors       │
│ ┌────────┐ │ ┌────────┐ │ ┌──────────────────┐  │
│ │Whisper │ │ │MiNotes │ │ │Hallucination Det.│  │
│ │Faster-W│ │ │Markdown│ │ │Confidence Score  │  │
│ │WhisperX│ │ │Obsidian│ │ │Repetition Filter │  │
│ │NeMo    │ │ │Notion  │ │ │Energy Validation │  │
│ │Canary  │ │ │SRT/VTT │ │ │Summary Generator │  │
│ └────────┘ │ └────────┘ │ └──────────────────┘  │
├────────────┴────────────┴───────────────────────┤
│              Config + State Layer                 │
│  Pydantic settings | Manifest | Checkpoints       │
└─────────────────────────────────────────────────┘
```

### 5.2 MiNotes Integration Architecture

```
AudioScript (registered as plugin:audioscript)
┌──────────────┐                         MiNotes
│  transcribe  │                       ┌──────────────┐
│   result     │── markdown sync ────→ │  sync-dir    │
│   (JSON)     │                       │              │
│              │── CLI commands ─────→ │  minotes CLI │
│              │   page/block/property │              │
│              │   actor: plugin:audio │  SQLite DB   │
│              │                       │  (WAL mode)  │
│  auto-trigger│←─ events --follow ── │  event log   │
│   (F-20)     │   page.created, etc. │              │
│              │                       │              │
│  sync state  │── plugin storage ──→ │  plugin_storage│
└──────────────┘                       └──────────────┘

Integration method (DECIDED):
1. Register as MiNotes plugin (actor attribution, persistent storage, event subscription)
2. Primary content: Markdown sync — write .md files → MiNotes sync-dir auto-imports
3. Metadata: CLI shelling — `minotes` commands for properties, classes, links
4. State: Plugin storage for sync state (last hash, timestamps, export status)
5. Automation: Event subscription for auto-transcription triggers (F-20)
```

### 5.3 Markdown Export Schema

For MiNotes compatibility, markdown files must follow:
- YAML frontmatter with `title`, `date`, and custom properties
- Bullet-list blocks (MiNotes parses `-`, `*`, `+` as blocks)
- `[[wiki links]]` for cross-referencing related transcripts
- One file per transcript in a designated sync directory

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Metric | Target | Current |
|--------|--------|---------|
| Transcription speed (faster-whisper, GPU) | >50x real-time | ~1x (vanilla Whisper) |
| Transcription speed (faster-whisper, CPU) | >4x real-time | ~0.3x (vanilla Whisper) |
| Streaming latency | <2s end-to-end | N/A (not supported) |
| Batch throughput (10 files, GPU) | <5min for 10hrs audio | ~10hrs (sequential) |
| MiNotes export overhead | <500ms per transcript | N/A |
| Memory (large-v3, int8) | <5GB VRAM | ~10GB (fp16) |

### 6.2 Reliability

- Atomic manifest writes (already implemented)
- Checkpoint recovery for interrupted transcriptions (already implemented)
- Graceful degradation: if faster-whisper unavailable, fall back to vanilla Whisper
- No data loss on crash: temp files cleaned up, partial results preserved in checkpoints

### 6.3 Privacy & Security

- **Local-first:** All audio processed on-device by default; no data leaves the machine
- **No telemetry:** Zero analytics, crash reporting, or usage tracking
- **Path validation:** Injection/traversal protection on all file inputs (already implemented)
- **Optional cloud:** LLM summarization and translation are opt-in with explicit API key configuration
- **Credential handling:** HuggingFace tokens and API keys read from env vars or config, never logged

### 6.4 Compatibility

- Python >=3.11
- OS: Linux, macOS, Windows (WSL)
- GPU: NVIDIA CUDA (primary), Apple MPS (best-effort), CPU fallback
- Audio formats: All ffmpeg-supported formats (mp3, wav, flac, m4a, ogg, webm, etc.)
- MiNotes: Compatible with MiNotes v0.1+ markdown import and CLI

---

## 7. MiNotes Integration Specification

### 7.1 Page Structure

When exporting to MiNotes, AudioScript creates the following structure:

```
Page: "Transcript: meeting-2026-03-23.mp3"
├── Properties:
│   ├── source_file: "/path/to/meeting-2026-03-23.mp3"
│   ├── duration: "01:23:45"
│   ├── language: "en"
│   ├── model: "large-v3"
│   ├── tier: "high_quality"
│   ├── speakers: 3
│   ├── transcribed_at: "2026-03-23T14:30:00Z"
│   ├── confidence_avg: 0.87
│   └── class: "transcript"
├── Blocks:
│   ├── "## Summary"
│   │   └── "Three participants discussed Q1 results..."
│   ├── "## Transcript"
│   │   ├── "**Alice** (00:00:00 → 00:00:15): Welcome everyone..."
│   │   ├── "**Bob** (00:00:15 → 00:01:02): Thanks. I've prepared..."
│   │   └── "**Alice** (00:01:02 → 00:01:30): Great, let's start..."
│   └── "## Action Items"
│       ├── "- [ ] Bob to send revenue slides by Friday"
│       └── "- [ ] Alice to schedule follow-up"
└── Links:
    ├── [[Daily Journal 2026-03-23]]  (auto-linked)
    └── [[Project: Q1 Review]]        (if folder context provided)
```

### 7.2 Class & Property Schema

AudioScript should register a `transcript` class in MiNotes:

```
Class: transcript
Properties:
  - source_file (text, required)
  - duration (text)
  - language (text)
  - model (text)
  - tier (select: draft|balanced|high_quality)
  - speakers (number)
  - transcribed_at (datetime)
  - confidence_avg (number)
  - word_count (number)
  - hallucination_flags (number)
```

### 7.3 CLI Integration Examples

```bash
# Basic export to MiNotes
audioscript transcribe --input meeting.mp3 --export minotes

# Export with journal entry
audioscript transcribe --input lecture.mp3 --export minotes --journal

# Export to specific MiNotes database
audioscript transcribe --input "*.mp3" --export minotes --minotes-db ~/work/.minotes/default.db

# Markdown sync mode (write files, let MiNotes sync-dir pick them up)
audioscript transcribe --input "*.mp3" --output-format markdown --output-dir ~/minotes-sync/
minotes sync-dir ~/minotes-sync/ --delete_missing=false

# Pipe mode for agent workflows
audioscript transcribe --input meeting.mp3 --pipe --format json | \
  minotes block create --page "Meeting Notes" --content -
```

---

## 8. Competitive Landscape

| Feature | AudioScript (current) | AudioScript (proposed) | WhisperX | faster-whisper | Otter.ai | Notion AI |
|---------|----------------------|----------------------|----------|----------------|----------|-----------|
| Local-first | Yes | Yes | Yes | Yes | No | No |
| Speaker diarization | Yes (pyannote) | Yes (pyannote + WhisperX) | Yes | No | Yes | Yes |
| Forced alignment | No | Yes (F-07) | Yes | No | N/A | N/A |
| Hallucination filter | VAD only | Multi-layer (F-06) | VAD | VAD | Unknown | Unknown |
| Markdown output | No | Yes (F-02) | No | No | No | Proprietary |
| Note app integration | No | MiNotes + Obsidian (F-01, F-12) | No | No | Notion | Native |
| Streaming | No | Yes (F-08) | No | Yes | Yes | Yes |
| Confidence scores | No | Yes (F-06) | No | Yes (raw) | No | No |
| Custom vocabulary | No | Yes (F-11) | No | Yes | Yes | No |
| Agent-friendly CLI | Yes | Yes (enhanced) | Limited | Limited | No | No |
| Swappable backends | No | Yes (F-15) | No | N/A | N/A | N/A |
| Batch processing | Sequential | Parallel (F-09) | Sequential | Sequential | Cloud | Cloud |

---

## 9. Milestones & Phasing

### Phase 1 — v0.2: Core Gaps + MiNotes (Target: 4-6 weeks)
- [ ] F-02: Markdown output format
- [ ] F-03: Faster-whisper backend
- [ ] F-01: MiNotes export (CLI integration + markdown sync)
- [ ] F-04: Real audio cleaning (noisereduce)
- [ ] F-06: Confidence scores + hallucination detection (layers 1-3)
- [ ] G-03: Fix `--concurrency` (basic multiprocessing)
- [ ] G-04: Fix thread-based timeout (switch to multiprocessing)
- [ ] G-12: Transient vs permanent error classification

### Phase 2 — v0.3: Power Features (Target: 4-6 weeks after v0.2)
- [ ] F-05: Real summarization (extractive + optional LLM)
- [ ] F-07: WhisperX integration
- [ ] F-08: Real-time streaming mode
- [ ] F-09: Full parallel processing
- [ ] F-10: Translation support
- [ ] F-11: Custom vocabulary / hot words

### Phase 3 — v0.4: Ecosystem (Target: ongoing)
- [ ] F-12: Additional export targets (Notion)
- [ ] F-13: Emotion/tone detection
- [ ] F-14: Accessibility compliance output
- [ ] F-15: Swappable model backends (NeMo, Canary, Parakeet)
- [ ] F-17: Manifest management
- [ ] F-18: MiNotes bidirectional sync
- [ ] F-19: Obsidian community plugin (TypeScript, wraps AudioScript CLI)
- [ ] F-20: Event-driven auto-transcription (filesystem watcher + MiNotes events)

---

## 10. Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Transcription accuracy (WER) | <8% on LibriSpeech clean | Automated benchmark suite |
| Processing speed (faster-whisper) | >50x real-time on GPU | Benchmark against 1hr test file |
| MiNotes export correctness | 100% structured page creation | Integration test suite |
| Hallucination rate | <1% phantom segments on silence | Test with known-silence audio samples |
| User workflow time (meeting → searchable notes) | <5 min for 1hr meeting | End-to-end timing test |
| Agent integration reliability | 0 crashes in pipe mode over 100 files | CI stress test |

---

## 11. Design Decisions (Resolved)

All architectural questions have been resolved. Decisions are recorded here for future reference.

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| Q1 | MiNotes integration method | **Hybrid: Markdown sync + CLI for metadata** | Markdown sync for bulk content (decoupled, Obsidian-compatible); CLI calls for properties/classes/links. Rust FFI deferred unless performance demands it. |
| Q2 | Default transcription backend | **faster-whisper as default** | 4-6x speed gain is too large to hide behind a flag. Vanilla Whisper as fallback if faster-whisper not installed. `audioscript check` reports active backend. |
| Q3 | Summarization without cloud | **Extractive default + opt-in cloud LLM** | TextRank extractive as zero-config default. Claude/OpenAI API as explicit opt-in (`--summary-provider`). No bundled local LLM — scope creep vs quality gap not worth the VRAM. |
| Q4 | Streaming architecture | **Integrated subcommand** (`audioscript stream`) | Single binary, shared config/manifest, discoverable. Consider separate daemon in v0.4+ if persistent streaming demand materializes. |
| Q5 | Markdown format | **Obsidian conventions** (YAML frontmatter + `[[wikilinks]]`) | De facto standard for knowledge tools. MiNotes, Obsidian, Logseq, Foam all parse it. CommonMark tools gracefully ignore `[[` syntax. |
| Q6 | Speaker database UX | **Enrollment + post-hoc labeling** | `audioscript speakers enroll` for known recurring speakers. Post-hoc labeling for ad-hoc recordings. Never block the pipeline with interactive prompts. |
| Q7 | Confidence score visibility | **Threshold-based** (default 0.4, only flag egregious) | `confidence` always in JSON. Markdown gets `[?]` only for clearly suspect text (below 0.4). Keeps transcripts clean while surfacing real problems. |
| Q8 | Re-transcription vs edits | **Skip if exists** (unless `--force`) for v0.2 | Safe default — respects user edits. Merge strategy (update auto blocks, preserve user blocks) planned for F-18 bidirectional sync, requires MiNotes event log actor attribution. |
| Q9 | GPU memory management | **Sequential load/unload** | Load transcriber → transcribe → unload → load diarizer → diarize → unload. ~2-5s load overhead is small vs transcription time. Avoids OOM on consumer 8GB GPUs. |
| Q10 | Whisper alternatives | **Build `TranscriberBackend` protocol now, implement later** | Define interface, implement `FasterWhisperBackend` first. Don't implement Canary/Parakeet yet — wait for ecosystem stability and user demand. Abstraction pays for itself with WhisperX in P1. |
| Q11 | Python version floor | **Raise to 3.11** | Python 3.9 EOL Oct 2025, 3.10 EOL Oct 2026. 3.11 gives exception groups, `tomllib`, 10-60% perf improvements, better typing. Most CUDA users have 3.11+. |
| Q12 | Manifest schema migration | **Additive schema (nullable new fields)** | Add `confidence`, `hallucination_flags`, `backend` as nullable. Old entries keep nulls. Breaking migration only if fundamental structure changes (unlikely). |
| Q13 | AudioScript as MiNotes plugin | **Yes — register as backend plugin** | MiNotes plugin backend is complete (registration, storage, events, actor attribution). AudioScript registers as `plugin:audioscript` for event tracking, uses persistent plugin storage for sync state, and subscribes to events for auto-transcription. No iframe/UI needed — CLI plugin only. |
| Q14 | Obsidian community plugin | **Yes — build it** | Obsidian has a massive user base and the CLI (Insiders, early 2026) enables scripting against vaults. Markdown export (F-02) with Obsidian conventions is the foundation; native plugin wraps AudioScript for in-app UX. Schedule for Phase 3. |
| Q15 | Event-driven auto-transcription | **Yes — build it** | MiNotes `events --follow` + filesystem watcher on inbox folder = fully automated pipeline. Pair with streaming daemon in Phase 3. Enables "drop audio file → transcript appears in MiNotes" workflow. |
| Q16 | Manifest scope | **Per-project** (stored alongside `--output-dir`) | Cleaner, portable, no global singleton. Cross-project dedup is rare and not worth the complexity. |

### Implications for Implementation

These decisions cascade into the following concrete changes:

1. **`pyproject.toml`**: Raise `requires-python` to `>=3.11`; add `faster-whisper` to core deps; add `mutagen` to optional
2. **Backend protocol**: Define `TranscriberBackend` ABC in `processors/backends.py` before implementing F-03
3. **Markdown exporter**: New module `exporters/markdown.py` with Obsidian conventions baked in
4. **MiNotes exporter**: New module `exporters/minotes.py` using hybrid markdown-sync + CLI approach
5. **Confidence pipeline**: Wire faster-whisper token probabilities through to JSON output schema
6. **Manifest**: Add nullable fields; store in `{output_dir}/.audioscript_manifest.json`
7. **Phase 3 additions**: Obsidian community plugin (F-19), event-driven auto-transcription (F-20)

---

## 12. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Whisper model stagnation (no v4) | Locked to 2023-era accuracy | Medium | Backend abstraction (F-15) enables switching to Canary/Parakeet |
| pyannote license changes | Diarization feature blocked | Low | NeMo as fallback; WhisperX bundles its own pipeline |
| VRAM exhaustion with multiple models | OOM crashes | High | Sequential model loading; int8 quantization; memory monitoring |
| MiNotes API changes | Export integration breaks | Medium | Markdown sync as stable fallback (filesystem is the API) |
| faster-whisper maintenance (CTranslate2 dependency) | Speed regression | Low | WhisperX or whisper.cpp as alternatives |
| WCAG compliance deadline (April 2026) | Miss accessibility market | Medium | Prioritize F-14 if accessibility users appear |

---

## 13. Appendix

### A. Model Comparison (2025-2026 Benchmarks)

| Model | WER (LibriSpeech) | Speed (RTFx) | VRAM | Languages |
|-------|--------------------|-------------|------|-----------|
| Canary Qwen 2.5B | 5.63% | ~418x | ~8GB | 25 |
| Whisper large-v3 | 7.4% | ~1x | ~10GB | 99+ |
| Whisper large-v3-turbo | 7.75% | ~6x | ~6GB | 99+ |
| faster-whisper large-v3 | 7.4% | ~4-6x | ~5GB (int8) | 99+ |
| Distil-Whisper | ~7.4% | ~5-6x | ~5GB | English |
| Parakeet TDT 1.1B | ~8.0% | >2000x | ~4GB | 25 |
| Moonshine (base) | Competitive | Fast | ~0.5GB | English |

### B. MiNotes Data Model Reference

- **Pages:** UUID, title (unique), folder, journal support
- **Blocks:** UUID, hierarchical (parent_id), markdown content
- **Properties:** Typed key-value (text, number, date, select, etc.)
- **Links:** Bidirectional `[[wiki links]]` and `((block refs))`
- **Events:** Append-only audit log with actor attribution
- **Classes:** Entity type definitions with property schemas
- **FTS:** SQLite FTS5 full-text search on blocks

### C. Sources

- HuggingFace Open ASR Leaderboard (2026)
- NVIDIA Speech AI Benchmarks (Canary, Parakeet)
- Interspeech 2025 — Calm-Whisper hallucination paper
- pyannote 3.1 documentation
- MiNotes PRD and codebase (`/root/projects/MiNotes`)
