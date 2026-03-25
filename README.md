# AudioScript

Local-first audio transcription with speaker identity, LLM analysis, and OneDrive sync.

## What It Does

Record voice memos, meetings, or calls → AudioScript automatically transcribes them, identifies speakers across calls, generates AI summaries with action items, and outputs structured markdown for your knowledge management system.

```
Audio file → Transcription → Speaker Identity → LLM Analysis → Markdown/MiNotes
              (faster-whisper)   (cross-call)     (Claude)       (Obsidian-ready)
```

## Quick Start

```bash
# Install
pip install -e .

# Verify setup
audioscript check

# Transcribe a single file
audioscript transcribe meeting.mp3

# Transcribe with speaker identification (requires HF_TOKEN)
export HF_TOKEN=hf_your_token
audioscript transcribe meeting.mp3 --diarize

# Enable AI summaries, titles, and action items (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-your_key
audioscript transcribe meeting.mp3 --diarize

# View the result
audioscript show --latest

# Search across all transcripts
audioscript search -q "budget discussion"
```

## Key Features

### Transcription
- **faster-whisper** backend — 50x real-time on GPU, int8 quantization
- 3 quality tiers: `draft` (base), `balanced` (turbo), `high_quality` (large-v3)
- Hallucination detection (confidence scoring + repetition filtering)
- Markdown, JSON, SRT, VTT output formats

### Speaker Identity
- Stable `spk_xxxx` cluster IDs that persist across calls
- Cross-call voice linking — same person in 10 calls gets one identity
- Confidence bands: confirmed (0.92+), probable (0.80+), candidate (0.60+)
- Review queue for unknown speakers
- CLI: `audioscript speakers list`, `label`, `merge`, `summary`

### LLM Analysis (Claude)
- Auto-generated titles ("Q1 Budget Review" instead of "Recording (215)")
- 2-3 sentence abstractive summaries
- Action items with assignee and deadline extraction
- Key decisions, questions raised, topic tags
- Meeting classification (business, family, sales-call, etc.)
- Speaker name extraction from transcript context
- Cost tracking: `audioscript cost`

### Sync Engine
- Watch a directory for new audio files
- WSL path translation (`C:\Users\...` → `/mnt/c/Users/...`)
- OneDrive Files On-Demand handling (auto-download trigger)
- Local staging for fast I/O on slow WSL mounts
- `audioscript sync --watch` for continuous monitoring

### Output
- Obsidian-compatible markdown with YAML frontmatter
- `[[wikilinks]]` to speakers and related transcripts
- Tags: classification + topics for filtering
- MiNotes export with plugin registration

## Commands

| Command | Description |
|---------|-------------|
| `audioscript transcribe` | Transcribe audio files |
| `audioscript sync` | Auto-transcribe from watched directories |
| `audioscript analyze` | Re-run LLM analysis on existing transcripts (parallel) |
| `audioscript show` | View a transcript in the terminal |
| `audioscript search` | Full-text search across all transcripts |
| `audioscript speakers` | List, label, merge, summarize speaker identities |
| `audioscript cost` | View cumulative LLM spending |
| `audioscript check` | Verify dependencies, auth, and GPU |
| `audioscript init` | Guided first-time setup |
| `audioscript schema` | Introspect capabilities (for agents) |

## Configuration

Create `.audioscript.yaml` or run `audioscript init`:

```yaml
output_dir: "./output"
tier: "balanced"

# Speaker diarization (requires HF_TOKEN env var)
# diarize: true

# Sync — auto-transcribe from OneDrive
# sync:
#   sources:
#     - path: "C:\\Users\\you\\OneDrive\\Documents\\Sound Recordings"
#   output_format: markdown
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace token for speaker diarization |
| `ANTHROPIC_API_KEY` | Claude API key for LLM analysis |
| `AUDIOSCRIPT_FORMAT` | Default output format (json, table, quiet, yaml) |
| `AUDIOSCRIPT_TIER` | Default quality tier |

## Requirements

- Python 3.11+
- NVIDIA GPU (CUDA) recommended, CPU works but slower
- faster-whisper, torch, pyannote.audio (for diarization)

## Architecture

```
audioscript/
├── cli/commands/     14 CLI commands (typer)
├── config/           Pydantic settings + YAML config
├── processors/       faster-whisper, pyannote, hallucination detection
├── speakers/         Identity DB, resolution engine, calendar, enrollment
├── sync/             Sync engine, WSL paths, file discovery, OneDrive
├── formatters/       Markdown output (Obsidian-compatible)
├── exporters/        MiNotes integration
├── llm/              Claude analysis + cost tracking
└── utils/            File hashing, validation, logging, metadata
```

## License

CC BY-NC 4.0
