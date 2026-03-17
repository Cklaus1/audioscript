---
name: audioscript-transcribe
version: 1.0.0
description: "Transcribe audio files using OpenAI Whisper with optional speaker diarization."
metadata:
  category: "audio"
  requires:
    bins: ["audioscript"]
  cliHelp: "audioscript transcribe --help"
  prerequisites: ["audioscript-shared"]
---

# audioscript transcribe

Transcribe audio files to text using OpenAI Whisper, running locally.

## Prerequisites

- Whisper installed (`pip install openai-whisper`)
- For diarization: HuggingFace token with pyannote access
- See [audioscript-shared](audioscript-shared.md) for global flags and output patterns

## Quick Start

```bash
# Basic transcription
audioscript transcribe --input "audio.mp3" --output-dir ./output --format json

# High quality with subtitles
audioscript transcribe --input "audio.mp3" --shortcut +subtitle --tier high_quality

# Meeting transcription with speaker identification
audioscript transcribe --input "meeting.mp3" --shortcut +meeting --hf-token $HF_TOKEN

# Batch process with NDJSON streaming
find . -name "*.mp3" | audioscript --pipe transcribe --output-dir ./output

# With timeout and delay between files
audioscript --timeout 120 transcribe --input "*.mp3" --delay 2
```

## Shortcuts

| Shortcut | Expands to |
|----------|------------|
| `+subtitle` | `--word-timestamps --output-format srt` |
| `+meeting` | `--diarize --summarize --word-timestamps` |
| `+draft` | `--tier draft` |
| `+hq` | `--tier high_quality --beam-size 5` |

## Output (--format json)

```json
{
  "ok": true,
  "command": "transcribe",
  "data": {
    "files_processed": 1,
    "successful": 1,
    "failed": 0,
    "output_dir": "/path/to/output",
    "manifest": "/path/to/output/manifest.json",
    "results": [
      {"file": "audio.mp3", "status": "completed"}
    ]
  },
  "meta": {"version": "0.1.0", "elapsed_seconds": 12.3}
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Transcription error |
| 2 | Auth error (missing HF token) |
| 3 | Validation error (bad input) |
| 4 | Internal error |

## Key Flags

- `--tier draft|balanced|high_quality` — Quality/speed tradeoff
- `--model turbo` — Explicit Whisper model
- `--language en` — Force language (auto-detect if omitted)
- `--word-timestamps` — Word-level timing
- `--output-format srt|vtt|txt|tsv|all` — Subtitle/text output
- `--diarize` — Speaker identification
- `--dry-run` — Validate without processing
- `--pipe` — NDJSON streaming mode
- `--timeout N` — Per-file timeout in seconds (global flag)
- `--fields "results.file,results.status"` — Filter output fields (global flag)
- `--delay N` — Delay in seconds between files
- `--concurrency N` — Parallel file processing (default: 1)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AUDIOSCRIPT_OUTPUT_DIR` | Default output directory |
| `AUDIOSCRIPT_TIER` | Default quality tier |
| `AUDIOSCRIPT_MODEL` | Default Whisper model |
| `HF_TOKEN` | HuggingFace token for diarization |

## Path Safety

Input paths are validated — `../`, absolute paths, and control characters are rejected. This prevents agent-generated paths from escaping the working directory.
