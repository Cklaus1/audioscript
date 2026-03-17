---
name: audioscript-detect-language
version: 1.0.0
description: "Detect audio language using Whisper — fast, no full transcription."
metadata:
  category: "audio"
  requires:
    bins: ["audioscript"]
  cliHelp: "audioscript detect-language --help"
  prerequisites: ["audioscript-shared"]
---

# audioscript detect-language

Detect the language of audio files using Whisper's language detection without running full transcription.

## Prerequisites

- Whisper installed (no HuggingFace token needed)
- See [audioscript-shared](audioscript-shared.md) for global flags and output patterns

## Quick Start

```bash
# Detect language
audioscript detect-language --input "audio.mp3" --format json

# With specific model
audioscript detect-language --input "audio.mp3" --model large-v3

# Batch
find . -name "*.mp3" | audioscript --pipe detect-language
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--input` / `-i` | Audio file or glob pattern (required) |
| `--model` / `-m` | Whisper model (default: base) |
| `--tier` / `-t` | Quality tier (draft, balanced, high_quality) |

## Output (--format json)

```json
{
  "ok": true,
  "command": "detect-language",
  "data": {
    "results": [
      {
        "file": "audio.mp3",
        "language": "en",
        "probability": 0.9847,
        "top_languages": { "en": 0.9847, "es": 0.0089, "fr": 0.0032 }
      }
    ]
  }
}
```
