---
name: audioscript-vad
version: 1.0.0
description: "Voice Activity Detection — find speech regions in audio files."
metadata:
  category: "audio"
  requires:
    bins: ["audioscript"]
  cliHelp: "audioscript vad --help"
  prerequisites: ["audioscript-shared"]
---

# audioscript vad

Detect speech regions in audio files without transcription or diarization.

## Prerequisites

- HuggingFace token with access to `pyannote/segmentation-3.0`
- See [audioscript-shared](audioscript-shared.md) for global flags and output patterns

## Quick Start

```bash
# Basic VAD
audioscript vad --input "audio.mp3" --hf-token $HF_TOKEN --format json

# Tuned thresholds
audioscript vad --input "audio.mp3" --onset 0.6 --offset 0.4 --min-duration-on 0.5

# Batch
find . -name "*.mp3" | audioscript --pipe vad --hf-token $HF_TOKEN
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--input` / `-i` | Audio file or glob pattern (required) |
| `--hf-token` | HuggingFace token (fallback: HF_TOKEN env) |
| `--onset` | Onset threshold (default: 0.5) |
| `--offset` | Offset threshold (default: 0.5) |
| `--min-duration-on` | Minimum speech segment duration in seconds |
| `--min-duration-off` | Minimum silence duration in seconds |

## Output (--format json)

```json
{
  "ok": true,
  "command": "vad",
  "data": {
    "results": [
      {
        "file": "audio.mp3",
        "speech_percentage": 45.3,
        "total_speech_duration": 27.15,
        "total_duration": 60.0,
        "speech_segments": [
          { "start": 0.5, "end": 12.3 },
          { "start": 15.0, "end": 42.8 }
        ]
      }
    ]
  }
}
```
