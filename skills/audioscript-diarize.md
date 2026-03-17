---
name: audioscript-diarize
version: 1.0.0
description: "Standalone speaker diarization — identify who spoke when."
metadata:
  category: "audio"
  requires:
    bins: ["audioscript"]
  cliHelp: "audioscript diarize --help"
  prerequisites: ["audioscript-shared"]
---

# audioscript diarize

Run speaker diarization on audio files without transcription.

## Prerequisites

- HuggingFace token with access to `pyannote/speaker-diarization-3.1`
- See [audioscript-shared](audioscript-shared.md) for global flags and output patterns

## Quick Start

```bash
# Basic diarization
audioscript diarize --input "meeting.mp3" --hf-token $HF_TOKEN --format json

# With speaker identification
audioscript diarize --input "meeting.mp3" --speaker-db speakers.json --hf-token $HF_TOKEN

# Known speaker count
audioscript diarize --input "meeting.mp3" --num-speakers 3 --hf-token $HF_TOKEN

# Batch with NDJSON streaming
find . -name "*.mp3" | audioscript --pipe diarize --hf-token $HF_TOKEN
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--input` / `-i` | Audio file or glob pattern (required) |
| `--output-dir` / `-o` | Output directory (default: ./output) |
| `--hf-token` | HuggingFace token (fallback: HF_TOKEN env) |
| `--num-speakers` | Exact speaker count (if known) |
| `--min-speakers` / `--max-speakers` | Speaker count range |
| `--allow-overlap` | Include overlapping speech regions |
| `--speaker-db` | JSON file mapping speaker names to embeddings |
| `--speaker-similarity-threshold` | Matching threshold 0.0-1.0 (default: 0.7) |

## Output (--format json)

```json
{
  "ok": true,
  "command": "diarize",
  "data": {
    "results": [
      {
        "file": "meeting.mp3",
        "num_speakers": 2,
        "speakers": ["Alice", "Bob"],
        "segments": 47,
        "overlap": { "overlap_percentage": 5.2 },
        "rttm": "/abs/path/meeting.rttm",
        "identified": { "SPEAKER_00": "Alice", "SPEAKER_01": "Bob" }
      }
    ]
  }
}
```

## Output Files

- `{stem}.rttm` — Rich Transcription Time Marked format
- `{stem}.embeddings.json` — Speaker embedding vectors
