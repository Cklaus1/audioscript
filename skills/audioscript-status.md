---
name: audioscript-status
version: 1.0.0
description: "Query processing status from a previous run's manifest."
metadata:
  category: "audio"
  requires:
    bins: ["audioscript"]
  cliHelp: "audioscript status --help"
  prerequisites: ["audioscript-shared"]
---

# audioscript status

Query the processing manifest from a previous transcription run.

## Quick Start

```bash
# Check status of default output directory
audioscript status --format json

# Check specific directory
audioscript status --output-dir ./my-output --format json
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--output-dir` / `-o` | Output directory containing manifest.json (default: ./output) |

## Output (--format json)

```json
{
  "ok": true,
  "command": "status",
  "data": {
    "manifest": "./output/manifest.json",
    "manifest_version": "1.0",
    "summary": { "completed": 5, "processing": 0, "error": 1 },
    "total_files": 6,
    "files": [
      { "hash": "abc123def456", "status": "completed", "tier": "draft", "version": "1.0" },
      { "hash": "789ghi012jkl", "status": "error", "tier": "draft", "error": "model failed" }
    ]
  }
}
```
