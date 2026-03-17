---
name: audioscript-shared
version: 1.0.0
description: "Shared patterns for all AudioScript commands: global flags, output formatting, env vars, auth, and safety."
metadata:
  category: "audio"
  requires:
    bins: ["audioscript"]
---

# audioscript — Shared Patterns

All AudioScript commands follow these patterns for agent-friendly operation.

## Global Flags

| Flag | Description |
|------|-------------|
| `--format json\|table\|quiet\|yaml` | Output format. Default: auto-detect (tty=table, pipe=json) |
| `--quiet` / `-q` | Alias for `--format quiet` |
| `--dry-run` | Validate inputs without processing |
| `--pipe` | Read file paths from stdin, emit NDJSON to stdout |
| `--fields "a,b.c"` | Filter output to named fields (dot-notation) |
| `--timeout N` | Per-file processing timeout in seconds |
| `--version` / `-v` | Print version and exit |

## Output Envelope

All commands emit a consistent JSON envelope:

```json
{
  "ok": true,
  "command": "transcribe",
  "data": { ... },
  "meta": { "version": "0.1.0", "elapsed_seconds": 12.3 }
}
```

Errors include actionable hints:

```json
{
  "ok": false,
  "error": {
    "code": 2,
    "error_type": "auth",
    "message": "Diarization requires a HuggingFace token.",
    "hint": "Set HF_TOKEN env var or pass --hf-token.",
    "docs_url": "https://huggingface.co/settings/tokens"
  }
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Transcription/processing error |
| 2 | Auth error (missing token/credentials) |
| 3 | Validation error (bad input/args) |
| 4 | Internal error |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace access token for diarization/VAD |
| `AUDIOSCRIPT_FORMAT` | Default output format |
| `AUDIOSCRIPT_OUTPUT_DIR` | Default output directory |
| `AUDIOSCRIPT_TIER` | Default quality tier |
| `AUDIOSCRIPT_MODEL` | Default Whisper model |
| `AUDIOSCRIPT_LOG` | Stderr log level (debug, info, warning, error) |
| `AUDIOSCRIPT_LOG_FILE` | Directory for JSON-line log files |

Use `audioscript schema env --format json` to discover all env vars programmatically.

## Structured Progress Events

In `--format json` or `--format quiet` mode, progress events are emitted as NDJSON to stderr:

```json
{"event": "progress", "file": "audio.mp3", "percent": 45.0, "message": "Processing audio.mp3", "elapsed_seconds": 5.2}
```

## Path Safety

All input paths are validated to prevent:
- Path traversal (`../`)
- Absolute paths
- Control character injection
- Symlink escapes

## Streaming / Pipe Mode

```bash
# Stream file paths from stdin
find audio/ -name "*.mp3" | audioscript --pipe transcribe --output-dir ./output
```

Each file result is emitted as one NDJSON line to stdout.

## Agent Best Practices

1. **Always use `--format json`** for programmatic access
2. **Check capabilities first**: `audioscript schema models --format json`
3. **Validate before processing**: `audioscript --dry-run transcribe --input "*.mp3"`
4. **Check readiness**: `audioscript check --format json`
5. **Query previous runs**: `audioscript status --output-dir ./output --format json`
6. **Filter large output**: `audioscript transcribe ... --fields "results.file,results.status"`
