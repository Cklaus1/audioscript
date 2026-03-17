---
name: audioscript-check
version: 1.0.0
description: "Check system dependencies, authentication, and GPU status."
metadata:
  category: "audio"
  requires:
    bins: ["audioscript"]
  cliHelp: "audioscript check --help"
  prerequisites: ["audioscript-shared"]
---

# audioscript check

Pre-flight check for agents — verify all dependencies, authentication, and hardware before running commands.

## Quick Start

```bash
audioscript check --format json
```

## Output (--format json)

```json
{
  "ok": true,
  "command": "check",
  "data": {
    "python": "3.11.5",
    "dependencies": {
      "whisper": { "installed": true, "version": "20250625" },
      "torch": { "installed": true, "version": "2.10.0" },
      "pyannote": { "installed": true, "version": "4.0.0" },
      "librosa": { "installed": true },
      "pyyaml": { "installed": true }
    },
    "auth": {
      "hf_token_set": true,
      "hf_token_source": "HF_TOKEN env var"
    },
    "hardware": {
      "device": "cuda",
      "cuda_version": "12.1",
      "gpu_name": "NVIDIA A100",
      "gpu_memory_mb": 40960
    },
    "models_cached": ["base", "turbo"],
    "ready": {
      "transcribe": true,
      "diarize": true,
      "vad": true
    }
  }
}
```

## Agent Usage

Run `audioscript check --format json` before any processing to verify the system is ready. Check the `ready` field to determine which commands are available.
