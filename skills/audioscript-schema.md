---
name: audioscript-schema
version: 1.0.0
description: "Introspect AudioScript capabilities: models, tiers, formats, config, env vars."
metadata:
  category: "audio"
  requires:
    bins: ["audioscript"]
  cliHelp: "audioscript schema --help"
  prerequisites: ["audioscript-shared"]
---

# audioscript schema

Discover available models, tiers, output formats, configuration options, and environment variables.

## Commands

```bash
# List available Whisper models
audioscript schema models --format json

# List quality tiers
audioscript schema tiers --format json

# List output formats
audioscript schema formats --format json

# Show full config schema (all fields, types, defaults)
audioscript schema config --format json

# List supported environment variables
audioscript schema env --format json
```

## Example Output

```json
{
  "ok": true,
  "command": "schema.models",
  "data": {
    "models": ["tiny", "tiny.en", "base", "base.en", "small", "medium", "large-v3", "turbo"],
    "tier_mapping": {"draft": "base", "balanced": "turbo", "high_quality": "large-v3"}
  }
}
```

## schema env

Lists all supported environment variables with descriptions and which commands use them:

```json
{
  "ok": true,
  "command": "schema.env",
  "data": {
    "env_vars": [
      {"name": "HF_TOKEN", "description": "HuggingFace access token", "used_by": ["transcribe", "diarize", "vad"]},
      {"name": "AUDIOSCRIPT_FORMAT", "description": "Default output format", "used_by": ["all"]}
    ]
  }
}
```
