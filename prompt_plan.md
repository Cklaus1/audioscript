## Prompt Plan for AudioScript

### 📁 1. CLI Setup & Config Loading
```python
"""
Create a Python CLI app using `typer`. Support the following flags:
--input (file or glob), --output-dir, --tier (draft or high_quality), --version, --clean-audio, --summarize, --force, --model (optional).

Also load settings from a YAML config file named `.audioscript.yaml`. CLI flags should override config values.
Return merged settings as a Python dict.
"""
```

...

### 🔁 9. Resume, Skip, and Retry Logic
```python
"""
Implement logic to:
- Skip files that are already processed at the same tier/version
- Resume long transcription jobs from last checkpoint
- Retry failed files unless --no-retry is set

Use manifest to determine current status of each file.
"""
```
