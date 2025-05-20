# AudioScript

A CLI tool for audio transcription that processes audio files and generates high-quality transcriptions using OpenAI's Whisper model.

## Features

- Process single audio files or batches using glob patterns
- Two quality tiers: draft (faster) and high_quality (more accurate)
- Locally run Whisper models (no API key needed)
- Optionally clean audio before processing
- Generate summaries of transcriptions
- Resume long transcription jobs from checkpoints
- Skip already processed files (unless forced to reprocess)
- Retry failed jobs automatically

## Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)
- PyTorch (automatically installed as a dependency)

### Install from Source

1. Clone the repository:

```bash
git clone https://github.com/yourusername/audioscript.git
cd audioscript
```

2. Install the package in development mode:

```bash
pip install -e .
```

Alternatively, you can use the provided Makefile:

```bash
make install
```

## Usage

Basic usage with a single file:

```bash
audioscript --input="path/to/audio.mp3" --output-dir="./transcripts"
```

Process multiple files using glob patterns:

```bash
audioscript --input="path/to/audio/*.mp3" --output-dir="./transcripts"
```

Generate high-quality transcriptions with summaries:

```bash
audioscript --input="path/to/audio/*.mp3" --tier="high_quality" --summarize
```

Force reprocessing of already processed files:

```bash
audioscript --input="path/to/audio/*.mp3" --force
```

Try the included demo script:

```bash
./try_audioscript.py
```

or

```bash
make run
```

## Whisper Models

AudioScript uses OpenAI's Whisper models for transcription, running locally on your machine. The model used depends on the tier:

- `draft` tier: Uses the "base" Whisper model (faster but less accurate)
- `high_quality` tier: Uses the "large" Whisper model (slower but more accurate)

You can also specify a specific model with the `--model` flag:

```bash
audioscript --input="path/to/audio.mp3" --model="medium"
```

Available models: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"

## Configuration

You can create a `.audioscript.yaml` file in your project directory to store default settings:

```yaml
# Default configuration
output_dir: "./transcripts"
tier: "draft"
version: "1.0"
clean_audio: false
summarize: false
force: false
no_retry: false
model: "base"  # Optional specific model
```

Command line arguments will override values from the config file.

## Command Line Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Input audio file or glob pattern |
| `--output-dir`, `-o` | Directory to save transcription outputs |
| `--tier`, `-t` | Transcription quality tier (draft or high_quality) |
| `--version` | Version of the transcription |
| `--clean-audio` | Clean audio before transcription |
| `--summarize` | Generate a summary of the transcription |
| `--force`, `-f` | Force re-processing of already processed files |
| `--model`, `-m` | Specific Whisper model to use (overrides tier setting) |
| `--no-retry` | Do not retry failed transcriptions |
| `--version`, `-v` | Show the version and exit |

## Output Format

Transcriptions are saved as JSON files with the same name as the input audio file in the specified output directory. The JSON format follows Whisper's output format and includes:

- Full transcription text
- Segments with timestamps
- Language detection information
- Confidence scores

If summarization is enabled, a separate summary file is created with the extension `.summary.txt`.

## Development

### Running Tests

```bash
pytest
```

or

```bash
make test
```

### Structure

- `audioscript/cli/`: Command-line interface
- `audioscript/config/`: Configuration handling
- `audioscript/processors/`: Audio processing and transcription
  - `audio_processor.py`: Main processor that handles the workflow
  - `whisper_transcriber.py`: Whisper model integration
- `audioscript/utils/`: Utility functions for file operations, etc.

### Dependencies

The project uses the following key dependencies:

- `typer`: For the command-line interface
- `openai-whisper`: OpenAI's Whisper model for transcription
- `torch`: PyTorch for the neural network backend
- `pyyaml`: For configuration file parsing
- `rich`: For prettier terminal output
- `pydantic`: For data validation and settings management
