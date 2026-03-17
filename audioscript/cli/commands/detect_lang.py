"""Detect language of audio files without full transcription."""

import glob
from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error, emit_ndjson, emit_progress
from audioscript.utils.validate import PathValidationError, validate_safe_input

detect_language_app = typer.Typer(
    name="detect-language",
    help="Detect the language of audio files.",
)


@detect_language_app.command(name="detect-language", hidden=True)
@detect_language_app.callback(invoke_without_command=True)
def detect_language(
    ctx: typer.Context,
    input: str = typer.Option(
        ..., "--input", "-i", help="Audio file or glob pattern",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Whisper model to use (default: base)",
    ),
    tier: Optional[str] = typer.Option(
        None, "--tier", "-t", help="Quality tier (draft, balanced, high_quality)",
    ),
) -> None:
    """Detect the language of one or more audio files."""
    cli: CLIContext = ctx.obj

    # Validate input path
    try:
        validate_safe_input(input)
    except PathValidationError as e:
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", str(e), hint=e.hint)

    input_files = glob.glob(input, recursive=True)
    if not input_files:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"No files found: {input}",
            hint="Check the glob pattern and ensure files exist.",
        )

    if cli.dry_run:
        emit(cli, "detect-language", {
            "dry_run": True,
            "files": input_files,
            "model": model or tier or "base",
        })
        return

    try:
        from audioscript.processors.whisper_transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber(
            model_name=model, tier=tier or "draft",
        )
    except Exception as e:
        emit_error(
            cli, ExitCode.TRANSCRIPTION_ERROR, "model", str(e),
            hint="Ensure whisper is installed and the model name is valid. Use 'audioscript schema models' to list available models.",
        )
        return

    results = []
    for i, audio_file in enumerate(input_files):
        cli.console.print(f"Detecting language: {audio_file}")
        emit_progress(cli, audio_file, (i / len(input_files)) * 100, f"Detecting {Path(audio_file).name}")

        try:
            detection = transcriber.detect_language(audio_file)
            result = {
                "file": audio_file,
                "language": detection["language"],
                "probability": round(detection["language_probability"], 4),
                "top_languages": detection.get("all_probabilities", {}),
            }
            if cli.pipe:
                emit_ndjson(result)
            else:
                results.append(result)
        except Exception as e:
            if cli.pipe:
                emit_ndjson({"file": audio_file, "error": str(e)})
            else:
                results.append({"file": audio_file, "error": str(e)})

    if not cli.pipe:
        emit(cli, "detect-language", {"results": results})
