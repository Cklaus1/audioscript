"""Standalone Voice Activity Detection command."""

import glob
import os
from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error, emit_ndjson, emit_progress
from audioscript.utils.validate import PathValidationError, validate_safe_input

vad_app = typer.Typer(name="vad", help="Detect speech regions in audio files.")


@vad_app.command(name="vad", hidden=True)
@vad_app.callback(invoke_without_command=True)
def vad(
    ctx: typer.Context,
    input: str = typer.Option(
        ..., "--input", "-i", help="Audio file or glob pattern",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", help="HuggingFace access token (falls back to HF_TOKEN env var)",
    ),
    onset: float = typer.Option(0.5, "--onset", help="Onset threshold for speech detection"),
    offset: float = typer.Option(0.5, "--offset", help="Offset threshold"),
    min_duration_on: float = typer.Option(0.0, "--min-duration-on", help="Min speech segment duration"),
    min_duration_off: float = typer.Option(0.0, "--min-duration-off", help="Min silence duration"),
) -> None:
    """Run Voice Activity Detection on audio files."""
    cli: CLIContext = ctx.obj

    # Validate input path
    try:
        validate_safe_input(input)
    except PathValidationError as e:
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", str(e), hint=e.hint)
        return

    input_files = glob.glob(input, recursive=True)
    if not input_files:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"No files found: {input}",
            hint="Check the glob pattern and ensure files exist.",
        )
        return

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        emit_error(
            cli, ExitCode.AUTH_ERROR, "auth",
            "VAD requires a HuggingFace token.",
            hint="Set HF_TOKEN env var or pass --hf-token.",
            docs_url="https://huggingface.co/settings/tokens",
        )
        return

    if cli.dry_run:
        emit(cli, "vad", {
            "dry_run": True,
            "files": input_files,
        })
        return

    try:
        from audioscript.processors.diarizer import SpeakerDiarizer
        diarizer = SpeakerDiarizer(hf_token=token)
    except Exception as e:
        emit_error(
            cli, ExitCode.AUTH_ERROR, "auth", str(e),
            hint="Check your HuggingFace token and pyannote model access.",
            docs_url="https://huggingface.co/pyannote/segmentation-3.0",
        )
        return

    results = []
    for i, audio_file in enumerate(input_files):
        cli.console.print(f"Running VAD: {audio_file}")
        emit_progress(cli, audio_file, (i / len(input_files)) * 100, f"VAD {Path(audio_file).name}")

        try:
            vad_result = diarizer.detect_speech(
                audio_file,
                onset=onset, offset=offset,
                min_duration_on=min_duration_on,
                min_duration_off=min_duration_off,
            )
            vad_result["file"] = audio_file
            if cli.pipe:
                emit_ndjson(vad_result)
            else:
                results.append(vad_result)
        except Exception as e:
            if cli.pipe:
                emit_ndjson({"file": audio_file, "error": str(e)})
            else:
                results.append({"file": audio_file, "error": str(e)})

    if not cli.pipe:
        emit(cli, "vad", {"results": results})
