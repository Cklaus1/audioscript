"""Standalone speaker diarization command (without transcription)."""

import glob
import json
import os
from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error, emit_ndjson, emit_progress
from audioscript.utils.file_utils import get_output_path
from audioscript.utils.validate import PathValidationError, validate_safe_input, validate_safe_output_dir, validate_safe_file_path

diarize_app = typer.Typer(name="diarize", help="Run speaker diarization on audio files.")


@diarize_app.command(name="diarize", hidden=True)
@diarize_app.callback(invoke_without_command=True)
def diarize(
    ctx: typer.Context,
    input: str = typer.Option(
        ..., "--input", "-i", help="Audio file or glob pattern",
    ),
    output_dir: str = typer.Option(
        "./output", "--output-dir", "-o", help="Output directory",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", help="HuggingFace access token",
    ),
    num_speakers: Optional[int] = typer.Option(
        None, "--num-speakers", help="Exact number of speakers",
    ),
    min_speakers: Optional[int] = typer.Option(
        None, "--min-speakers", help="Min expected speakers",
    ),
    max_speakers: Optional[int] = typer.Option(
        None, "--max-speakers", help="Max expected speakers",
    ),
    allow_overlap: bool = typer.Option(
        False, "--allow-overlap", help="Include overlapping speech",
    ),
    speaker_db: Optional[str] = typer.Option(
        None, "--speaker-db", help="Speaker database for identification",
    ),
    speaker_similarity_threshold: float = typer.Option(
        0.7, "--speaker-similarity-threshold", help="Match threshold (0-1)",
    ),
) -> None:
    """Run standalone speaker diarization."""
    cli: CLIContext = ctx.obj

    # Validate paths
    try:
        validate_safe_input(input)
    except PathValidationError as e:
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", str(e), hint=e.hint)
        return

    try:
        validate_safe_output_dir(output_dir)
    except PathValidationError as e:
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", str(e), hint=e.hint)
        return

    if speaker_db:
        try:
            validate_safe_file_path(speaker_db, label="speaker database")
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
            "Diarization requires a HuggingFace token.",
            hint="Set HF_TOKEN env var or pass --hf-token.",
            docs_url="https://huggingface.co/settings/tokens",
        )
        return

    if cli.dry_run:
        emit(cli, "diarize", {
            "dry_run": True,
            "files": input_files,
            "output_dir": output_dir,
        })
        return

    try:
        from audioscript.processors.diarizer import SpeakerDatabase, SpeakerDiarizer
        diarizer = SpeakerDiarizer(hf_token=token)
    except Exception as e:
        emit_error(
            cli, ExitCode.AUTH_ERROR, "auth", str(e),
            hint="Check your HuggingFace token and pyannote model access.",
            docs_url="https://huggingface.co/pyannote/speaker-diarization-3.1",
        )
        return

    # Load speaker database if provided
    spk_db = None
    if speaker_db:
        spk_db = SpeakerDatabase(speaker_db)
        names = spk_db.speaker_names
        if names:
            cli.console.print(f"Loaded speaker DB: {', '.join(names)}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, audio_file in enumerate(input_files):
        file_path = Path(audio_file)
        cli.console.print(f"Diarizing: {file_path.name}")
        emit_progress(cli, audio_file, (i / len(input_files)) * 100, f"Diarizing {file_path.name}")

        try:
            diar_result = diarizer.diarize(
                file_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                allow_overlap=allow_overlap,
            )

            # Speaker identification
            name_mapping = {}
            if spk_db and diar_result.get("speaker_embeddings"):
                name_mapping = spk_db.identify(
                    diar_result["speaker_embeddings"],
                    threshold=speaker_similarity_threshold,
                )

            # Save RTTM
            rttm_path = get_output_path(file_path, out_dir, "rttm")
            diarizer.save_rttm(diar_result["segments"], rttm_path, file_id=file_path.stem)

            # Save embeddings
            if diar_result.get("speaker_embeddings"):
                emb_path = get_output_path(file_path, out_dir, "embeddings.json")
                diarizer.save_embeddings(diar_result["speaker_embeddings"], emb_path)

            result = {
                "file": audio_file,
                "num_speakers": diar_result["num_speakers"],
                "speakers": [name_mapping.get(s, s) for s in diar_result["speakers"]],
                "segments": len(diar_result["segments"]),
                "overlap": diar_result.get("overlap", {}),
                "rttm": str(rttm_path),
            }
            if name_mapping:
                result["identified"] = name_mapping

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
        emit(cli, "diarize", {"results": results})
