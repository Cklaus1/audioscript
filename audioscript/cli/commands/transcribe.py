"""Main transcription command — the core AudioScript workflow."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from audioscript import __version__
from audioscript.config.settings import TranscriptionTier, get_settings
from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error, emit_ndjson, emit_progress
from audioscript.processors.audio_processor import AudioProcessor
from audioscript.utils.file_utils import ProcessingManifest
# extract_metadata imported lazily in pipe/batch mode (reads from saved JSON now)
from audioscript.utils.validate import (
    PathValidationError,
    validate_safe_input,
    validate_safe_output_dir,
    validate_safe_file_path,
)

transcribe_app = typer.Typer(name="transcribe", help="Transcribe audio files using Whisper.")

# Shortcut presets for agent-friendly high-level operations
SHORTCUTS = {
    "+subtitle": {"word_timestamps": True, "output_format": "srt"},
    "+meeting": {"diarize": True, "summarize": True, "word_timestamps": True},
    "+draft": {"tier": "draft"},
    "+hq": {"tier": "high_quality", "beam_size": 5},
}


class _TimeoutError(Exception):
    """Raised when per-file processing exceeds --timeout."""
    pass


def _run_with_timeout(
    func: Any, timeout_seconds: int,
) -> tuple[Any, Exception | None]:
    """Run func() in a thread with a timeout.

    Note: Thread-based timeout cannot forcefully kill GPU operations,
    but it does report the timeout correctly. The thread may continue
    running in the background until the operation completes naturally.
    This is acceptable for per-file timeouts where the operation will
    eventually finish.
    """
    import threading

    result_holder: list[Any] = [None]
    error_holder: list[Exception | None] = [None]

    def target() -> None:
        try:
            result_holder[0] = func()
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return None, _TimeoutError(f"Processing exceeded {timeout_seconds}s timeout")

    if error_holder[0]:
        return None, error_holder[0]

    return result_holder[0], None


@transcribe_app.command(name="transcribe", hidden=True)
@transcribe_app.callback(invoke_without_command=True)
def transcribe(
    ctx: typer.Context,
    # --- Core ---
    input_file: Optional[str] = typer.Argument(None, help="Audio file or glob (also accepts --input)"),
    input: Optional[str] = typer.Option(None, "--input", "-i", help="Audio file or glob pattern"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    tier: Optional[TranscriptionTier] = typer.Option(None, "--tier", "-t", help="Quality tier: draft, balanced, high_quality"),
    doc_version: Optional[str] = typer.Option(None, "--doc-version", help="Transcription version string"),
    clean_audio: Optional[bool] = typer.Option(None, "--clean-audio", help="Clean audio before transcription"),
    summarize: Optional[bool] = typer.Option(None, "--summarize", help="Generate summary"),
    force: Optional[bool] = typer.Option(None, "--force", "-f", help="Force re-processing"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Whisper model (e.g. tiny, base.en, turbo, large-v3)"),
    no_retry: Optional[bool] = typer.Option(None, "--no-retry", help="Disable retries"),
    max_retries: Optional[int] = typer.Option(None, "--max-retries", help="Max retries (default: 3)"),
    # --- Diarization ---
    diarize: Optional[bool] = typer.Option(None, "--diarize", help="Enable speaker diarization"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="HuggingFace token"),
    diarization_model: Optional[str] = typer.Option(None, "--diarization-model", help="Diarization pipeline"),
    num_speakers: Optional[int] = typer.Option(None, "--num-speakers", help="Exact speaker count"),
    min_speakers: Optional[int] = typer.Option(None, "--min-speakers", help="Min speakers"),
    max_speakers: Optional[int] = typer.Option(None, "--max-speakers", help="Max speakers"),
    allow_overlap: Optional[bool] = typer.Option(None, "--allow-overlap", help="Overlapping diarization"),
    speaker_db: Optional[str] = typer.Option(None, "--speaker-db", help="Speaker database path"),
    speaker_similarity_threshold: Optional[float] = typer.Option(None, "--speaker-similarity-threshold", help="Match threshold"),
    vad: Optional[bool] = typer.Option(None, "--vad", help="Run VAD before transcription"),
    reference_rttm: Optional[str] = typer.Option(None, "--reference-rttm", help="Reference RTTM for DER"),
    segmentation_batch_size: Optional[int] = typer.Option(None, "--segmentation-batch-size", help="Segmentation batch size"),
    embedding_batch_size: Optional[int] = typer.Option(None, "--embedding-batch-size", help="Embedding batch size"),
    # --- Whisper ---
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (auto-detect if omitted)"),
    temperature: Optional[str] = typer.Option(None, "--temperature", help="Temperature(s), comma-separated for fallback"),
    word_timestamps: Optional[bool] = typer.Option(None, "--word-timestamps", help="Word-level timestamps"),
    hallucination_silence_threshold: Optional[float] = typer.Option(None, "--hallucination-silence-threshold", help="Skip hallucinated silence"),
    beam_size: Optional[int] = typer.Option(None, "--beam-size", help="Beam search width"),
    best_of: Optional[int] = typer.Option(None, "--best-of", help="Sampling candidates"),
    clip_timestamps: Optional[str] = typer.Option(None, "--clip-timestamps", help="Time ranges to process"),
    carry_initial_prompt: Optional[bool] = typer.Option(None, "--carry-initial-prompt", help="Carry prompt across windows"),
    condition_on_previous_text: Optional[bool] = typer.Option(None, "--condition-on-previous-text/--no-condition-on-previous-text", help="Use previous text as prompt"),
    # --- Decode ---
    suppress_tokens: Optional[str] = typer.Option(None, "--suppress-tokens", help="Token IDs to suppress"),
    suppress_blank: Optional[bool] = typer.Option(None, "--suppress-blank", help="Suppress blanks"),
    fp16: Optional[bool] = typer.Option(None, "--fp16/--no-fp16", help="Half-precision inference"),
    patience: Optional[float] = typer.Option(None, "--patience", help="Beam patience factor"),
    length_penalty: Optional[float] = typer.Option(None, "--length-penalty", help="Length penalty"),
    # --- Output ---
    output_format: Optional[str] = typer.Option(None, "--output-format", help="File format: json, txt, vtt, srt, tsv, all"),
    highlight_words: Optional[bool] = typer.Option(None, "--highlight-words", help="Underline words in subtitles"),
    max_line_width: Optional[int] = typer.Option(None, "--max-line-width", help="Max subtitle line width"),
    max_line_count: Optional[int] = typer.Option(None, "--max-line-count", help="Max subtitle lines"),
    max_words_per_line: Optional[int] = typer.Option(None, "--max-words-per-line", help="Max words per subtitle line"),
    download_root: Optional[str] = typer.Option(None, "--download-root", help="Model cache directory"),
    # --- Metadata ---
    metadata: Optional[bool] = typer.Option(None, "--metadata", help="Extract and embed audio file metadata (date, device, author, codec, etc.)"),
    # --- Audio cleaning ---
    clean_level: Optional[str] = typer.Option(None, "--clean-level", help="Noise reduction level: light, moderate, aggressive"),
    # --- Hallucination detection ---
    min_confidence: Optional[float] = typer.Option(None, "--min-confidence", help="Min confidence threshold (0-1) for segments"),
    hallucination_filter: Optional[str] = typer.Option(None, "--hallucination-filter", help="Hallucination filter mode: auto, flag, off"),
    # --- Error handling ---
    retry_strategy: Optional[str] = typer.Option(None, "--retry-strategy", help="Retry strategy: smart, always, never"),
    # --- Export ---
    export: Optional[str] = typer.Option(None, "--export", help="Export target: minotes"),
    minotes_sync_dir: Optional[str] = typer.Option(None, "--minotes-sync-dir", help="MiNotes sync directory path"),
    # --- Shortcut ---
    shortcut: Optional[str] = typer.Option(None, "--shortcut", "-s", help="Preset shortcut: +subtitle, +meeting, +draft, +hq"),
    # --- Batch control ---
    delay: Optional[float] = typer.Option(None, "--delay", help="Delay in seconds between files (for resource management)"),
) -> None:
    """Transcribe audio files with optional diarization, VAD, and subtitle output."""
    cli: CLIContext = ctx.obj

    # Merge positional arg with --input (positional takes precedence)
    if input_file and not input:
        input = input_file

    # Apply shortcut presets
    shortcut_overrides = {}
    if shortcut:
        if shortcut not in SHORTCUTS:
            emit_error(
                cli, ExitCode.VALIDATION_ERROR, "validation",
                f"Unknown shortcut '{shortcut}'. Available: {', '.join(SHORTCUTS.keys())}",
                hint="Use --shortcut +subtitle, +meeting, +draft, or +hq.",
            )
            return
        shortcut_overrides = SHORTCUTS[shortcut]

    # --- Validate paths (agent safety) ---
    if input:
        try:
            validate_safe_input(input)
        except PathValidationError as e:
            emit_error(
                cli, ExitCode.VALIDATION_ERROR, "validation", str(e),
                hint=e.hint,
            )
            return

    # Env var fallbacks
    effective_output_dir = output_dir or os.environ.get("AUDIOSCRIPT_OUTPUT_DIR")
    effective_tier = tier
    if effective_tier is None:
        env_tier = os.environ.get("AUDIOSCRIPT_TIER")
        if env_tier:
            try:
                effective_tier = TranscriptionTier(env_tier)
            except ValueError:
                pass
    effective_model = model or os.environ.get("AUDIOSCRIPT_MODEL")

    if effective_output_dir:
        try:
            validate_safe_output_dir(effective_output_dir)
        except PathValidationError as e:
            emit_error(
                cli, ExitCode.VALIDATION_ERROR, "validation", str(e),
                hint=e.hint,
            )
            return

    if speaker_db:
        try:
            validate_safe_file_path(speaker_db, label="speaker database")
        except PathValidationError as e:
            emit_error(
                cli, ExitCode.VALIDATION_ERROR, "validation", str(e),
                hint=e.hint,
            )
            return

    if reference_rttm:
        try:
            validate_safe_file_path(reference_rttm, label="reference RTTM")
        except PathValidationError as e:
            emit_error(
                cli, ExitCode.VALIDATION_ERROR, "validation", str(e),
                hint=e.hint,
            )
            return

    # Handle --pipe: read file paths from stdin
    if cli.pipe and not input:
        input_files = [line.strip() for line in sys.stdin if line.strip()]
    elif input:
        input_files = glob.glob(input, recursive=True)
    else:
        input_files = None

    # Build CLI args
    cli_args = {
        "input": input, "output_dir": effective_output_dir, "tier": effective_tier,
        "version": doc_version, "clean_audio": clean_audio, "summarize": summarize,
        "force": force, "model": effective_model, "no_retry": no_retry, "max_retries": max_retries,
        "diarize": diarize, "hf_token": hf_token, "diarization_model": diarization_model,
        "num_speakers": num_speakers, "min_speakers": min_speakers,
        "max_speakers": max_speakers, "allow_overlap": allow_overlap,
        "speaker_db": speaker_db, "speaker_similarity_threshold": speaker_similarity_threshold,
        "vad": vad, "reference_rttm": reference_rttm,
        "segmentation_batch_size": segmentation_batch_size,
        "embedding_batch_size": embedding_batch_size,
        "language": language, "temperature": temperature,
        "word_timestamps": word_timestamps,
        "hallucination_silence_threshold": hallucination_silence_threshold,
        "beam_size": beam_size, "best_of": best_of, "clip_timestamps": clip_timestamps,
        "carry_initial_prompt": carry_initial_prompt,
        "condition_on_previous_text": condition_on_previous_text,
        "suppress_tokens": suppress_tokens, "suppress_blank": suppress_blank,
        "fp16": fp16, "patience": patience, "length_penalty": length_penalty,
        "output_format": output_format, "highlight_words": highlight_words,
        "max_line_width": max_line_width, "max_line_count": max_line_count,
        "max_words_per_line": max_words_per_line, "download_root": download_root,
        "metadata": metadata,
        "clean_level": clean_level,
        "min_confidence": min_confidence, "hallucination_filter": hallucination_filter,
        "retry_strategy": retry_strategy,
        "export": export, "minotes_sync_dir": minotes_sync_dir,
    }
    # Merge shortcut presets (lower priority than explicit CLI args)
    for k, v in shortcut_overrides.items():
        if cli_args.get(k) is None:
            cli_args[k] = v

    try:
        settings = get_settings(cli_args)
    except ValueError as e:
        msg = str(e)
        hint = None
        docs_url = None
        if "HuggingFace token" in msg:
            hint = "Set HF_TOKEN env var or pass --hf-token."
            docs_url = "https://huggingface.co/settings/tokens"
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", msg, hint=hint, docs_url=docs_url)
        return
    except Exception as e:
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", str(e))
        return

    if not input_files:
        if not settings.input:
            emit_error(
                cli, ExitCode.VALIDATION_ERROR, "validation",
                "No input files specified.",
                hint="Use --input <file_or_glob> or pipe file paths via --pipe.",
            )
            return
        input_files = glob.glob(settings.input, recursive=True)

    if not input_files:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"No files found matching: {settings.input}",
            hint="Check the glob pattern and ensure files exist.",
        )
        return

    output_path = Path(settings.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Dry run ---
    if cli.dry_run:
        emit(cli, "transcribe", {
            "dry_run": True,
            "files": input_files,
            "file_count": len(input_files),
            "output_dir": str(output_path.absolute()),
            "tier": settings.tier.value,
            "model": settings.model or settings.tier.value,
            "diarize": settings.diarize,
            "word_timestamps": settings.word_timestamps or settings.diarize,
            "output_format": settings.output_format,
            "timeout": cli.timeout,
        })
        return

    # --- Status output (goes to stderr in structured mode) ---
    cli.console.print(f"[bold green]AudioScript[/] v{__version__}")
    cli.console.print(f"Files: [bold]{len(input_files)}[/] | Tier: [bold]{settings.tier.value}[/] | Output: [bold]{output_path.absolute()}[/]")
    if settings.diarize:
        cli.console.print("Diarization: [bold]enabled[/]")
    if cli.timeout:
        cli.console.print(f"Timeout: [bold]{cli.timeout}s[/] per file")

    manifest_path = output_path / "manifest.json"
    manifest = ProcessingManifest(manifest_path)
    processor = AudioProcessor(settings, manifest, cli.console)

    # Determine effective timeout
    effective_timeout = cli.timeout

    # --- Process files ---
    file_results = []

    if cli.pipe:
        # NDJSON streaming: one result per file
        for i, audio_file in enumerate(input_files):
            file_path = Path(audio_file)
            cli.console.print(f"Processing: {file_path.name}")
            emit_progress(cli, audio_file, (i / len(input_files)) * 100, f"Starting {file_path.name}")

            if effective_timeout:
                result, err = _run_with_timeout(lambda fp=file_path: processor.process_file(fp), effective_timeout)
                success = err is None and result
                if err and isinstance(err, _TimeoutError):
                    emit_ndjson({"file": str(audio_file), "status": "timeout", "error": str(err)})
                    continue
            else:
                success = processor.process_file(file_path)

            ndjson_result = {
                "file": str(audio_file),
                "status": "completed" if success else "failed",
                "output_dir": str(output_path.absolute()),
            }
            if settings.metadata and success:
                try:
                    from audioscript.utils.file_utils import get_output_path
                    json_out = get_output_path(file_path, output_path, "json")
                    if json_out.exists():
                        import json as _json
                        with open(json_out) as _f:
                            saved = _json.load(_f)
                        if "metadata" in saved:
                            ndjson_result["metadata"] = saved["metadata"]
                    if "metadata" not in ndjson_result:
                        from audioscript.utils.metadata import extract_metadata
                        ndjson_result["metadata"] = extract_metadata(file_path)
                except Exception:
                    pass
            emit_ndjson(ndjson_result)

            if delay and i < len(input_files) - 1:
                import time
                time.sleep(delay)
    else:
        # Batch: collect results, emit summary
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=cli.console,
        ) as progress:
            task = progress.add_task(f"[cyan]Processing {len(input_files)} files...", total=len(input_files))
            successful = 0
            failed = 0

            for i, audio_file in enumerate(input_files):
                file_path = Path(audio_file)
                progress.update(task, description=f"[cyan]Processing {file_path.name}...")
                emit_progress(cli, audio_file, (i / len(input_files)) * 100, f"Processing {file_path.name}")

                if effective_timeout:
                    result, err = _run_with_timeout(lambda fp=file_path: processor.process_file(fp), effective_timeout)
                    success = err is None and result
                    if err and isinstance(err, _TimeoutError):
                        file_results.append({"file": str(audio_file), "status": "timeout", "error": str(err)})
                        failed += 1
                        progress.advance(task)
                        continue
                else:
                    success = processor.process_file(file_path)

                file_result = {
                    "file": str(audio_file),
                    "status": "completed" if success else "failed",
                }
                if settings.metadata and success:
                    try:
                        # Try reading metadata from saved JSON first (avoid re-running ffprobe)
                        from audioscript.utils.file_utils import get_output_path
                        json_out = get_output_path(file_path, output_path, "json")
                        if json_out.exists():
                            import json as _json
                            with open(json_out) as _f:
                                saved = _json.load(_f)
                            if "metadata" in saved:
                                file_result["metadata"] = saved["metadata"]
                        if "metadata" not in file_result:
                            # Fallback: extract directly
                            from audioscript.utils.metadata import extract_metadata
                            file_result["metadata"] = extract_metadata(file_path)
                    except Exception:
                        pass
                file_results.append(file_result)
                if success:
                    successful += 1
                else:
                    failed += 1
                progress.advance(task)

                if delay and i < len(input_files) - 1:
                    import time
                    time.sleep(delay)

        emit_progress(cli, "", 100, "Complete")
        cli.console.print(f"\n[bold green]Complete![/] {successful} succeeded, {failed} failed")
        cli.console.print(f"Output: [bold]{output_path.absolute()}[/]")

        emit(cli, "transcribe", {
            "files_processed": len(input_files),
            "successful": successful,
            "failed": failed,
            "output_dir": str(output_path.absolute()),
            "manifest": str(manifest_path),
            "results": file_results,
        })

        if failed > 0:
            raise SystemExit(ExitCode.TRANSCRIPTION_ERROR)
