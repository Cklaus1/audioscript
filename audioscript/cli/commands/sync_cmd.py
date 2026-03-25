"""Sync command — auto-transcribe new audio files from watched directories."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from audioscript import __version__
from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error
from audioscript.config.settings import (
    SyncConfig,
    SyncSourceConfig,
    get_settings,
    load_sync_config,
)

sync_app = typer.Typer(
    name="sync",
    help="Auto-transcribe new audio files from watched directories.",
)


@sync_app.command(name="sync", hidden=True)
@sync_app.callback(invoke_without_command=True)
def sync(
    ctx: typer.Context,
    # Source override
    source: Optional[str] = typer.Option(
        None, "--source", "-s",
        help="Source directory (overrides config). Supports Windows paths on WSL.",
    ),
    # Sync behavior
    watch: bool = typer.Option(False, "--watch", help="Continuous polling mode"),
    poll_interval: Optional[int] = typer.Option(
        None, "--poll-interval", help="Seconds between scans in watch mode (default: 300)",
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Max files per sync cycle (default: 10, 0=unlimited)",
    ),
    force: Optional[bool] = typer.Option(
        None, "--force", "-f", help="Re-transcribe all files",
    ),
    # OneDrive
    no_download: bool = typer.Option(
        False, "--no-download", help="Skip OneDrive download trigger (process only local files)",
    ),
    download_only: bool = typer.Option(
        False, "--download-only", help="Trigger downloads without transcribing",
    ),
    download_timeout: Optional[int] = typer.Option(
        None, "--download-timeout", help="Seconds to wait for OneDrive downloads (default: 300)",
    ),
    # Transcription overrides
    tier: Optional[str] = typer.Option(None, "--tier", "-t", help="Quality tier override"),
    output_format: Optional[str] = typer.Option(
        None, "--output-format", help="Output format override (default: markdown)",
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory override",
    ),
    diarize: Optional[bool] = typer.Option(None, "--diarize", help="Enable diarization"),
    export: Optional[str] = typer.Option(None, "--export", help="Export target: minotes"),
    summarize: Optional[bool] = typer.Option(None, "--summarize", help="Generate summaries"),
) -> None:
    """Sync directories — scan for new audio files and auto-transcribe them."""
    cli: CLIContext = ctx.obj

    # Load sync config from .audioscript.yaml
    sync_config = load_sync_config()

    # Apply CLI overrides to sync config
    if source:
        sync_config.sources = [SyncSourceConfig(
            path=source,
            tier=tier,
            diarize=diarize,
            export=export,
            output_format=output_format,
            summarize=summarize,
        )]
    if poll_interval is not None:
        sync_config.poll_interval = poll_interval
    if batch_size is not None:
        sync_config.batch_size = batch_size
    if output_dir is not None:
        sync_config.output_dir = output_dir
    if output_format is not None:
        sync_config.output_format = output_format
    if download_timeout is not None:
        sync_config.onedrive.download_timeout = download_timeout

    # Validate we have at least one source
    if not sync_config.sources:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            "No sync sources configured.",
            hint=(
                "Use --source <directory> or add sources to .audioscript.yaml:\n"
                "  sync:\n"
                "    sources:\n"
                '      - path: "C:\\\\Users\\\\...\\\\Sound Recordings"'
            ),
        )
        return

    # Load global transcription config
    global_config = get_settings({
        "output_dir": sync_config.output_dir,
        "output_format": sync_config.output_format,
        "force": force or False,
    })

    # Dry-run mode
    if cli.dry_run:
        from audioscript.sync.wsl import translate_path

        source_info = []
        for s in sync_config.sources:
            translated = translate_path(s.path)
            source_info.append({
                "path": s.path,
                "translated": translated,
                "tier": (s.tier or global_config.tier).value if hasattr(s.tier or global_config.tier, 'value') else str(s.tier or global_config.tier),
                "export": s.export or sync_config.minotes.enabled and "minotes" or None,
            })

        emit(cli, "sync", {
            "dry_run": True,
            "sources": source_info,
            "output_dir": sync_config.output_dir,
            "output_format": sync_config.output_format,
            "batch_size": sync_config.batch_size,
            "poll_interval": sync_config.poll_interval if watch else None,
            "watch": watch,
            "onedrive_download": not no_download,
        })
        return

    # Build and run the sync engine
    from audioscript.sync.engine import SyncEngine

    engine = SyncEngine(sync_config, global_config, cli.console)

    cli.console.print(f"[bold green]AudioScript Sync[/] v{__version__}")

    if watch:
        engine.run_watch(
            poll_interval=poll_interval,
            force=force or False,
            no_download=no_download,
        )
    else:
        report = engine.run_once(
            force=force or False,
            no_download=no_download,
            download_only=download_only,
        )

        # Emit structured report
        emit(cli, "sync", {
            "sources": [
                {
                    "source": sr.source,
                    "scanned": sr.scanned,
                    "local": sr.local,
                    "cloud_only": sr.cloud_only,
                    "downloaded": sr.downloaded,
                    "new": sr.new,
                    "skipped": sr.skipped,
                    "transcribed": sr.transcribed,
                    "failed": sr.failed,
                    "results": sr.results,
                }
                for sr in report.sources
            ],
            "total_scanned": report.total_scanned,
            "total_new": report.total_new,
            "total_transcribed": report.total_transcribed,
            "total_failed": report.total_failed,
            "elapsed_seconds": round(report.elapsed_seconds, 2),
        })

        if report.total_failed > 0:
            raise SystemExit(ExitCode.TRANSCRIPTION_ERROR)
