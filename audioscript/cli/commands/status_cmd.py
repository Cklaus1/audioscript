"""Status command — query processing manifest for previous run results."""

import json
from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error

status_app = typer.Typer(name="status", help="Query processing status from manifest.")


@status_app.command(name="status", hidden=True)
@status_app.callback(invoke_without_command=True)
def status(
    ctx: typer.Context,
    output_dir: str = typer.Option(
        "./output", "--output-dir", "-o", help="Output directory containing manifest.json",
    ),
) -> None:
    """Show processing status from a previous run's manifest."""
    cli: CLIContext = ctx.obj

    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.exists():
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"No manifest found at {manifest_path}",
            hint=f"Run 'audioscript transcribe --output-dir {output_dir}' first.",
        )

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        emit_error(
            cli, ExitCode.INTERNAL_ERROR, "internal",
            f"Failed to read manifest: {e}",
        )
        return

    files = manifest.get("files", {})

    # Build summary counts
    counts = {"completed": 0, "processing": 0, "error": 0, "transcribed": 0}
    file_list = []
    for file_hash, info in files.items():
        st = info.get("status", "unknown")
        counts[st] = counts.get(st, 0) + 1
        file_list.append({
            "hash": file_hash[:12],
            "status": st,
            "tier": info.get("tier"),
            "version": info.get("version"),
            "error": info.get("error"),
            "last_updated": info.get("last_updated"),
        })

    emit(cli, "status", {
        "manifest": str(manifest_path),
        "manifest_version": manifest.get("version"),
        "summary": counts,
        "total_files": len(files),
        "files": file_list,
    })
