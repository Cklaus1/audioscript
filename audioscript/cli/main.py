"""AudioScript CLI — Agent-friendly audio transcription tool.

Architecture:
  - Global options (--format, --quiet, --dry-run, --pipe, --fields, --timeout) set via callback
  - Rich Console writes to stderr, structured JSON to stdout
  - Subcommands: transcribe, diarize, detect-language, vad, schema, status, check
  - Env vars: AUDIOSCRIPT_FORMAT, AUDIOSCRIPT_OUTPUT_DIR, AUDIOSCRIPT_TIER, AUDIOSCRIPT_MODEL
"""

import os
from typing import Optional

import typer

from audioscript import __version__
from audioscript.cli.output import CLIContext, ExitCode, auto_detect_format, emit_error
from audioscript.cli.commands.transcribe import transcribe_app
from audioscript.cli.commands.schema_cmd import schema_app
from audioscript.cli.commands.detect_lang import detect_language_app
from audioscript.cli.commands.vad_cmd import vad_app
from audioscript.cli.commands.diarize_cmd import diarize_app
from audioscript.cli.commands.status_cmd import status_app
from audioscript.cli.commands.check_cmd import check_app
from audioscript.cli.commands.sync_cmd import sync_app
from audioscript.cli.commands.speakers_cmd import speakers_app
from audioscript.cli.commands.analyze_cmd import analyze_app
from audioscript.cli.commands.cost_cmd import cost_app

from rich.console import Console

# --- App setup ---
app = typer.Typer(
    name="audioscript",
    help="Agent-friendly audio transcription CLI using OpenAI Whisper.",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)

# Register subcommands
app.add_typer(transcribe_app, name="transcribe", help="Transcribe audio files using Whisper.")
app.add_typer(schema_app, name="schema", help="Introspect AudioScript capabilities.")
app.add_typer(detect_language_app, name="detect-language", help="Detect audio language.")
app.add_typer(vad_app, name="vad", help="Voice Activity Detection.")
app.add_typer(diarize_app, name="diarize", help="Standalone speaker diarization.")
app.add_typer(status_app, name="status", help="Query processing status from manifest.")
app.add_typer(check_app, name="check", help="Check dependencies, auth, and GPU status.")
app.add_typer(sync_app, name="sync", help="Auto-transcribe new audio files from watched directories.")
app.add_typer(speakers_app, name="speakers", help="Manage speaker identities across calls.")
app.add_typer(analyze_app, name="analyze", help="Re-run LLM analysis on existing transcripts.")
app.add_typer(cost_app, name="cost", help="View LLM token usage and costs.")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(f"AudioScript {__version__}")
        raise typer.Exit()


@app.callback()
def global_options(
    ctx: typer.Context,
    format: Optional[str] = typer.Option(
        None, "--format",
        help="Output format: json, table, quiet, yaml. Default: auto-detect (tty=table, pipe=json).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress UI output, emit minimal JSON to stdout. Alias for --format quiet.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Validate inputs and show what would happen without processing.",
    ),
    pipe: bool = typer.Option(
        False, "--pipe",
        help="Streaming mode: read file paths from stdin, emit NDJSON results to stdout.",
    ),
    fields: Optional[str] = typer.Option(
        None, "--fields",
        help="Comma-separated dot-notation fields to include in output (e.g. 'results.file,results.status').",
    ),
    timeout: Optional[int] = typer.Option(
        None, "--timeout",
        help="Per-file processing timeout in seconds.",
    ),
    version: Optional[bool] = typer.Option(
        None, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Global options applied to all subcommands."""
    # Env var fallback for format
    effective_format = format or os.environ.get("AUDIOSCRIPT_FORMAT", "auto")
    resolved_format = auto_detect_format(effective_format, quiet)

    # When outputting structured data, route Rich to stderr
    if resolved_format.value in ("json", "quiet", "yaml"):
        console = Console(stderr=True)
    else:
        console = Console()

    # Parse --fields
    parsed_fields = None
    if fields:
        parsed_fields = [f.strip() for f in fields.split(",") if f.strip()]

    ctx.ensure_object(dict)
    ctx.obj = CLIContext(
        format=resolved_format,
        dry_run=dry_run,
        pipe=pipe,
        fields=parsed_fields,
        timeout=timeout,
        console=console,
    )


if __name__ == "__main__":
    app()
