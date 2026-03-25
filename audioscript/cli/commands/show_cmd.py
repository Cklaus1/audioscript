"""Show command — view a transcript in the terminal with Rich formatting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error

show_app = typer.Typer(name="show", help="View a transcript in the terminal.")

# Speaker color palette for consistent coloring
_SPEAKER_COLORS = [
    "cyan", "green", "yellow", "magenta", "blue",
    "red", "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
]


@show_app.command(name="show", hidden=True)
@show_app.callback(invoke_without_command=True)
def show(
    ctx: typer.Context,
    input: Optional[str] = typer.Option(
        None, "--input", "-i",
        help="Path to a transcript JSON or markdown file.",
    ),
    latest: bool = typer.Option(
        False, "--latest",
        help="Show the most recently modified .json or .md in ./output.",
    ),
) -> None:
    """View a transcript in the terminal with Rich formatting.

    Examples:
      audioscript show -i output/Recording.json
      audioscript show --latest
      audioscript --format json show -i output/Recording.json
    """
    cli: CLIContext = ctx.obj

    # Resolve file path
    file_path = _resolve_input(cli, input, latest)
    if file_path is None:
        return  # error already emitted

    if file_path.suffix == ".md":
        _render_markdown_file(cli, file_path)
    elif file_path.suffix == ".json":
        _render_json_transcript(cli, file_path)
    else:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"Unsupported file type: {file_path.suffix}",
            hint="Provide a .json or .md transcript file.",
        )


def _resolve_input(
    cli: CLIContext,
    input_path: Optional[str],
    latest: bool,
) -> Optional[Path]:
    """Resolve --input or --latest to a concrete file path."""
    if not input_path and not latest:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            "Either --input or --latest is required.",
            hint="Use -i path/to/file.json or --latest to view the newest transcript.",
        )
        return None

    if latest:
        return _find_latest(cli)

    path = Path(input_path)
    if not path.exists():
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"File not found: {input_path}",
        )
        return None
    return path


def _find_latest(cli: CLIContext) -> Optional[Path]:
    """Find the most recently modified .json or .md in ./output."""
    output_dir = Path("./output")
    if not output_dir.is_dir():
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            "No ./output directory found.",
            hint="Run 'audioscript transcribe' first, or use --input to specify a file.",
        )
        return None

    candidates = []
    for ext in ("*.json", "*.md"):
        for f in output_dir.glob(ext):
            # Skip internal files
            if f.name in ("manifest.json", "speaker_identities.json", "sync_cache.json"):
                continue
            if f.name.startswith("."):
                continue
            candidates.append(f)

    if not candidates:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            "No transcript files found in ./output.",
            hint="Run 'audioscript transcribe' first.",
        )
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _render_markdown_file(cli: CLIContext, file_path: Path) -> None:
    """Render a markdown file using Rich.Markdown."""
    content = file_path.read_text(encoding="utf-8")

    if cli.is_structured:
        emit(cli, "show", {
            "file": str(file_path),
            "format": "markdown",
            "content": content,
        })
        return

    from rich.markdown import Markdown
    from rich.panel import Panel

    cli.console.print(Panel(
        Markdown(content),
        title=f"[bold]{file_path.name}[/]",
        border_style="blue",
    ))


def _render_json_transcript(cli: CLIContext, file_path: Path) -> None:
    """Parse and render a JSON transcript with Rich panels and tables."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        emit_error(
            cli, ExitCode.INTERNAL_ERROR, "internal",
            f"Failed to read transcript: {e}",
        )
        return

    # In structured mode, emit the raw JSON
    if cli.is_structured:
        emit(cli, "show", {
            "file": str(file_path),
            "format": "json",
            "transcript": data,
        })
        return

    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = cli.console

    # --- Title ---
    llm = data.get("llm_analysis", {})
    title = llm.get("title") or data.get("title") or file_path.stem
    console.print()
    console.print(Panel(
        f"[bold white]{title}[/]",
        border_style="bright_blue",
        subtitle=str(file_path),
    ))

    # --- Metadata table ---
    metadata = data.get("metadata", {})
    if metadata:
        meta_table = Table(title="Metadata", show_header=False, border_style="dim")
        meta_table.add_column("Key", style="bold cyan", min_width=18)
        meta_table.add_column("Value")

        file_meta = metadata.get("file", {})
        if file_meta.get("name"):
            meta_table.add_row("File", file_meta["name"])
        if file_meta.get("size_bytes"):
            size_mb = file_meta["size_bytes"] / (1024 * 1024)
            meta_table.add_row("Size", f"{size_mb:.1f} MB")
        if metadata.get("audio", {}).get("duration_seconds"):
            dur = metadata["audio"]["duration_seconds"]
            mins, secs = divmod(int(dur), 60)
            meta_table.add_row("Duration", f"{mins}:{secs:02d}")
        if metadata.get("processing", {}).get("tier"):
            meta_table.add_row("Tier", metadata["processing"]["tier"])
        if data.get("language"):
            meta_table.add_row("Language", data["language"])
        if llm.get("classification"):
            meta_table.add_row("Type", llm["classification"])

        console.print(meta_table)

    # --- Summary ---
    summary = llm.get("summary") or data.get("summary")
    if summary:
        console.print()
        console.print(Panel(
            summary,
            title="[bold]Summary[/]",
            border_style="green",
        ))

    # --- Action items ---
    actions = llm.get("action_items", [])
    if actions:
        console.print()
        action_lines = []
        for item in actions:
            if isinstance(item, dict):
                owner = item.get("owner", "")
                text = item.get("action", item.get("text", str(item)))
                action_lines.append(f"  [bold yellow]>[/] {text}" + (f" [dim]({owner})[/]" if owner else ""))
            else:
                action_lines.append(f"  [bold yellow]>[/] {item}")
        console.print(Panel(
            "\n".join(action_lines),
            title="[bold]Action Items[/]",
            border_style="yellow",
        ))

    # --- Topics ---
    topics = llm.get("topics", [])
    if topics:
        topic_str = ", ".join(str(t) for t in topics)
        console.print(f"\n[bold]Topics:[/] {topic_str}")

    # --- Speaker stats ---
    speakers = llm.get("speakers", [])
    segments = data.get("segments", [])
    if speakers or segments:
        _render_speaker_stats(console, speakers, segments)

    # --- Transcript ---
    if segments:
        console.print()
        _render_transcript_segments(console, segments)
    elif data.get("text"):
        console.print()
        console.print(Panel(
            data["text"],
            title="[bold]Transcript[/]",
            border_style="white",
        ))


def _render_speaker_stats(console, speakers: list, segments: list) -> None:
    """Render a speaker statistics table."""
    from rich.table import Table

    stats_table = Table(title="Speakers", border_style="dim")
    stats_table.add_column("ID", style="bold")
    stats_table.add_column("Name")
    stats_table.add_column("Segments", justify="right")
    stats_table.add_column("Words", justify="right")

    # Build speaker map from LLM data
    speaker_map = {}
    for sp in speakers:
        sid = sp.get("id") or sp.get("speaker_id", "")
        speaker_map[sid] = sp.get("name") or sp.get("label", sid)

    # Count segments and words per speaker
    seg_counts: dict[str, int] = {}
    word_counts: dict[str, int] = {}
    for seg in segments:
        spk = seg.get("speaker", "unknown")
        seg_counts[spk] = seg_counts.get(spk, 0) + 1
        text = seg.get("text", "")
        word_counts[spk] = word_counts.get(spk, 0) + len(text.split())

    all_speakers = sorted(set(list(speaker_map.keys()) + list(seg_counts.keys())))
    for i, spk in enumerate(all_speakers):
        color = _SPEAKER_COLORS[i % len(_SPEAKER_COLORS)]
        name = speaker_map.get(spk, "")
        stats_table.add_row(
            f"[{color}]{spk}[/]",
            name,
            str(seg_counts.get(spk, 0)),
            str(word_counts.get(spk, 0)),
        )

    console.print()
    console.print(stats_table)


def _render_transcript_segments(console, segments: list) -> None:
    """Render transcript segments with colored speaker labels and timestamps."""
    from rich.text import Text

    console.rule("[bold]Transcript[/]")
    console.print()

    # Assign colors to speakers
    speaker_ids = sorted(set(seg.get("speaker", "unknown") for seg in segments))
    color_map = {
        spk: _SPEAKER_COLORS[i % len(_SPEAKER_COLORS)]
        for i, spk in enumerate(speaker_ids)
    }

    prev_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        if not text:
            continue

        color = color_map.get(speaker, "white")
        mins, secs = divmod(int(start), 60)
        timestamp = f"{mins:02d}:{secs:02d}"

        # Show speaker label when it changes
        if speaker != prev_speaker:
            console.print()
            console.print(f"  [{color} bold]{speaker}[/] [dim]{timestamp}[/]")
            prev_speaker = speaker

        console.print(f"    {text}")
