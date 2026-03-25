"""Search command — full-text search across transcript JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error

search_app = typer.Typer(name="search", help="Search across all transcripts.")


@search_app.command(name="search", hidden=True)
@search_app.callback(invoke_without_command=True)
def search(
    ctx: typer.Context,
    query: str = typer.Option(
        ..., "--query", "-q",
        help="Text to search for in transcripts.",
    ),
    dir: str = typer.Option(
        "./output", "--dir", "-d",
        help="Directory containing transcript JSON files.",
    ),
    speaker: Optional[str] = typer.Option(
        None, "--speaker",
        help="Filter results to a specific speaker ID (e.g. spk_a91f).",
    ),
    topic: Optional[str] = typer.Option(
        None, "--topic",
        help="Search within LLM-generated topics instead of transcript text.",
    ),
) -> None:
    """Search text across all transcript JSONs in a directory.

    Examples:
      audioscript search -q "budget" --dir ./output
      audioscript search -q "budget" --speaker spk_a91f
      audioscript search --topic "hiring" -q ""
    """
    cli: CLIContext = ctx.obj

    search_dir = Path(dir)
    if not search_dir.is_dir():
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"Directory not found: {dir}",
            hint="Use --dir to specify the transcript output directory.",
        )
        return

    # Find all transcript JSONs
    json_files = sorted(search_dir.glob("*.json"))
    json_files = [
        f for f in json_files
        if f.name not in ("manifest.json", "speaker_identities.json", "sync_cache.json")
        and not f.name.startswith(".")
    ]

    if not json_files:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"No transcript JSON files found in {dir}",
            hint="Run 'audioscript transcribe' first.",
        )
        return

    results = []
    query_lower = query.lower()

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Topic search mode
        if topic is not None:
            _search_topics(json_path, data, topic, results)
            continue

        # Text search: full text field
        full_text = data.get("text", "")
        if query_lower and query_lower in full_text.lower() and speaker is None:
            snippet = _extract_snippet(full_text, query, context_chars=80)
            results.append({
                "file": json_path.name,
                "path": str(json_path),
                "speaker": None,
                "start": None,
                "match_type": "full_text",
                "snippet": snippet,
            })

        # Text search: per-segment
        segments = data.get("segments", [])
        for seg in segments:
            seg_text = seg.get("text", "")
            seg_speaker = seg.get("speaker", "unknown")

            # Speaker filter
            if speaker and seg_speaker != speaker:
                continue

            if query_lower and query_lower not in seg_text.lower():
                continue

            snippet = _extract_snippet(seg_text, query, context_chars=80)
            results.append({
                "file": json_path.name,
                "path": str(json_path),
                "speaker": seg_speaker,
                "start": seg.get("start"),
                "match_type": "segment",
                "snippet": snippet,
            })

    # Emit results
    if cli.is_structured:
        emit(cli, "search", {
            "query": query,
            "speaker_filter": speaker,
            "topic_filter": topic,
            "directory": str(search_dir),
            "files_searched": len(json_files),
            "total_matches": len(results),
            "results": results,
        })
        return

    # Rich table output
    _render_results(cli, query, results, len(json_files))


def _search_topics(
    json_path: Path, data: dict, topic_query: str, results: list,
) -> None:
    """Search within LLM-generated topics."""
    llm = data.get("llm_analysis", {})
    topics = llm.get("topics", [])
    topic_lower = topic_query.lower()

    for t in topics:
        t_str = str(t)
        if topic_lower in t_str.lower():
            results.append({
                "file": json_path.name,
                "path": str(json_path),
                "speaker": None,
                "start": None,
                "match_type": "topic",
                "snippet": t_str,
            })


def _extract_snippet(text: str, query: str, context_chars: int = 80) -> str:
    """Extract a text snippet around the first occurrence of query."""
    if not query:
        return text[:context_chars * 2].strip()

    idx = text.lower().find(query.lower())
    if idx == -1:
        return text[:context_chars * 2].strip()

    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(query) + context_chars)

    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


def _render_results(
    cli: CLIContext, query: str, results: list, files_searched: int,
) -> None:
    """Render search results as a Rich table with highlighted matches."""
    from rich.table import Table
    from rich.text import Text

    console = cli.console

    console.print(
        f"\n[bold]Search results for[/] [yellow]\"{query}\"[/] "
        f"[dim]({len(results)} matches in {files_searched} files)[/]\n"
    )

    if not results:
        console.print("[dim]No matches found.[/]")
        return

    table = Table(show_header=True, border_style="dim")
    table.add_column("File", style="bold blue", max_width=30)
    table.add_column("Speaker", style="cyan", max_width=12)
    table.add_column("Time", style="dim", max_width=8)
    table.add_column("Match", no_wrap=False)

    for r in results:
        # Format timestamp
        timestamp = ""
        if r.get("start") is not None:
            mins, secs = divmod(int(r["start"]), 60)
            timestamp = f"{mins:02d}:{secs:02d}"

        # Highlight the query in the snippet
        snippet_text = Text(r["snippet"])
        if query:
            snippet_text.highlight_regex(
                re.escape(query), style="bold yellow on dark_red",
            )

        table.add_row(
            r["file"],
            r.get("speaker") or "",
            timestamp,
            snippet_text,
        )

    console.print(table)
