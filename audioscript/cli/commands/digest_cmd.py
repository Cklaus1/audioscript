"""Digest command — generate a summary digest of recent transcripts.

Scans transcript JSONs in the output directory, filters by date, and
aggregates statistics: total calls, hours, speakers, topics, open
action items, and unidentified speakers.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error

digest_app = typer.Typer(
    name="digest",
    help="Generate a summary digest of recent transcripts.",
)


def _parse_creation_time(data: dict) -> datetime | None:
    """Extract creation time from transcript JSON metadata."""
    meta = data.get("metadata", {})

    # Try audio format tags first
    creation_str = meta.get("audio", {}).get("format_tags", {}).get("creation_time")
    if creation_str:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(creation_str, fmt)
            except ValueError:
                continue

    # Fall back to file modification time from metadata
    file_meta = meta.get("file", {})
    for key in ("modified_time", "created_time"):
        ts = file_meta.get(key)
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                pass

    # Fall back to processing timestamp
    processing_ts = data.get("processing", {}).get("timestamp")
    if processing_ts:
        try:
            return datetime.fromisoformat(processing_ts)
        except (ValueError, TypeError):
            pass

    return None


def _get_duration_hours(data: dict) -> float:
    """Extract duration in hours from transcript metadata."""
    duration_s = data.get("metadata", {}).get("audio", {}).get("duration_seconds", 0)
    return duration_s / 3600.0


@digest_app.command(name="digest", hidden=True)
@digest_app.callback(invoke_without_command=True)
def digest(
    ctx: typer.Context,
    days: int = typer.Option(7, "--days", "-d", help="Number of days to look back."),
    dir: str = typer.Option("./output", "--dir", help="Directory containing transcript JSONs."),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON instead of Rich table."),
) -> None:
    """Generate a summary digest of recent transcripts.

    Scans transcript JSONs, filters by recency, and shows aggregated
    stats: calls, hours, speakers, topics, action items, and
    unidentified speakers.

    Example: audioscript digest --days 7 --dir ./output
    """
    cli: CLIContext = ctx.obj
    output_dir = Path(dir)

    if not output_dir.is_dir():
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"Directory not found: {output_dir}",
            hint="Use --dir to specify the output directory.",
        )
        return

    # Collect transcript JSONs (skip manifests and internal files)
    json_files = sorted(output_dir.glob("*.json"))
    json_files = [
        f for f in json_files
        if "manifest" not in f.name
        and "speaker_identities" not in f.name
        and "sync_cache" not in f.name
        and not f.name.startswith(".")
    ]

    if not json_files:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"No transcript JSON files found in {output_dir}",
        )
        return

    cutoff = datetime.now() - timedelta(days=days)
    end_date = datetime.now()

    # Aggregate data
    calls: list[dict] = []
    total_hours = 0.0
    identified_speakers: set[str] = set()
    unidentified_speakers: Counter[str] = Counter()
    unidentified_minutes: Counter[str] = Counter()
    all_topics: Counter[str] = Counter()
    open_actions: list[dict] = []

    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        creation = _parse_creation_time(data)
        # If we can't determine the date, use file mtime
        if creation is None:
            try:
                creation = datetime.fromtimestamp(json_path.stat().st_mtime)
            except OSError:
                continue

        if creation < cutoff:
            continue

        title = data.get("title") or data.get("llm_analysis", {}).get("title") or json_path.stem
        duration_h = _get_duration_hours(data)
        total_hours += duration_h

        call_info = {
            "file": json_path.name,
            "title": title,
            "date": creation.strftime("%b %d"),
            "hours": round(duration_h, 2),
        }
        calls.append(call_info)

        # Speakers
        llm = data.get("llm_analysis", {})
        for spk in llm.get("speakers", []):
            label = spk.get("label", "")
            name = spk.get("likely_name")
            if name:
                identified_speakers.add(name)
            else:
                unidentified_speakers[label] += 1
                unidentified_minutes[label] += round(duration_h * 60)

        # Topics
        for topic in llm.get("topics", []):
            all_topics[topic.lower()] += 1

        # Action items
        for action in llm.get("action_items", []):
            open_actions.append({
                "text": action.get("text", ""),
                "assignee": action.get("assignee"),
                "deadline": action.get("deadline"),
                "source_title": title,
                "source_date": creation.strftime("%b %d"),
            })

        # Also check top-level action items
        for action in data.get("action_items", []):
            if action not in llm.get("action_items", []):
                open_actions.append({
                    "text": action.get("text", ""),
                    "assignee": action.get("assignee"),
                    "deadline": action.get("deadline"),
                    "source_title": title,
                    "source_date": creation.strftime("%b %d"),
                })

    if not calls:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"No transcripts found in the last {days} days.",
        )
        return

    # Build digest data
    digest_data = {
        "period": {
            "days": days,
            "start": cutoff.strftime("%b %d"),
            "end": end_date.strftime("%b %d, %Y"),
        },
        "calls": len(calls),
        "total_hours": round(total_hours, 1),
        "identified_speakers": len(identified_speakers),
        "unidentified_speakers": len(unidentified_speakers),
        "top_topics": [
            {"topic": t, "count": c}
            for t, c in all_topics.most_common(10)
        ],
        "action_items": open_actions,
        "unidentified": [
            {"label": label, "calls": count, "minutes": unidentified_minutes.get(label, 0)}
            for label, count in unidentified_speakers.most_common(10)
        ],
        "call_list": calls,
    }

    # Structured output
    if output_json or cli.format.value in ("json", "quiet", "yaml"):
        emit(cli, "digest", digest_data)
        return

    # Rich table output
    from rich.panel import Panel
    from rich.table import Table

    start_str = cutoff.strftime("%b %d")
    end_str = end_date.strftime("%b %d, %Y")

    cli.console.print()
    cli.console.print(
        Panel(
            f"[bold]Calls:[/] {len(calls)}  |  "
            f"[bold]Hours:[/] {total_hours:.1f}  |  "
            f"[bold]Speakers:[/] {len(identified_speakers)} identified, "
            f"{len(unidentified_speakers)} unknown",
            title=f"[bold cyan]Digest ({start_str} - {end_str})[/]",
            expand=False,
        )
    )

    # Top topics
    if all_topics:
        cli.console.print("\n[bold]Top Topics:[/]")
        for i, (topic, count) in enumerate(all_topics.most_common(5), 1):
            cli.console.print(f"  {i}. {topic} ({count} calls)")

    # Action items
    if open_actions:
        cli.console.print(f"\n[bold]Action Items ({len(open_actions)}):[/]")
        for action in open_actions[:15]:
            assignee = action.get("assignee") or "unassigned"
            source = action.get("source_title", "")
            date = action.get("source_date", "")
            cli.console.print(
                f"  - [ ] {assignee}: {action['text']}"
                f"  [dim](from \"{source}\", {date})[/]"
            )
        if len(open_actions) > 15:
            cli.console.print(f"  ... and {len(open_actions) - 15} more")

    # Unidentified speakers
    if unidentified_speakers:
        cli.console.print("\n[bold]Unidentified Speakers:[/]")
        for label, count in unidentified_speakers.most_common(5):
            minutes = unidentified_minutes.get(label, 0)
            cli.console.print(f"  {label}: {count} calls, {minutes} min")

    cli.console.print()

    # Also emit structured data
    emit(cli, "digest", digest_data)
