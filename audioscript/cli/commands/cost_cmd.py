"""Cost tracking command — view cumulative LLM spending."""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, emit

cost_app = typer.Typer(name="cost", help="View LLM token usage and costs.")


@cost_app.command(name="cost", hidden=True)
@cost_app.callback(invoke_without_command=True)
def cost(
    ctx: typer.Context,
    log: Optional[str] = typer.Option(
        None, "--log", help="Path to cost log (default: searches common locations)",
    ),
) -> None:
    """Show cumulative LLM token usage and costs.

    Reads from .audioscript_llm_costs.jsonl and shows per-day and
    per-model breakdowns.
    """
    cli: CLIContext = ctx.obj

    from audioscript.llm.cost_tracker import CostTracker

    # Find cost log
    log_path = None
    if log:
        log_path = Path(log)
    else:
        # Search common locations
        candidates = glob.glob("**/.audioscript_llm_costs.jsonl", recursive=True)
        if candidates:
            log_path = Path(candidates[0])

    if not log_path or not log_path.exists():
        emit(cli, "cost", {
            "total_calls": 0,
            "total_cost_usd": 0,
            "message": "No LLM cost log found. Costs are logged when ANTHROPIC_API_KEY is set.",
        })
        return

    tracker = CostTracker(log_path)
    records = tracker._load_log()

    if not records:
        emit(cli, "cost", {
            "total_calls": 0,
            "total_cost_usd": 0,
            "log_path": str(log_path),
        })
        return

    # Aggregate
    total_input = sum(r.get("input_tokens", 0) for r in records)
    total_output = sum(r.get("output_tokens", 0) for r in records)
    total_cost = sum(r.get("cost_usd", 0) for r in records)

    # Per-model breakdown
    by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    for r in records:
        model = r.get("model", "unknown")
        by_model[model]["calls"] += 1
        by_model[model]["input_tokens"] += r.get("input_tokens", 0)
        by_model[model]["output_tokens"] += r.get("output_tokens", 0)
        by_model[model]["cost_usd"] += r.get("cost_usd", 0)

    # Per-day breakdown
    import datetime
    by_day: dict[str, dict] = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0})
    for r in records:
        ts = r.get("timestamp", 0)
        day = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "unknown"
        by_day[day]["calls"] += 1
        by_day[day]["cost_usd"] += r.get("cost_usd", 0)

    emit(cli, "cost", {
        "log_path": str(log_path),
        "total_calls": len(records),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 4),
        "by_model": {k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in by_model.items()},
        "by_day": {k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in sorted(by_day.items())},
    })
