"""Analyze command — re-run LLM analysis on existing transcript JSONs.

Skips transcription and diarization entirely. Useful for:
- Re-analyzing with an improved LLM prompt
- Adding LLM analysis to transcripts that were processed without an API key
- Regenerating markdown after LLM improvements
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Optional

import typer

from audioscript import __version__
from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error, emit_ndjson

analyze_app = typer.Typer(
    name="analyze",
    help="Re-run LLM analysis on existing transcript JSONs (no re-transcription).",
)


@analyze_app.command(name="analyze", hidden=True)
@analyze_app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    input: str = typer.Option(
        ..., "--input", "-i",
        help="Transcript JSON file or glob pattern (e.g. './output/*.json')",
    ),
    regenerate_markdown: bool = typer.Option(
        True, "--regenerate-markdown/--no-regenerate-markdown",
        help="Regenerate .md files from enriched JSON",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="LLM model (default: claude-sonnet-4-6)",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="LLM provider: anthropic, openai, nim",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Base URL for OpenAI/NIM endpoint",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key (or set ANTHROPIC_API_KEY / NVIDIA_API_KEY env)",
    ),
) -> None:
    """Re-run LLM analysis on existing transcript JSONs.

    Skips transcription and diarization — only runs the LLM analysis step
    and optionally regenerates markdown. Much faster and cheaper than
    re-processing from audio.

    Example: audioscript analyze --input "./output/*.json"
    """
    cli: CLIContext = ctx.obj

    effective_provider = provider or "anthropic"
    effective_key = api_key
    if not effective_key:
        if effective_provider == "anthropic":
            effective_key = os.environ.get("ANTHROPIC_API_KEY")
        elif effective_provider == "nim":
            effective_key = os.environ.get("NVIDIA_API_KEY")
        else:
            effective_key = os.environ.get("OPENAI_API_KEY")
    if not effective_key:
        env_var = {"anthropic": "ANTHROPIC_API_KEY", "nim": "NVIDIA_API_KEY", "openai": "OPENAI_API_KEY"}.get(effective_provider, "API_KEY")
        emit_error(
            cli, ExitCode.AUTH_ERROR, "auth",
            f"{env_var} required for LLM analysis with {effective_provider} provider.",
            hint=f"Set {env_var} env var, pass --api-key, or use --provider to switch.",
        )
        return

    # Find JSON files
    json_files = glob.glob(input, recursive=True)
    json_files = [f for f in json_files if f.endswith(".json") and "manifest" not in f
                  and "speaker_identities" not in f and "sync_cache" not in f]

    if not json_files:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"No transcript JSON files found matching: {input}",
            hint="Use --input './output/*.json' to match transcript files.",
        )
        return

    if cli.dry_run:
        # Estimate cost from transcript sizes
        total_chars = 0
        for jf in json_files:
            try:
                total_chars += Path(jf).stat().st_size
            except OSError:
                pass
        # Rough estimate: ~4 chars per token, Sonnet $3/M input + $15/M output
        est_tokens = total_chars // 4
        est_cost = (est_tokens / 1_000_000) * 3.0 + (len(json_files) * 800 / 1_000_000) * 15.0
        emit(cli, "analyze", {
            "dry_run": True,
            "files": json_files,
            "file_count": len(json_files),
            "model": model or "claude-sonnet-4-6",
            "regenerate_markdown": regenerate_markdown,
            "estimated_input_tokens": est_tokens,
            "estimated_cost_usd": round(est_cost, 4),
        })
        return

    from audioscript.llm.analyzer import analyze_transcript, apply_llm_results
    from audioscript.llm.cost_tracker import CostTracker

    cli.console.print(f"[bold green]AudioScript Analyze[/] v{__version__}")
    cli.console.print(f"Re-analyzing {len(json_files)} transcript(s) with LLM")

    # Cost tracker — use same dir as first file
    first_dir = Path(json_files[0]).parent
    cost_tracker = CostTracker(first_dir / ".audioscript_llm_costs.jsonl")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _analyze_one(json_path_str: str) -> dict:
        """Analyze a single file — safe for concurrent execution."""
        json_path = Path(json_path_str)
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return {"file": json_path.name, "status": "error", "error": str(e)}

        default_model = {"anthropic": "claude-sonnet-4-6", "nim": "qwen/qwen3-next-80b-a3b-instruct", "openai": "gpt-4o"}.get(effective_provider, "claude-sonnet-4-6")
        llm_result = analyze_transcript(
            transcript_text=data.get("text", ""),
            segments=data.get("segments"),
            metadata=data.get("metadata"),
            model=model or default_model,
            api_key=effective_key,
            cost_tracker=cost_tracker,
            call_id=json_path.stem,
            provider=effective_provider,
            base_url=base_url,
        )

        if not llm_result:
            return {"file": json_path.name, "status": "no_result"}

        data = apply_llm_results(data, llm_result)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        if regenerate_markdown:
            from audioscript.formatters.markdown_formatter import render_markdown
            audio_path = Path(data.get("metadata", {}).get("file", {}).get("name", json_path.stem))
            summary_text = llm_result.get("summary") or data.get("text", "")[:100]
            md_content = render_markdown(data, audio_path, metadata=data.get("metadata"), summary=summary_text)
            json_path.with_suffix(".md").write_text(md_content, encoding="utf-8")

        speakers = llm_result.get("speakers", [])
        return {
            "file": json_path.name,
            "status": "completed",
            "title": llm_result.get("title"),
            "classification": llm_result.get("classification"),
            "speakers": len(speakers),
            "actions": len(llm_result.get("action_items", [])),
            "topics": llm_result.get("topics", []),
        }

    # Parallel LLM calls — up to 8 concurrent API requests
    max_workers = min(8, len(json_files))
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_analyze_one, f): f for f in json_files}
        for future in as_completed(futures):
            file_result = future.result()
            cli.console.print(
                f"  {file_result['file']}: {file_result.get('title', file_result['status'])}"
            )
            if cli.pipe:
                emit_ndjson(file_result)
            else:
                results.append(file_result)

    # Summary
    usage = cost_tracker.session_summary()
    cli.console.print(f"\n[bold]Done![/] {len(results)} files analyzed")
    cli.console.print(
        f"LLM cost: ${usage['total_cost_usd']:.4f} "
        f"({usage['total_input_tokens']}+{usage['total_output_tokens']} tokens)"
    )

    emit(cli, "analyze", {
        "files_analyzed": len(results),
        "results": results,
        "llm_usage": usage,
    })
