"""Init command — setup wizard for first-time AudioScript configuration."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import typer

from audioscript.cli.output import CLIContext, emit

init_app = typer.Typer(name="init", help="Set up AudioScript for first use.")


@init_app.command(name="init", hidden=True)
@init_app.callback(invoke_without_command=True)
def init(ctx: typer.Context) -> None:
    """Interactive setup wizard for AudioScript.

    Checks dependencies, authentication, hardware, and configuration.
    Offers to create a .audioscript.yaml config if one doesn't exist.

    Example: audioscript init
    """
    cli: CLIContext = ctx.obj

    result: dict[str, Any] = {}

    # --- Step 1: Run check logic ---
    from audioscript.cli.commands.check_cmd import (
        _check_auth,
        _check_cached_models,
        _check_dependencies,
        _check_hardware,
    )

    deps = _check_dependencies()
    auth = _check_auth()
    hardware = _check_hardware()
    cached_models = _check_cached_models()

    result["dependencies"] = deps
    result["auth"] = auth
    result["hardware"] = hardware
    result["models_cached"] = cached_models

    # --- Step 2: Display system check (Rich mode) ---
    if not cli.is_structured:
        _render_system_check(cli, deps, auth, hardware, cached_models)

    # --- Step 3: Config file ---
    config_path = Path(".audioscript.yaml")
    result["config_exists"] = config_path.exists()
    if not config_path.exists():
        result["config_hint"] = (
            "No .audioscript.yaml found. Create one to customize defaults "
            "(tier, model, output directory, speaker names, etc.)."
        )
        if not cli.is_structured:
            cli.console.print("\n[bold yellow]Configuration[/]")
            cli.console.print(
                "  No [bold].audioscript.yaml[/] found in current directory."
            )
            cli.console.print(
                "  [dim]Create one to set default tier, model, output directory, and speaker names.[/]"
            )

    # --- Step 4: WSL detection ---
    is_wsl = _detect_wsl()
    result["wsl"] = is_wsl
    if is_wsl and not cli.is_structured:
        cli.console.print("\n[bold cyan]WSL Detected[/]")
        cli.console.print(
            "  You're running inside WSL. To sync with Windows OneDrive:"
        )
        cli.console.print(
            "  [dim]Set your input directory to /mnt/c/Users/<you>/OneDrive/Recordings[/]"
        )
        cli.console.print(
            "  [dim]Use 'audioscript sync --watch /mnt/c/...' for auto-transcription.[/]"
        )

    # --- Step 5: Auth guidance ---
    issues = []

    if not auth["hf_token_set"]:
        issues.append("hf_token")
        result["hf_token_hint"] = (
            "HF_TOKEN not set. Required for speaker diarization (pyannote.audio). "
            "Get a token at: https://huggingface.co/settings/tokens"
        )
        if not cli.is_structured:
            cli.console.print("\n[bold yellow]HuggingFace Token[/]")
            cli.console.print("  [red]HF_TOKEN[/] is not set.")
            cli.console.print(
                "  Required for speaker diarization via pyannote.audio."
            )
            cli.console.print(
                "  Get a token: [link]https://huggingface.co/settings/tokens[/link]"
            )
            cli.console.print(
                "  [dim]export HF_TOKEN=hf_...[/]"
            )

    if not auth["anthropic_api_key_set"]:
        issues.append("anthropic_key")
        result["anthropic_hint"] = (
            "ANTHROPIC_API_KEY not set. Required for LLM analysis (summaries, titles, "
            "action items). Get a key at: https://console.anthropic.com/settings/keys"
        )
        if not cli.is_structured:
            cli.console.print("\n[bold yellow]Anthropic API Key[/]")
            cli.console.print("  [red]ANTHROPIC_API_KEY[/] is not set.")
            cli.console.print(
                "  Required for LLM analysis (summaries, titles, action items, topics)."
            )
            cli.console.print(
                "  Get a key: [link]https://console.anthropic.com/settings/keys[/link]"
            )
            cli.console.print(
                "  [dim]export ANTHROPIC_API_KEY=sk-ant-...[/]"
            )

    # --- Step 6: GPU recommendation ---
    gpu_rec = _recommend_tier(hardware)
    result["recommended_tier"] = gpu_rec["tier"]
    result["gpu_recommendation"] = gpu_rec["reason"]

    if not cli.is_structured:
        cli.console.print(f"\n[bold cyan]Recommended Tier[/]")
        cli.console.print(f"  [bold]{gpu_rec['tier']}[/] — {gpu_rec['reason']}")

    # --- Step 7: Summary ---
    all_good = (
        deps.get("faster_whisper", {}).get("installed", False)
        and not issues
    )
    result["ready"] = all_good

    if not cli.is_structured:
        cli.console.print()
        if all_good:
            cli.console.print(
                "[bold green]You're all set![/] AudioScript is ready to use.\n"
            )
            cli.console.print("  [bold]Next steps:[/]")
            cli.console.print("    audioscript transcribe -i ./recordings/*.mp3")
            cli.console.print("    audioscript show --latest")
            cli.console.print("    audioscript search -q \"keyword\"")
        else:
            cli.console.print("[bold yellow]Almost there![/] Fix the items above, then run [bold]audioscript init[/] again.\n")
            if not deps.get("faster_whisper", {}).get("installed"):
                cli.console.print("  [dim]pip install faster-whisper[/]")
            if "hf_token" in issues:
                cli.console.print("  [dim]export HF_TOKEN=hf_...[/]")
            if "anthropic_key" in issues:
                cli.console.print("  [dim]export ANTHROPIC_API_KEY=sk-ant-...[/]")

    emit(cli, "init", result)


def _detect_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
        return "microsoft" in version_info or "wsl" in version_info
    except OSError:
        return False


def _recommend_tier(hardware: dict[str, Any]) -> dict[str, str]:
    """Recommend a processing tier based on available hardware."""
    device = hardware.get("device", "cpu")
    vram = hardware.get("gpu_memory_mb", 0)

    if device == "cuda" and vram >= 8000:
        return {
            "tier": "high_quality",
            "reason": f"GPU with {vram} MB VRAM detected — use large-v3 model for best accuracy.",
        }
    elif device == "cuda" and vram >= 4000:
        return {
            "tier": "balanced",
            "reason": f"GPU with {vram} MB VRAM detected — medium model offers good speed/accuracy trade-off.",
        }
    elif device == "cuda":
        return {
            "tier": "fast",
            "reason": f"GPU with {vram} MB VRAM detected — use tiny/base model to fit in VRAM.",
        }
    elif device == "mps":
        return {
            "tier": "balanced",
            "reason": "Apple Silicon detected — medium model works well with MPS acceleration.",
        }
    else:
        return {
            "tier": "fast",
            "reason": "CPU only — use tiny/base model for reasonable speed without GPU.",
        }


def _render_system_check(
    cli: CLIContext,
    deps: dict,
    auth: dict,
    hardware: dict,
    cached_models: list,
) -> None:
    """Render dependency and system check as Rich output."""
    from rich.table import Table

    console = cli.console

    console.print("\n[bold bright_blue]AudioScript Setup Wizard[/]\n")

    # Dependencies table
    dep_table = Table(title="Dependencies", show_header=True, border_style="dim")
    dep_table.add_column("Package", style="bold")
    dep_table.add_column("Status")
    dep_table.add_column("Version", style="dim")

    for pkg, info in deps.items():
        installed = info.get("installed", False)
        status = "[green]installed[/]" if installed else "[red]missing[/]"
        version = info.get("version", "") if installed else ""
        dep_table.add_row(pkg, status, str(version))

    console.print(dep_table)

    # Hardware
    console.print(f"\n[bold]Hardware:[/] {hardware.get('device', 'cpu').upper()}", end="")
    if hardware.get("gpu_name"):
        console.print(f" — {hardware['gpu_name']} ({hardware.get('gpu_memory_mb', '?')} MB)")
    else:
        console.print()

    # Cached models
    if cached_models:
        console.print(f"[bold]Cached models:[/] {', '.join(cached_models)}")
    else:
        console.print("[bold]Cached models:[/] [dim]none[/]")
