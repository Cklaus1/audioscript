"""Check command — verify dependencies, auth, and GPU status."""

from __future__ import annotations

import os
import sys
from typing import Any

import typer

from audioscript.cli.output import CLIContext, emit

check_app = typer.Typer(name="check", help="Check dependencies, auth, and GPU status.")


@check_app.command(name="check", hidden=True)
@check_app.callback(invoke_without_command=True)
def check(ctx: typer.Context) -> None:
    """Check system dependencies, authentication, and hardware."""
    cli: CLIContext = ctx.obj

    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "dependencies": _check_dependencies(),
        "auth": _check_auth(),
        "hardware": _check_hardware(),
        "models_cached": _check_cached_models(),
    }

    # Overall readiness
    deps = result["dependencies"]
    result["ready"] = {
        "transcribe": deps["faster_whisper"]["installed"],
        "diarize": deps["pyannote"]["installed"] and result["auth"]["hf_token_set"],
        "vad": deps["faster_whisper"]["installed"],  # Built-in Silero VAD
        "analyze": deps.get("anthropic", {}).get("installed", False) and result["auth"]["anthropic_api_key_set"],
    }

    emit(cli, "check", result)


def _check_dependencies() -> dict[str, Any]:
    """Check if required packages are installed."""
    deps: dict[str, Any] = {}

    # faster-whisper
    try:
        import faster_whisper
        deps["faster_whisper"] = {
            "installed": True,
            "version": getattr(faster_whisper, "__version__", "unknown"),
        }
    except ImportError:
        deps["faster_whisper"] = {"installed": False}

    # PyTorch
    try:
        import torch
        deps["torch"] = {"installed": True, "version": torch.__version__}
    except ImportError:
        deps["torch"] = {"installed": False}

    # pyannote.audio
    try:
        import pyannote.audio
        deps["pyannote"] = {
            "installed": True,
            "version": getattr(pyannote.audio, "__version__", "unknown"),
        }
    except ImportError:
        deps["pyannote"] = {"installed": False}

    # librosa
    try:
        import librosa
        deps["librosa"] = {"installed": True, "version": librosa.__version__}
    except ImportError:
        deps["librosa"] = {"installed": False}

    # PyYAML
    try:
        import yaml
        deps["pyyaml"] = {"installed": True, "version": yaml.__version__}
    except ImportError:
        deps["pyyaml"] = {"installed": False}

    # anthropic (for LLM analysis)
    try:
        import anthropic
        deps["anthropic"] = {"installed": True, "version": getattr(anthropic, "__version__", "unknown")}
    except ImportError:
        deps["anthropic"] = {"installed": False}

    return deps


def _check_auth() -> dict[str, Any]:
    """Check authentication status."""
    hf_token = os.environ.get("HF_TOKEN", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "hf_token_set": bool(hf_token),
        "hf_token_source": "HF_TOKEN env var" if hf_token else None,
        "anthropic_api_key_set": bool(anthropic_key),
        "anthropic_api_key_source": "ANTHROPIC_API_KEY env var" if anthropic_key else None,
    }


def _check_hardware() -> dict[str, Any]:
    """Check hardware capabilities."""
    hw: dict[str, Any] = {"device": "cpu"}

    try:
        import torch
        if torch.cuda.is_available():
            hw["device"] = "cuda"
            hw["cuda_version"] = torch.version.cuda or "unknown"
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["gpu_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw["device"] = "mps"
    except ImportError:
        pass

    return hw


def _check_cached_models() -> list[str]:
    """List locally cached faster-whisper models."""
    try:
        from pathlib import Path

        # faster-whisper caches via huggingface_hub
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.is_dir():
            return []

        cached = []
        for d in cache_dir.iterdir():
            if d.is_dir() and "whisper" in d.name.lower():
                # Extract model name from directory name
                name = d.name.replace("models--", "").replace("--", "/")
                cached.append(name)
        return sorted(cached)
    except Exception:
        return []
