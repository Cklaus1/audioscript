"""Check command — verify dependencies, auth, and GPU status."""

import os
import sys
from typing import Any, Dict, List

import typer

from audioscript.cli.output import CLIContext, emit

check_app = typer.Typer(name="check", help="Check dependencies, auth, and GPU status.")


@check_app.command(name="check", hidden=True)
@check_app.callback(invoke_without_command=True)
def check(ctx: typer.Context) -> None:
    """Check system dependencies, authentication, and hardware."""
    cli: CLIContext = ctx.obj

    result: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "dependencies": _check_dependencies(),
        "auth": _check_auth(),
        "hardware": _check_hardware(),
        "models_cached": _check_cached_models(),
    }

    # Overall readiness
    deps = result["dependencies"]
    result["ready"] = {
        "transcribe": deps["whisper"]["installed"],
        "diarize": deps["pyannote"]["installed"] and result["auth"]["hf_token_set"],
        "vad": deps["pyannote"]["installed"] and result["auth"]["hf_token_set"],
    }

    emit(cli, "check", result)


def _check_dependencies() -> Dict[str, Any]:
    """Check if required packages are installed."""
    deps = {}

    # Whisper
    try:
        import whisper
        deps["whisper"] = {"installed": True, "version": getattr(whisper, "__version__", "unknown")}
    except ImportError:
        deps["whisper"] = {"installed": False}

    # PyTorch
    try:
        import torch
        deps["torch"] = {"installed": True, "version": torch.__version__}
    except ImportError:
        deps["torch"] = {"installed": False}

    # pyannote.audio
    try:
        import pyannote.audio
        deps["pyannote"] = {"installed": True, "version": getattr(pyannote.audio, "__version__", "unknown")}
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

    return deps


def _check_auth() -> Dict[str, Any]:
    """Check authentication status."""
    hf_token = os.environ.get("HF_TOKEN", "")
    return {
        "hf_token_set": bool(hf_token),
        "hf_token_source": "HF_TOKEN env var" if hf_token else None,
    }


def _check_hardware() -> Dict[str, Any]:
    """Check hardware capabilities."""
    hw: Dict[str, Any] = {"device": "cpu"}

    try:
        import torch
        if torch.cuda.is_available():
            hw["device"] = "cuda"
            hw["cuda_version"] = torch.version.cuda or "unknown"
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["gpu_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw["device"] = "mps"
    except ImportError:
        pass

    return hw


def _check_cached_models() -> List[str]:
    """List locally cached Whisper models."""
    try:
        import whisper
        from pathlib import Path

        # Whisper caches models in ~/.cache/whisper by default
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        if not os.path.isdir(cache_dir):
            return []

        cached = []
        for f in os.listdir(cache_dir):
            if f.endswith(".pt"):
                # Extract model name from filename
                name = f.replace(".pt", "")
                cached.append(name)
        return sorted(cached)
    except Exception:
        return []
