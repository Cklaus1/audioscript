"""WSL detection and Windows→Linux path translation."""

from __future__ import annotations

import functools
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]):[/\\](.*)")


@functools.cache
def is_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux.

    Checks WSL_DISTRO_NAME env var (fast), falls back to /proc/version.
    Result is cached for the process lifetime.
    """
    if os.environ.get("WSL_DISTRO_NAME"):
        return True

    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (OSError, IOError):
        return False


def is_windows_path(path: str) -> bool:
    """Check if a path looks like a Windows path (C:\\... or C:/...)."""
    return bool(_WINDOWS_PATH_RE.match(path))


def translate_path(path: str) -> str:
    """Translate a Windows path to a WSL path if needed.

    - If not WSL or already a Unix path: returns as-is
    - If wslpath available: uses subprocess (handles custom mount points)
    - Fallback: regex C:\\... → /mnt/c/...
    - Handles forward and backslashes
    - Preserves spaces and special characters
    """
    if not is_windows_path(path):
        return path

    if not is_wsl():
        return path

    # Try wslpath first (handles custom mount points from wsl.conf)
    try:
        result = subprocess.run(
            ["wslpath", "-u", path],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            translated = result.stdout.strip()
            logger.info("Translated WSL path: %s → %s", path, translated)
            return translated
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: regex translation
    match = _WINDOWS_PATH_RE.match(path)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        translated = f"/mnt/{drive}/{rest}"
        logger.info("Translated WSL path (fallback): %s → %s", path, translated)
        return translated

    return path


def resolve_sync_path(path: str) -> Path:
    """Translate, validate, and return a resolved Path.

    Raises FileNotFoundError with WSL-specific hints on failure.
    """
    translated = translate_path(path)
    resolved = Path(translated)

    if not resolved.exists():
        hint = ""
        if is_wsl() and is_windows_path(path):
            hint = (
                f"\nHint: The Windows path was translated to: {translated}"
                f"\n  - Ensure the Windows drive is mounted in WSL"
                f"\n  - Check that OneDrive has synced the files locally"
                f"\n  - Verify the path exists in Windows Explorer"
            )
        raise FileNotFoundError(
            f"Source directory not found: {translated}{hint}"
        )

    if not resolved.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {translated}")

    return resolved
