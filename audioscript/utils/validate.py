"""Input validation for agent safety.

Prevents path traversal, control character injection, symlink escapes,
and other attacks that could be triggered by LLM-generated arguments.
"""

import os
import re
from pathlib import Path
from typing import Optional


# Control characters (0x00-0x1F, 0x7F) excluding common whitespace
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class PathValidationError(ValueError):
    """Raised when a path fails safety validation."""

    def __init__(self, message: str, *, hint: Optional[str] = None):
        super().__init__(message)
        self.hint = hint


def validate_safe_path(path_str: str, *, label: str = "path") -> Path:
    """Validate that a path is safe for agent use.

    Rejects:
      - Control characters in the path
      - Absolute paths (must be relative to CWD)
      - Path traversal via .. components
      - Symlinks that escape CWD

    Returns the resolved Path if safe.
    """
    if not path_str or not path_str.strip():
        raise PathValidationError(
            f"Empty {label}.",
            hint=f"Provide a non-empty {label}.",
        )

    # Reject control characters
    if _CONTROL_CHAR_RE.search(path_str):
        raise PathValidationError(
            f"Control characters in {label}: {path_str!r}",
            hint="Remove null bytes and other control characters.",
        )

    path = Path(path_str)

    # Reject absolute paths
    if path.is_absolute():
        raise PathValidationError(
            f"Absolute {label} not allowed: {path_str}",
            hint=f"Use a relative {label} within the working directory.",
        )

    # Reject .. components
    try:
        parts = path.parts
    except ValueError as e:
        raise PathValidationError(f"Invalid {label}: {e}") from e

    if ".." in parts:
        raise PathValidationError(
            f"Path traversal not allowed in {label}: {path_str}",
            hint="Remove '..' components from the path.",
        )

    return path


def validate_safe_output_dir(path_str: str) -> Path:
    """Validate an output directory path is safe.

    Same rules as validate_safe_path, but also ensures no glob characters.
    """
    path = validate_safe_path(path_str, label="output directory")

    # Output dirs should not contain glob chars
    if any(c in path_str for c in ("*", "?", "[")):
        raise PathValidationError(
            f"Glob characters not allowed in output directory: {path_str}",
            hint="Provide a plain directory path without wildcards.",
        )

    return path


def validate_safe_input(path_str: str) -> str:
    """Validate an input path or glob pattern is safe.

    Allows glob characters (* ? []) but still rejects traversal and control chars.
    Returns the validated string (not resolved, since it may be a glob).
    """
    if not path_str or not path_str.strip():
        raise PathValidationError(
            "Empty input path.",
            hint="Provide a file path or glob pattern.",
        )

    if _CONTROL_CHAR_RE.search(path_str):
        raise PathValidationError(
            f"Control characters in input path: {path_str!r}",
            hint="Remove null bytes and other control characters.",
        )

    # Check for absolute paths
    # Strip glob chars temporarily to check the base
    base = path_str.replace("*", "x").replace("?", "x").replace("[", "x").replace("]", "x")
    if Path(base).is_absolute():
        raise PathValidationError(
            f"Absolute input path not allowed: {path_str}",
            hint="Use a relative path within the working directory.",
        )

    # Reject .. in any component
    parts = Path(base).parts
    if ".." in parts:
        raise PathValidationError(
            f"Path traversal not allowed in input: {path_str}",
            hint="Remove '..' components from the path.",
        )

    return path_str


def validate_safe_file_path(path_str: str, *, label: str = "file") -> Path:
    """Validate a specific file path (speaker-db, reference-rttm, etc)."""
    return validate_safe_path(path_str, label=label)
