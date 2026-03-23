"""Error classification for intelligent retry behavior."""

from __future__ import annotations

from enum import Enum


class ErrorCategory(str, Enum):
    """Classification of errors for retry decisions."""
    TRANSIENT = "transient"
    PERMANENT = "permanent"


def classify_error(exc: Exception) -> ErrorCategory:
    """Classify an exception as transient or permanent.

    Transient errors (worth retrying):
    - Out of memory (GPU/CPU)
    - Timeouts
    - Network/connection errors
    - CUDA errors
    - Generic OS errors (disk full, etc.)

    Permanent errors (fail immediately):
    - File not found
    - Invalid/unsupported format
    - Validation errors (bad params)
    - Type errors
    - Permission denied
    """
    # Permanent by exception type
    if isinstance(exc, (FileNotFoundError, ValueError, TypeError, PermissionError)):
        return ErrorCategory.PERMANENT

    # Permanent by exception type (import errors, attribute errors)
    if isinstance(exc, (ImportError, AttributeError, KeyError)):
        return ErrorCategory.PERMANENT

    # Transient by exception type
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError)):
        return ErrorCategory.TRANSIENT

    if isinstance(exc, MemoryError):
        return ErrorCategory.TRANSIENT

    # Check message content for classification
    msg = str(exc).lower()

    # Permanent patterns
    permanent_patterns = [
        "not found", "invalid", "unsupported", "codec", "format",
        "permission denied", "no such file", "corrupt", "malformed",
        "too short", "empty file",
    ]
    for pattern in permanent_patterns:
        if pattern in msg:
            return ErrorCategory.PERMANENT

    # Transient patterns
    transient_patterns = [
        "cuda", "out of memory", "oom", "timeout", "connection",
        "busy", "resource", "temporary", "unavailable", "retry",
        "disk full", "no space",
    ]
    for pattern in transient_patterns:
        if pattern in msg:
            return ErrorCategory.TRANSIENT

    # Default: treat unknown errors as transient (allows retry)
    return ErrorCategory.TRANSIENT


def should_retry(
    exc: Exception,
    strategy: str,
    attempt: int,
    max_attempts: int,
) -> bool:
    """Determine whether to retry based on error category and strategy.

    Args:
        exc: The exception that occurred.
        strategy: One of "smart", "always", "never".
        attempt: Current attempt number (1-indexed).
        max_attempts: Maximum number of attempts allowed.

    Returns:
        True if the operation should be retried.
    """
    if attempt >= max_attempts:
        return False

    if strategy == "never":
        return False

    if strategy == "always":
        return True

    # strategy == "smart": only retry transient errors
    return classify_error(exc) == ErrorCategory.TRANSIENT
