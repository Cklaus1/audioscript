"""Tests for the error classification module."""

import pytest

from audioscript.utils.error_classification import (
    ErrorCategory,
    classify_error,
    should_retry,
)


# --- classify_error: permanent exception types ---

def test_classify_file_not_found():
    assert classify_error(FileNotFoundError("missing.wav")) == ErrorCategory.PERMANENT


def test_classify_value_error():
    assert classify_error(ValueError("bad sample rate")) == ErrorCategory.PERMANENT


def test_classify_type_error():
    assert classify_error(TypeError("expected str")) == ErrorCategory.PERMANENT


def test_classify_permission_error():
    assert classify_error(PermissionError("access denied")) == ErrorCategory.PERMANENT


def test_classify_import_error():
    assert classify_error(ImportError("no module named foo")) == ErrorCategory.PERMANENT


def test_classify_attribute_error():
    assert classify_error(AttributeError("has no attribute x")) == ErrorCategory.PERMANENT


def test_classify_key_error():
    assert classify_error(KeyError("missing_key")) == ErrorCategory.PERMANENT


# --- classify_error: transient exception types ---

def test_classify_timeout_error():
    assert classify_error(TimeoutError("timed out")) == ErrorCategory.TRANSIENT


def test_classify_connection_error():
    assert classify_error(ConnectionError("refused")) == ErrorCategory.TRANSIENT


def test_classify_memory_error():
    assert classify_error(MemoryError()) == ErrorCategory.TRANSIENT


def test_classify_broken_pipe():
    assert classify_error(BrokenPipeError()) == ErrorCategory.TRANSIENT


# --- classify_error: message-based classification ---

def test_classify_cuda_oom_message():
    exc = RuntimeError("CUDA out of memory")
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_timeout_message():
    exc = RuntimeError("operation timeout exceeded")
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_disk_full_message():
    exc = OSError("disk full, no space left on device")
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_resource_busy_message():
    exc = RuntimeError("resource busy")
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_unsupported_codec_message():
    exc = RuntimeError("unsupported codec aac")
    assert classify_error(exc) == ErrorCategory.PERMANENT


def test_classify_invalid_format_message():
    exc = RuntimeError("invalid format specified")
    assert classify_error(exc) == ErrorCategory.PERMANENT


def test_classify_corrupt_file_message():
    exc = RuntimeError("corrupt file header")
    assert classify_error(exc) == ErrorCategory.PERMANENT


def test_classify_empty_file_message():
    exc = RuntimeError("empty file detected")
    assert classify_error(exc) == ErrorCategory.PERMANENT


# --- classify_error: unknown errors default to transient ---

def test_classify_unknown_runtime_error():
    exc = RuntimeError("something completely unexpected")
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_classify_unknown_os_error():
    exc = OSError("mysterious failure")
    assert classify_error(exc) == ErrorCategory.TRANSIENT


# --- should_retry: strategy="never" ---

def test_should_retry_never_transient():
    assert should_retry(TimeoutError(), strategy="never", attempt=1, max_attempts=3) is False


def test_should_retry_never_permanent():
    assert should_retry(ValueError("bad"), strategy="never", attempt=1, max_attempts=3) is False


# --- should_retry: strategy="always" ---

def test_should_retry_always_transient():
    assert should_retry(TimeoutError(), strategy="always", attempt=1, max_attempts=3) is True


def test_should_retry_always_permanent():
    assert should_retry(ValueError("bad"), strategy="always", attempt=1, max_attempts=3) is True


def test_should_retry_always_respects_max():
    assert should_retry(TimeoutError(), strategy="always", attempt=3, max_attempts=3) is False


# --- should_retry: strategy="smart" ---

def test_should_retry_smart_transient():
    assert should_retry(TimeoutError(), strategy="smart", attempt=1, max_attempts=3) is True


def test_should_retry_smart_permanent():
    assert should_retry(ValueError("bad"), strategy="smart", attempt=1, max_attempts=3) is False


def test_should_retry_smart_cuda_oom():
    exc = RuntimeError("CUDA out of memory")
    assert should_retry(exc, strategy="smart", attempt=1, max_attempts=3) is True


def test_should_retry_smart_unsupported():
    exc = RuntimeError("unsupported codec")
    assert should_retry(exc, strategy="smart", attempt=1, max_attempts=3) is False


# --- should_retry: at max_attempts ---

def test_should_retry_at_max_never():
    assert should_retry(TimeoutError(), strategy="never", attempt=3, max_attempts=3) is False


def test_should_retry_at_max_always():
    assert should_retry(TimeoutError(), strategy="always", attempt=3, max_attempts=3) is False


def test_should_retry_at_max_smart():
    assert should_retry(TimeoutError(), strategy="smart", attempt=3, max_attempts=3) is False
