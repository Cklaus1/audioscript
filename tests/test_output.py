"""Tests for the output formatting module."""

import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from audioscript.cli.output import (
    CLIContext,
    ExitCode,
    OutputFormat,
    auto_detect_format,
    emit,
    emit_error,
    emit_progress,
    _filter_fields,
)


def test_exit_code_classify_auth():
    assert ExitCode.classify(ValueError("missing HuggingFace token")) == ExitCode.AUTH_ERROR


def test_exit_code_classify_validation():
    assert ExitCode.classify(FileNotFoundError("file not found")) == ExitCode.VALIDATION_ERROR
    assert ExitCode.classify(ValueError("bad value")) == ExitCode.VALIDATION_ERROR


def test_exit_code_classify_runtime():
    assert ExitCode.classify(RuntimeError("model failed")) == ExitCode.TRANSCRIPTION_ERROR


def test_exit_code_classify_unknown():
    assert ExitCode.classify(Exception("unknown")) == ExitCode.INTERNAL_ERROR


def test_auto_detect_format_quiet():
    assert auto_detect_format("auto", quiet=True) == OutputFormat.QUIET


def test_auto_detect_format_explicit():
    assert auto_detect_format("json", quiet=False) == OutputFormat.JSON
    assert auto_detect_format("table", quiet=False) == OutputFormat.TABLE


def test_auto_detect_format_yaml():
    assert auto_detect_format("yaml", quiet=False) == OutputFormat.YAML


def test_cli_context_is_structured():
    ctx = CLIContext(format=OutputFormat.JSON)
    assert ctx.is_structured is True

    ctx = CLIContext(format=OutputFormat.QUIET)
    assert ctx.is_structured is True

    ctx = CLIContext(format=OutputFormat.YAML)
    assert ctx.is_structured is True

    ctx = CLIContext(format=OutputFormat.TABLE)
    assert ctx.is_structured is False


def test_cli_context_defaults():
    ctx = CLIContext()
    assert ctx.format == OutputFormat.JSON
    assert ctx.dry_run is False
    assert ctx.pipe is False
    assert ctx.fields is None
    assert ctx.timeout is None
    assert ctx.start_time > 0


# --- Fields filtering ---

def test_filter_fields_top_level():
    data = {"a": 1, "b": 2, "c": 3}
    result = _filter_fields(data, ["a", "c"])
    assert result == {"a": 1, "c": 3}


def test_filter_fields_nested():
    data = {"results": [{"file": "a.mp3", "status": "ok", "extra": 1}]}
    result = _filter_fields(data, ["results.file", "results.status"])
    assert result == {"results": [{"file": "a.mp3", "status": "ok"}]}


def test_filter_fields_missing_key():
    data = {"a": 1}
    result = _filter_fields(data, ["b"])
    assert result == {}


def test_filter_fields_non_dict():
    assert _filter_fields("hello", ["a"]) == "hello"
    assert _filter_fields(42, ["a"]) == 42


def test_filter_fields_dict_value():
    data = {"info": {"name": "test", "size": 100}}
    result = _filter_fields(data, ["info.name"])
    assert result == {"info": {"name": "test"}}


# --- Actionable errors ---

def test_emit_error_with_hint():
    """Test that errors include hint and docs_url."""
    ctx = CLIContext(format=OutputFormat.JSON)
    captured = StringIO()
    with patch("sys.stdout", captured):
        with pytest.raises(SystemExit) as exc:
            emit_error(
                ctx, ExitCode.AUTH_ERROR, "auth", "Missing token",
                hint="Set HF_TOKEN env var",
                docs_url="https://huggingface.co/settings/tokens",
            )
    assert exc.value.code == 2
    data = json.loads(captured.getvalue())
    assert data["ok"] is False
    assert data["error"]["hint"] == "Set HF_TOKEN env var"
    assert data["error"]["docs_url"] == "https://huggingface.co/settings/tokens"


def test_emit_error_without_hint():
    """Test that errors work without hint/docs_url."""
    ctx = CLIContext(format=OutputFormat.JSON)
    captured = StringIO()
    with patch("sys.stdout", captured):
        with pytest.raises(SystemExit):
            emit_error(ctx, ExitCode.VALIDATION_ERROR, "validation", "Bad input")
    data = json.loads(captured.getvalue())
    assert "hint" not in data["error"]
    assert "docs_url" not in data["error"]


# --- Progress events ---

def test_emit_progress_structured():
    """Progress events go to stderr in structured mode."""
    ctx = CLIContext(format=OutputFormat.JSON)
    captured = StringIO()
    with patch("sys.stderr", captured):
        emit_progress(ctx, "test.mp3", 50.0, "Processing")
    line = captured.getvalue().strip()
    data = json.loads(line)
    assert data["event"] == "progress"
    assert data["file"] == "test.mp3"
    assert data["percent"] == 50.0
    assert data["message"] == "Processing"


def test_emit_progress_table_silent():
    """Progress events are not emitted in table mode."""
    ctx = CLIContext(format=OutputFormat.TABLE)
    captured = StringIO()
    with patch("sys.stderr", captured):
        emit_progress(ctx, "test.mp3", 50.0)
    assert captured.getvalue() == ""


# --- YAML output ---

def test_emit_yaml():
    """Test that YAML format produces valid output."""
    ctx = CLIContext(format=OutputFormat.YAML)
    captured = StringIO()
    with patch("sys.stdout", captured):
        emit(ctx, "test", {"key": "value"})
    output = captured.getvalue()
    assert "key: value" in output


# --- Fields in emit ---

def test_emit_with_fields():
    """Test that --fields filters the data in emit."""
    ctx = CLIContext(format=OutputFormat.JSON, fields=["a"])
    captured = StringIO()
    with patch("sys.stdout", captured):
        emit(ctx, "test", {"a": 1, "b": 2})
    data = json.loads(captured.getvalue())
    assert data["data"] == {"a": 1}
    assert "b" not in data["data"]
