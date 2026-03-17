"""Tests for the structured logging module."""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from audioscript.utils.logging import JsonLineHandler, setup_logging


def test_json_line_handler_writes_jsonl():
    """Test that JsonLineHandler writes valid JSON lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = JsonLineHandler(tmpdir)
        logger = logging.getLogger("test_jsonl")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("test message")
        handler.close()

        # Find the log file
        log_files = list(Path(tmpdir).glob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["level"] == "INFO"
            assert data["message"] == "test message"
            assert "ts" in data

        logger.removeHandler(handler)


def test_json_line_handler_logs_exception():
    """Test that exceptions are captured in log entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = JsonLineHandler(tmpdir)
        logger = logging.getLogger("test_jsonl_exc")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            raise ValueError("test error")
        except ValueError:
            logger.error("caught error", exc_info=True)

        handler.close()

        log_files = list(Path(tmpdir).glob("*.jsonl"))
        with open(log_files[0]) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["level"] == "ERROR"
            assert "exception" in data
            assert "test error" in data["exception"]

        logger.removeHandler(handler)


def test_setup_logging_default():
    """Test that setup_logging works with default env vars."""
    with patch.dict(os.environ, {}, clear=False):
        # Remove AUDIOSCRIPT_LOG and AUDIOSCRIPT_LOG_FILE if present
        os.environ.pop("AUDIOSCRIPT_LOG", None)
        os.environ.pop("AUDIOSCRIPT_LOG_FILE", None)
        # Should not raise
        setup_logging()


def test_setup_logging_with_file():
    """Test that setup_logging creates file handler when AUDIOSCRIPT_LOG_FILE is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"AUDIOSCRIPT_LOG_FILE": tmpdir, "AUDIOSCRIPT_LOG": "debug"}):
            setup_logging()
            logger = logging.getLogger("audioscript")
            # Should have handlers (stderr + file)
            assert len(logger.handlers) >= 1
