"""Tests for the check command."""

import json

import pytest
from typer.testing import CliRunner

from audioscript.cli.main import app

runner = CliRunner()


def test_check_returns_json():
    """Test that check command returns valid JSON with expected structure."""
    result = runner.invoke(app, ["--format=json", "check"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["ok"] is True
    assert data["command"] == "check"
    assert "dependencies" in data["data"]
    assert "auth" in data["data"]
    assert "hardware" in data["data"]
    assert "ready" in data["data"]
    assert "models_cached" in data["data"]


def test_check_dependencies_structure():
    """Test that dependencies have installed field."""
    result = runner.invoke(app, ["--format=json", "check"])
    data = json.loads(result.stdout)
    deps = data["data"]["dependencies"]
    # All deps should have installed field
    for name in ["faster_whisper", "torch", "pyannote", "pyyaml"]:
        assert name in deps
        assert "installed" in deps[name]


def test_check_ready_field():
    """Test that ready field has transcribe/diarize/vad keys."""
    result = runner.invoke(app, ["--format=json", "check"])
    data = json.loads(result.stdout)
    ready = data["data"]["ready"]
    assert "transcribe" in ready
    assert "diarize" in ready
    assert "vad" in ready


def test_check_hardware_has_device():
    """Test that hardware section has a device field."""
    result = runner.invoke(app, ["--format=json", "check"])
    data = json.loads(result.stdout)
    assert "device" in data["data"]["hardware"]
