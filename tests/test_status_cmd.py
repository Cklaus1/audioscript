"""Tests for the status command."""

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from audioscript.cli.main import app

runner = CliRunner()


def test_status_no_manifest():
    """Test status when no manifest exists."""
    result = runner.invoke(app, ["--format=json", "status", "--output-dir=/tmp/nonexistent_audioscript_test"])
    assert result.exit_code == 3  # VALIDATION_ERROR
    data = json.loads(result.stdout)
    assert data["ok"] is False
    assert "hint" in data["error"]


def test_status_with_manifest():
    """Test status reads manifest correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = {
            "version": "1.0",
            "files": {
                "abc123": {"status": "completed", "tier": "draft", "version": "1.0", "last_updated": 1710000000.0},
                "def456": {"status": "error", "tier": "draft", "version": "1.0", "error": "failed", "last_updated": 1710000001.0},
            },
        }
        manifest_path = Path(tmpdir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        result = runner.invoke(app, ["--format=json", "status", f"--output-dir={tmpdir}"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["ok"] is True
        assert data["data"]["total_files"] == 2
        assert data["data"]["summary"]["completed"] == 1
        assert data["data"]["summary"]["error"] == 1


def test_status_empty_manifest():
    """Test status with empty manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = {"version": "1.0", "files": {}}
        manifest_path = Path(tmpdir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        result = runner.invoke(app, ["--format=json", "status", f"--output-dir={tmpdir}"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["data"]["total_files"] == 0
