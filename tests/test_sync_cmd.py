"""Tests for the sync CLI command."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from audioscript.cli.main import app

runner = CliRunner()


def test_sync_help():
    """sync --help shows help text."""
    result = runner.invoke(app, ["sync", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower() or "Source" in result.stdout


def test_sync_dry_run_with_source():
    """sync --dry-run --source shows dry-run output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("audioscript.cli.commands.sync_cmd.load_sync_config") as mock_load:
            from audioscript.config.settings import SyncConfig
            mock_load.return_value = SyncConfig()

            with patch("audioscript.cli.commands.sync_cmd.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    tier=MagicMock(value="draft"),
                    model_dump=MagicMock(return_value={
                        "output_dir": "./output",
                        "output_format": "json",
                        "tier": "draft",
                    }),
                )

                with patch("audioscript.sync.wsl.translate_path", return_value=tmp_dir):
                    result = runner.invoke(app, [
                        "--format=json", "--dry-run",
                        "sync",
                        "--source", tmp_dir,
                    ])
                    assert result.exit_code == 0
                    data = json.loads(result.stdout)
                    assert data["ok"] is True
                    assert data["data"]["dry_run"] is True


def test_sync_without_sources_shows_validation_error():
    """sync without sources shows validation error."""
    with patch("audioscript.cli.commands.sync_cmd.load_sync_config") as mock_load:
        from audioscript.config.settings import SyncConfig
        mock_load.return_value = SyncConfig(sources=[])

        result = runner.invoke(app, ["--format=json", "sync"])
        assert result.exit_code == 3  # VALIDATION_ERROR
        data = json.loads(result.stdout)
        assert data["ok"] is False
        assert "no sync sources" in data["error"]["message"].lower()


def test_sync_batch_size_passed():
    """sync --source --batch-size passes batch size to config."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("audioscript.cli.commands.sync_cmd.load_sync_config") as mock_load:
            from audioscript.config.settings import SyncConfig
            mock_load.return_value = SyncConfig()

            with patch("audioscript.cli.commands.sync_cmd.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    tier=MagicMock(value="draft"),
                    model_dump=MagicMock(return_value={
                        "output_dir": "./output",
                        "output_format": "json",
                        "tier": "draft",
                    }),
                )

                with patch("audioscript.sync.wsl.translate_path", return_value=tmp_dir):
                    result = runner.invoke(app, [
                        "--format=json", "--dry-run",
                        "sync",
                        "--source", tmp_dir,
                        "--batch-size", "5",
                    ])
                    assert result.exit_code == 0
                    data = json.loads(result.stdout)
                    assert data["data"]["batch_size"] == 5


def test_sync_watch_flag_accepted():
    """sync --watch flag is accepted (don't actually run watch loop)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("audioscript.cli.commands.sync_cmd.load_sync_config") as mock_load:
            from audioscript.config.settings import SyncConfig
            mock_load.return_value = SyncConfig()

            with patch("audioscript.cli.commands.sync_cmd.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    tier=MagicMock(value="draft"),
                    model_dump=MagicMock(return_value={
                        "output_dir": "./output",
                        "output_format": "json",
                        "tier": "draft",
                    }),
                )

                with patch("audioscript.sync.wsl.translate_path", return_value=tmp_dir):
                    result = runner.invoke(app, [
                        "--format=json", "--dry-run",
                        "sync",
                        "--source", tmp_dir,
                        "--watch",
                    ])
                    assert result.exit_code == 0
                    data = json.loads(result.stdout)
                    assert data["data"]["watch"] is True


def test_sync_runs_engine_without_dry_run():
    """sync without --dry-run runs the SyncEngine."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("audioscript.cli.commands.sync_cmd.load_sync_config") as mock_load:
            from audioscript.config.settings import SyncConfig, SyncSourceConfig
            mock_load.return_value = SyncConfig(
                sources=[SyncSourceConfig(path=tmp_dir)],
                output_dir=tmp_dir,
            )

            with patch("audioscript.cli.commands.sync_cmd.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    tier=MagicMock(value="draft"),
                )

                mock_report = MagicMock()
                mock_report.sources = []
                mock_report.total_scanned = 0
                mock_report.total_new = 0
                mock_report.total_transcribed = 0
                mock_report.total_failed = 0
                mock_report.elapsed_seconds = 0.1

                with patch("audioscript.sync.engine.SyncEngine") as MockEngine:
                    mock_engine = MockEngine.return_value
                    mock_engine.run_once.return_value = mock_report

                    result = runner.invoke(app, [
                        "--format=json",
                        "sync",
                        "--source", tmp_dir,
                    ])
                    # Should succeed (exit 0) since no failures
                    assert result.exit_code == 0


def test_sync_output_format_override():
    """sync --output-format is included in dry-run output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("audioscript.cli.commands.sync_cmd.load_sync_config") as mock_load:
            from audioscript.config.settings import SyncConfig
            mock_load.return_value = SyncConfig()

            with patch("audioscript.cli.commands.sync_cmd.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    tier=MagicMock(value="draft"),
                    model_dump=MagicMock(return_value={
                        "output_dir": "./output",
                        "output_format": "markdown",
                        "tier": "draft",
                    }),
                )

                with patch("audioscript.sync.wsl.translate_path", return_value=tmp_dir):
                    result = runner.invoke(app, [
                        "--format=json", "--dry-run",
                        "sync",
                        "--source", tmp_dir,
                        "--output-format", "markdown",
                    ])
                    assert result.exit_code == 0
                    data = json.loads(result.stdout)
                    assert data["data"]["output_format"] == "markdown"
