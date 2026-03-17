"""Tests for the CLI module."""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from audioscript.cli.main import app

runner = CliRunner()


def test_version_flag():
    """Test --version prints version and exits."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "AudioScript" in result.stdout


def test_no_args_shows_help():
    """Test that no arguments shows help (exit code 2 is Typer's convention)."""
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)


# --- Transcribe subcommand ---

def test_transcribe_no_input_error():
    """Test error when no input provided to transcribe."""
    with patch("audioscript.config.settings.load_yaml_config", return_value={}):
        result = runner.invoke(app, ["transcribe"])
        assert result.exit_code != 0


def test_transcribe_basic():
    """Test basic transcription workflow."""
    with patch("glob.glob", return_value=["test1.mp3", "test2.mp3"]):
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                with patch(
                    "audioscript.processors.audio_processor.AudioProcessor.process_file",
                    return_value=True,
                ):
                    result = runner.invoke(app, [
                        "--format=table",
                        "transcribe",
                        "--input=*.mp3",
                        "--output-dir=./test_output",
                        "--tier=high_quality",
                    ])
                    assert result.exit_code == 0


def test_transcribe_json_format():
    """Test that --format json produces valid JSON on stdout."""
    with patch("glob.glob", return_value=["test.mp3"]):
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                with patch(
                    "audioscript.processors.audio_processor.AudioProcessor.process_file",
                    return_value=True,
                ):
                    result = runner.invoke(app, [
                        "--format=json",
                        "transcribe",
                        "--input=test.mp3",
                    ])
                    assert result.exit_code == 0
                    data = json.loads(result.stdout)
                    assert data["ok"] is True
                    assert data["command"] == "transcribe"
                    assert data["data"]["successful"] == 1


def test_transcribe_dry_run():
    """Test --dry-run validates without processing."""
    with patch("glob.glob", return_value=["test1.mp3", "test2.mp3"]):
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                result = runner.invoke(app, [
                    "--format=json", "--dry-run",
                    "transcribe",
                    "--input=*.mp3",
                ])
                assert result.exit_code == 0
                data = json.loads(result.stdout)
                assert data["data"]["dry_run"] is True
                assert data["data"]["file_count"] == 2


def test_transcribe_shortcut_subtitle():
    """Test +subtitle shortcut sets word_timestamps and srt format."""
    with patch("glob.glob", return_value=["test.mp3"]):
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                result = runner.invoke(app, [
                    "--format=json", "--dry-run",
                    "transcribe",
                    "--input=test.mp3",
                    "--shortcut=+subtitle",
                ])
                assert result.exit_code == 0
                data = json.loads(result.stdout)
                assert data["data"]["word_timestamps"] is True


def test_transcribe_no_files_found():
    """Test error when glob matches nothing."""
    with patch("glob.glob", return_value=[]):
        result = runner.invoke(app, [
            "--format=json",
            "transcribe",
            "--input=nonexistent/*.mp3",
        ])
        assert result.exit_code == 3  # VALIDATION_ERROR


def test_transcribe_path_traversal_rejected():
    """Test that path traversal in --input is rejected."""
    result = runner.invoke(app, [
        "--format=json",
        "transcribe",
        "--input=../../etc/passwd",
    ])
    assert result.exit_code == 3
    data = json.loads(result.stdout)
    assert "traversal" in data["error"]["message"].lower()


def test_transcribe_absolute_input_rejected():
    """Test that absolute paths in --input are rejected."""
    result = runner.invoke(app, [
        "--format=json",
        "transcribe",
        "--input=/etc/audio/*.mp3",
    ])
    assert result.exit_code == 3


def test_transcribe_absolute_output_rejected():
    """Test that absolute paths in --output-dir are rejected."""
    with patch("glob.glob", return_value=["test.mp3"]):
        result = runner.invoke(app, [
            "--format=json",
            "transcribe",
            "--input=test.mp3",
            "--output-dir=/tmp/evil",
        ])
        assert result.exit_code == 3


def test_transcribe_metadata_in_results():
    """Test that --metadata includes file metadata in results."""
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake mp3 data " * 100)
        f.flush()
        audio_path = f.name

    try:
        with patch("glob.glob", return_value=[audio_path]):
            with patch("pathlib.Path.mkdir"):
                with patch(
                    "audioscript.utils.file_utils.ProcessingManifest",
                ):
                    with patch(
                        "audioscript.processors.audio_processor.AudioProcessor.process_file",
                        return_value=True,
                    ):
                        result = runner.invoke(app, [
                            "--format=json",
                            "transcribe",
                            "--input=test.mp3",
                            "--metadata",
                        ])
                        assert result.exit_code == 0
                        data = json.loads(result.stdout)
                        assert data["ok"] is True
                        file_result = data["data"]["results"][0]
                        assert "metadata" in file_result
                        meta = file_result["metadata"]
                        # file section
                        assert "file" in meta
                        assert meta["file"]["name"].endswith(".mp3")
                        assert meta["file"]["extension"] == "mp3"
                        assert meta["file"]["size_bytes"] == 1400
                        assert meta["file"]["mime_type"] == "audio/mpeg"
                        assert "modified" in meta["file"]
                        assert "created" in meta["file"]
                        assert "content_hash" in meta["file"]
                        assert len(meta["file"]["content_hash"]) == 64
                        assert "path" in meta["file"]
    finally:
        os.unlink(audio_path)


def test_transcribe_no_metadata_by_default():
    """Test that metadata is NOT included by default."""
    with patch("glob.glob", return_value=["test.mp3"]):
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                with patch(
                    "audioscript.processors.audio_processor.AudioProcessor.process_file",
                    return_value=True,
                ):
                    result = runner.invoke(app, [
                        "--format=json",
                        "transcribe",
                        "--input=test.mp3",
                    ])
                    assert result.exit_code == 0
                    data = json.loads(result.stdout)
                    file_result = data["data"]["results"][0]
                    assert "metadata" not in file_result


def test_transcribe_dry_run_with_timeout():
    """Test that timeout appears in dry-run output."""
    with patch("glob.glob", return_value=["test.mp3"]):
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                result = runner.invoke(app, [
                    "--format=json", "--dry-run", "--timeout=60",
                    "transcribe",
                    "--input=test.mp3",
                ])
                assert result.exit_code == 0
                data = json.loads(result.stdout)
                assert data["data"]["timeout"] == 60


def test_transcribe_with_fields():
    """Test that --fields filters output."""
    with patch("glob.glob", return_value=["test.mp3"]):
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                with patch(
                    "audioscript.processors.audio_processor.AudioProcessor.process_file",
                    return_value=True,
                ):
                    result = runner.invoke(app, [
                        "--format=json", "--fields=successful,failed",
                        "transcribe",
                        "--input=test.mp3",
                    ])
                    assert result.exit_code == 0
                    data = json.loads(result.stdout)
                    assert "successful" in data["data"]
                    assert "results" not in data["data"]


# --- Schema subcommand ---

def test_schema_models():
    """Test schema models returns model list."""
    result = runner.invoke(app, ["--format=json", "schema", "models"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "models" in data["data"]
    assert "turbo" in data["data"]["models"]


def test_schema_tiers():
    """Test schema tiers returns tier info."""
    result = runner.invoke(app, ["--format=json", "schema", "tiers"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data["data"]["tiers"]) == 3


def test_schema_formats():
    """Test schema formats returns format lists including yaml."""
    result = runner.invoke(app, ["--format=json", "schema", "formats"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "srt" in data["data"]["output_formats"]
    assert "yaml" in data["data"]["cli_formats"]


def test_schema_config():
    """Test schema config returns Pydantic JSON schema."""
    result = runner.invoke(app, ["--format=json", "schema", "config"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "properties" in data["data"]


def test_schema_env():
    """Test schema env returns list of supported env vars."""
    result = runner.invoke(app, ["--format=json", "schema", "env"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    env_vars = data["data"]["env_vars"]
    names = [e["name"] for e in env_vars]
    assert "HF_TOKEN" in names
    assert "AUDIOSCRIPT_FORMAT" in names
    assert "AUDIOSCRIPT_LOG" in names


# --- Output formatting ---

def test_quiet_mode():
    """Test --quiet produces compact JSON."""
    result = runner.invoke(app, ["--quiet", "schema", "tiers"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["ok"] is True


def test_yaml_format():
    """Test --format yaml produces YAML output."""
    result = runner.invoke(app, ["--format=yaml", "schema", "tiers"])
    assert result.exit_code == 0
    assert "ok: true" in result.stdout or "ok:" in result.stdout


# --- Exit codes ---

def test_exit_code_validation_error():
    """Test that validation errors return exit code 3."""
    with patch("glob.glob", return_value=[]):
        result = runner.invoke(app, [
            "--format=json",
            "transcribe",
            "--input=nothing",
        ])
        assert result.exit_code == 3
