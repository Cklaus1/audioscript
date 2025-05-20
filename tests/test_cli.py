"""Tests for the CLI module."""

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from audioscript.cli.main import app


# Create a test runner
runner = CliRunner()


def test_version_flag():
    """Test that the version flag prints the version and exits."""
    with patch("audioscript.__version__", "0.1.0"):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "AudioScript version: 0.1.0" in result.stdout


def test_no_input_error():
    """Test that an error is shown when no input is provided."""
    # Create a temporary config file with no input specified
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w") as temp_file:
        temp_file.write("output_dir: ./test_output\n")
        temp_file.flush()
        
        # Run with the config file in the current directory
        with patch("pathlib.Path.exists", return_value=True):
            with patch("audioscript.config.settings.load_yaml_config", return_value={}):
                result = runner.invoke(app)
                assert result.exit_code == 1
                assert "Error: No input files specified" in result.stdout


def test_cli_flags():
    """Test that CLI flags are correctly parsed."""
    # Mock the glob and other external calls
    with patch("glob.glob", return_value=["test1.mp3", "test2.mp3"]):
        with patch("pathlib.Path.mkdir"):
            with patch("pathlib.Path.exists", return_value=False):
                with patch("audioscript.processors.audio_processor.AudioProcessor.process_file", return_value=True):
                    result = runner.invoke(
                        app,
                        [
                            "--input=*.mp3",
                            "--output-dir=./test_output",
                            "--tier=high_quality",
                            "--clean-audio",
                            "--summarize",
                            "--force",
                            "--model=test-model",
                        ],
                    )
                    
                    assert result.exit_code == 0
                    assert "Found 2 audio files to process" in result.stdout
                    assert "Transcription tier: high_quality" in result.stdout
                    assert "Output directory:" in result.stdout
                    assert "Processing complete!" in result.stdout
                    assert "Successful: 2" in result.stdout