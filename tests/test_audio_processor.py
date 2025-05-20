"""Tests for the audio processor."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioscript.processors.audio_processor import AudioProcessor
from audioscript.utils.file_utils import ProcessingManifest


def test_audio_processor_initialization():
    """Test initializing the audio processor."""
    # Create a mock manifest
    manifest = MagicMock(spec=ProcessingManifest)
    
    # Create settings
    settings = {
        "tier": "draft",
        "version": "1.0",
        "force": False,
        "clean_audio": False,
        "summarize": False,
        "no_retry": False,
    }
    
    # Initialize the processor
    processor = AudioProcessor(settings, manifest)
    
    # Check that the processor was initialized with the correct settings
    assert processor.settings == settings
    assert processor.manifest == manifest


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
        # Write some dummy content
        temp_file.write(b"dummy audio content")
        temp_file.flush()
        yield Path(temp_file.name)


def test_process_file_skip_already_processed(temp_audio_file):
    """Test that files are skipped if already processed."""
    # Create a mock manifest
    manifest = MagicMock(spec=ProcessingManifest)
    manifest.is_processed.return_value = True
    
    # Create settings
    settings = {
        "tier": "draft",
        "version": "1.0",
        "force": False,
        "clean_audio": False,
        "summarize": False,
        "no_retry": False,
    }
    
    # Initialize the processor
    processor = AudioProcessor(settings, manifest)
    
    # Process the file
    with patch("builtins.print") as mock_print:
        result = processor.process_file(temp_audio_file)
    
    # The file should be skipped
    assert result is True
    manifest.is_processed.assert_called_once()
    mock_print.assert_called_once()
    assert "Skipping" in mock_print.call_args[0][0]


def test_process_file_force_reprocess(temp_audio_file):
    """Test that force flag causes files to be reprocessed."""
    # Create a mock manifest
    manifest = MagicMock(spec=ProcessingManifest)
    manifest.is_processed.return_value = True
    manifest.get_status.return_value = None
    manifest.get_checkpoint.return_value = None
    
    # Create settings with force=True
    settings = {
        "tier": "draft",
        "version": "1.0",
        "force": True,
        "clean_audio": False,
        "summarize": False,
        "no_retry": False,
        "output_dir": "./output",
    }
    
    # Initialize the processor
    processor = AudioProcessor(settings, manifest)
    
    # Mock the transcription methods
    processor._transcribe_draft = MagicMock()
    
    # Process the file
    with patch("pathlib.Path.mkdir"):
        result = processor.process_file(temp_audio_file)
    
    # The file should be processed despite being already processed
    assert result is True
    processor._transcribe_draft.assert_called_once()
    manifest.update_file_status.assert_called()


def test_process_file_with_clean_audio(temp_audio_file):
    """Test processing a file with clean_audio flag."""
    # Create a mock manifest
    manifest = MagicMock(spec=ProcessingManifest)
    manifest.is_processed.return_value = False
    manifest.get_status.return_value = None
    manifest.get_checkpoint.return_value = None
    
    # Create settings with clean_audio=True
    settings = {
        "tier": "draft",
        "version": "1.0",
        "force": False,
        "clean_audio": True,
        "summarize": False,
        "no_retry": False,
        "output_dir": "./output",
    }
    
    # Initialize the processor
    processor = AudioProcessor(settings, manifest)
    
    # Mock the transcription methods
    processor._transcribe_draft = MagicMock()
    
    # Process the file
    with patch("pathlib.Path.mkdir"):
        result = processor.process_file(temp_audio_file)
    
    # The file should be processed
    assert result is True
    processor._transcribe_draft.assert_called_once()


def test_process_file_with_summarize(temp_audio_file):
    """Test processing a file with summarize flag."""
    # Create a mock manifest
    manifest = MagicMock(spec=ProcessingManifest)
    manifest.is_processed.return_value = False
    manifest.get_status.return_value = None
    manifest.get_checkpoint.return_value = None
    
    # Create settings with summarize=True
    settings = {
        "tier": "draft",
        "version": "1.0",
        "force": False,
        "clean_audio": False,
        "summarize": True,
        "no_retry": False,
        "output_dir": "./output",
    }
    
    # Initialize the processor
    processor = AudioProcessor(settings, manifest)
    
    # Mock the transcription and summary methods
    processor._transcribe_draft = MagicMock()
    processor._generate_summary = MagicMock()
    
    # Process the file
    with patch("pathlib.Path.mkdir"):
        result = processor.process_file(temp_audio_file)
    
    # The file should be processed and summarized
    assert result is True
    processor._transcribe_draft.assert_called_once()
    processor._generate_summary.assert_called_once()


def test_process_file_high_quality(temp_audio_file):
    """Test processing a file with high_quality tier."""
    # Create a mock manifest
    manifest = MagicMock(spec=ProcessingManifest)
    manifest.is_processed.return_value = False
    manifest.get_status.return_value = None
    manifest.get_checkpoint.return_value = None
    
    # Create settings with tier=high_quality
    settings = {
        "tier": "high_quality",
        "version": "1.0",
        "force": False,
        "clean_audio": False,
        "summarize": False,
        "no_retry": False,
        "output_dir": "./output",
    }
    
    # Initialize the processor
    processor = AudioProcessor(settings, manifest)
    
    # Mock the transcription methods
    processor._transcribe_high_quality = MagicMock()
    
    # Process the file
    with patch("pathlib.Path.mkdir"):
        result = processor.process_file(temp_audio_file)
    
    # The high-quality transcription should be used
    assert result is True
    processor._transcribe_high_quality.assert_called_once()


def test_process_file_with_error_and_retry(temp_audio_file):
    """Test processing a file with an error that triggers a retry."""
    # Create a mock manifest
    manifest = MagicMock(spec=ProcessingManifest)
    manifest.is_processed.return_value = False
    manifest.get_status.return_value = None
    manifest.get_checkpoint.return_value = None
    
    # Create settings with no_retry=False
    settings = {
        "tier": "draft",
        "version": "1.0",
        "force": False,
        "clean_audio": False,
        "summarize": False,
        "no_retry": False,
        "output_dir": "./output",
    }
    
    # Initialize the processor
    processor = AudioProcessor(settings, manifest)
    
    # Mock the transcription methods to raise an error
    processor._transcribe_draft = MagicMock(side_effect=[Exception("Test error"), None])
    
    # Process the file
    with patch("pathlib.Path.mkdir"), patch("builtins.print"), patch("time.sleep"):
        # Mock process_file to track calls and avoid infinite recursion
        original_process_file = processor.process_file
        call_count = [0]
        
        def mock_process_file(file_path):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call, trigger the error and retry
                return original_process_file(file_path)
            else:
                # Second call (retry), succeed
                return True
        
        processor.process_file = mock_process_file
        
        result = processor.process_file(temp_audio_file)
    
    # The file should be processed successfully after retry
    assert result is True
    assert call_count[0] == 2  # Original call + one retry
    manifest.update_file_status.assert_called()


def test_process_file_with_error_no_retry(temp_audio_file):
    """Test processing a file with an error and no retry."""
    # Create a mock manifest
    manifest = MagicMock(spec=ProcessingManifest)
    manifest.is_processed.return_value = False
    manifest.get_status.return_value = None
    manifest.get_checkpoint.return_value = None
    
    # Create settings with no_retry=True
    settings = {
        "tier": "draft",
        "version": "1.0",
        "force": False,
        "clean_audio": False,
        "summarize": False,
        "no_retry": True,
        "output_dir": "./output",
    }
    
    # Initialize the processor
    processor = AudioProcessor(settings, manifest)
    
    # Mock the transcription methods to raise an error
    processor._transcribe_draft = MagicMock(side_effect=Exception("Test error"))
    
    # Process the file
    with patch("pathlib.Path.mkdir"), patch("builtins.print"):
        result = processor.process_file(temp_audio_file)
    
    # The file should fail processing
    assert result is False
    manifest.update_file_status.assert_called_with(
        manifest.get_file_hash.return_value,
        "error",
        settings["tier"],
        settings["version"],
        manifest.get_checkpoint.return_value,
        "Test error",
    )