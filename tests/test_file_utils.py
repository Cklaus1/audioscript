"""Tests for file utility functions."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from audioscript.utils.file_utils import (
    ProcessingManifest,
    get_file_hash,
    get_output_path,
)


def test_get_file_hash():
    """Test calculating a file hash."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+") as temp_file:
        temp_file.write("test content")
        temp_file.flush()
        
        # Get the file hash
        file_path = Path(temp_file.name)
        hash1 = get_file_hash(file_path)
        
        # The hash should be a string
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # The hash should be the same for the same file
        hash2 = get_file_hash(file_path)
        assert hash1 == hash2
        
        # The hash should change if the file is modified
        temp_file.write("more content")
        temp_file.flush()
        hash3 = get_file_hash(file_path)
        assert hash1 != hash3


def test_get_output_path():
    """Test generating an output path."""
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "test.mp3"
        output_dir = Path(temp_dir) / "output"
        
        # Get the output path
        output_path = get_output_path(input_file, output_dir)
        
        # The output path should be in the output directory
        assert output_path.parent == output_dir
        
        # The output path should have the same stem as the input file
        assert output_path.stem == "test"
        
        # The output path should have the specified extension
        assert output_path.suffix == ".json"
        
        # The output directory should be created if it doesn't exist
        assert output_dir.exists()
        
        # Test with a different extension
        output_path = get_output_path(input_file, output_dir, "txt")
        assert output_path.suffix == ".txt"


def test_processing_manifest():
    """Test the ProcessingManifest class."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"
        
        # Create a new manifest
        manifest = ProcessingManifest(manifest_path)
        
        # The manifest should be created
        assert manifest.data["version"] == "1.0"
        assert manifest.data["files"] == {}
        
        # Update a file's status
        manifest.update_file_status(
            "test_hash",
            "processing",
            "draft",
            "1.0",
            "test_checkpoint",
        )
        
        # The file should be in the manifest
        assert "test_hash" in manifest.data["files"]
        assert manifest.data["files"]["test_hash"]["status"] == "processing"
        assert manifest.data["files"]["test_hash"]["tier"] == "draft"
        assert manifest.data["files"]["test_hash"]["version"] == "1.0"
        assert manifest.data["files"]["test_hash"]["checkpoint"] == "test_checkpoint"
        
        # The manifest should be saved to disk
        assert manifest_path.exists()
        
        # Check is_processed
        assert not manifest.is_processed("test_hash", "draft", "1.0")
        
        # Update the file status to completed
        manifest.update_file_status(
            "test_hash",
            "completed",
            "draft",
            "1.0",
        )
        
        # The file should be processed now
        assert manifest.is_processed("test_hash", "draft", "1.0")
        
        # But not with a different tier or version
        assert not manifest.is_processed("test_hash", "high_quality", "1.0")
        assert not manifest.is_processed("test_hash", "draft", "2.0")
        
        # Test get_checkpoint
        assert manifest.get_checkpoint("test_hash") is None
        
        # Update with a checkpoint
        manifest.update_file_status(
            "test_hash",
            "processing",
            "draft",
            "1.0",
            "new_checkpoint",
        )
        
        # The checkpoint should be updated
        assert manifest.get_checkpoint("test_hash") == "new_checkpoint"
        
        # Test get_status
        assert manifest.get_status("test_hash") == "processing"
        assert manifest.get_status("nonexistent_hash") is None
        
        # Create a new manifest instance with the same path
        manifest2 = ProcessingManifest(manifest_path)
        
        # It should load the existing manifest
        assert "test_hash" in manifest2.data["files"]
        assert manifest2.data["files"]["test_hash"]["status"] == "processing"