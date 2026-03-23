"""Tests for file utility functions."""

import json
import tempfile
from pathlib import Path

import pytest

from audioscript.utils.file_utils import (
    ProcessingManifest,
    get_file_hash,
    get_output_path,
)


def test_get_file_hash():
    """Test calculating a content-based file hash."""
    with tempfile.NamedTemporaryFile(mode="wb") as temp_file:
        temp_file.write(b"test content")
        temp_file.flush()

        file_path = Path(temp_file.name)
        hash1 = get_file_hash(file_path)

        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex digest

        # Same content = same hash
        hash2 = get_file_hash(file_path)
        assert hash1 == hash2


def test_get_file_hash_changes_with_content():
    """Test that hash changes when file content changes."""
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_file:
        temp_file.write(b"original content")
        temp_file.flush()
        file_path = Path(temp_file.name)

        hash1 = get_file_hash(file_path)

    # Overwrite with different content
    with open(file_path, "wb") as f:
        f.write(b"modified content")

    hash2 = get_file_hash(file_path)
    assert hash1 != hash2

    file_path.unlink()


def test_get_file_hash_same_content_different_files():
    """Test that identical content in different files produces the same hash."""
    content = b"identical content"
    with tempfile.NamedTemporaryFile(mode="wb") as f1, \
         tempfile.NamedTemporaryFile(mode="wb") as f2:
        f1.write(content)
        f1.flush()
        f2.write(content)
        f2.flush()

        assert get_file_hash(Path(f1.name)) == get_file_hash(Path(f2.name))


def test_get_file_hash_missing_file():
    """Test that hashing a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        get_file_hash(Path("/nonexistent/file.mp3"))


def test_get_output_path():
    """Test generating an output path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "test.mp3"
        output_dir = Path(temp_dir) / "output"

        output_path = get_output_path(input_file, output_dir)

        assert output_path.parent == output_dir
        assert output_path.stem == "test"
        assert output_path.suffix == ".json"
        assert output_dir.exists()

        # Test with a different extension
        output_path = get_output_path(input_file, output_dir, "txt")
        assert output_path.suffix == ".txt"


def test_processing_manifest_lifecycle():
    """Test the full manifest lifecycle: create, update, query, persist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"

        # Create a new manifest
        manifest = ProcessingManifest(manifest_path)
        assert manifest.data["version"] == "1.1"
        assert manifest.data["files"] == {}

        # Update a file's status
        manifest.update_file_status(
            "test_hash", "processing", "draft", "1.0", "test_checkpoint",
        )

        assert "test_hash" in manifest.data["files"]
        assert manifest.data["files"]["test_hash"]["status"] == "processing"
        assert manifest.data["files"]["test_hash"]["tier"] == "draft"
        assert manifest.data["files"]["test_hash"]["version"] == "1.0"
        assert manifest.data["files"]["test_hash"]["checkpoint"] == "test_checkpoint"
        assert manifest_path.exists()

        # Not yet completed
        assert not manifest.is_processed("test_hash", "draft", "1.0")

        # Mark as completed
        manifest.update_file_status("test_hash", "completed", "draft", "1.0")

        assert manifest.is_processed("test_hash", "draft", "1.0")
        assert not manifest.is_processed("test_hash", "high_quality", "1.0")
        assert not manifest.is_processed("test_hash", "draft", "2.0")

        # Test get_status
        assert manifest.get_status("test_hash") == "completed"
        assert manifest.get_status("nonexistent_hash") is None


def test_processing_manifest_last_updated():
    """Test that last_updated uses wall-clock time, not file mtime."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"
        manifest = ProcessingManifest(manifest_path)

        manifest.update_file_status("hash1", "processing", "draft", "1.0")

        last_updated = manifest.data["files"]["hash1"]["last_updated"]
        # Should be a recent unix timestamp (not 0, not file mtime)
        assert isinstance(last_updated, float)
        assert last_updated > 1_000_000_000  # After year 2001


def test_processing_manifest_persistence():
    """Test that manifest data survives a reload from disk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"

        manifest1 = ProcessingManifest(manifest_path)
        manifest1.update_file_status("hash1", "completed", "draft", "1.0")

        # Load from disk in a new instance
        manifest2 = ProcessingManifest(manifest_path)
        assert "hash1" in manifest2.data["files"]
        assert manifest2.data["files"]["hash1"]["status"] == "completed"


def test_processing_manifest_atomic_save():
    """Test that save writes valid JSON even on re-read."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"
        manifest = ProcessingManifest(manifest_path)

        manifest.update_file_status("hash1", "completed", "draft", "1.0")

        # Read the raw file and verify it's valid JSON
        with open(manifest_path, "r") as f:
            data = json.load(f)
        assert data["files"]["hash1"]["status"] == "completed"


def test_processing_manifest_error_field():
    """Test that error messages are stored in the manifest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.json"
        manifest = ProcessingManifest(manifest_path)

        manifest.update_file_status(
            "hash1", "error", "draft", "1.0", error="something went wrong",
        )

        assert manifest.data["files"]["hash1"]["error"] == "something went wrong"
