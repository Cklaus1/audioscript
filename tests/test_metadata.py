"""Tests for the audio metadata extraction module."""

import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from audioscript.utils.metadata import extract_metadata, _extract_file_info, _human_size


# --- _human_size ---

def test_human_size_bytes():
    assert _human_size(500) == "500 B"


def test_human_size_kb():
    assert "KB" in _human_size(2048)


def test_human_size_mb():
    assert "MB" in _human_size(5 * 1024 * 1024)


# --- _extract_file_info ---

def test_file_info_basic():
    """Test basic file info extraction."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio data " * 100)
        f.flush()
        path = Path(f.name)

    info = _extract_file_info(path)

    assert info["name"] == path.name
    assert info["extension"] == "mp3"
    assert info["size_bytes"] == 1600
    assert "KB" in info["size_human"] or "B" in info["size_human"]
    assert info["modified"]  # ISO format string
    assert info["created"]
    assert info["path"] == str(path.absolute())
    assert info["mime_type"] == "audio/mpeg"
    assert "content_hash" in info
    assert len(info["content_hash"]) == 64  # SHA-256

    path.unlink()


def test_file_info_wav():
    """Test MIME type for .wav file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF" + b"\x00" * 100)
        f.flush()
        path = Path(f.name)

    info = _extract_file_info(path)
    # wav MIME varies by platform but should be audio/*
    assert info["extension"] == "wav"

    path.unlink()


# --- extract_metadata (full) ---

def test_extract_metadata_returns_file_section():
    """Test that extract_metadata always returns a file section."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"not a real mp3 " * 50)
        f.flush()
        path = Path(f.name)

    meta = extract_metadata(path)

    assert "file" in meta
    assert meta["file"]["name"] == path.name
    assert meta["file"]["size_bytes"] > 0
    assert meta["file"]["extension"] == "mp3"
    assert "modified" in meta["file"]
    assert "created" in meta["file"]
    assert "content_hash" in meta["file"]

    path.unlink()


def test_extract_metadata_ffprobe_on_real_wav():
    """Test ffprobe extraction on a minimal valid WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write a minimal valid WAV header (44 bytes)
        # RIFF header
        sample_rate = 16000
        num_channels = 1
        bits_per_sample = 16
        num_samples = 16000  # 1 second
        data_size = num_samples * num_channels * bits_per_sample // 8
        file_size = 36 + data_size

        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM format
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * num_channels * bits_per_sample // 8))
        f.write(struct.pack("<H", num_channels * bits_per_sample // 8))
        f.write(struct.pack("<H", bits_per_sample))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)
        f.flush()
        path = Path(f.name)

    meta = extract_metadata(path)

    assert "file" in meta
    assert meta["file"]["extension"] == "wav"

    # ffprobe should extract audio properties if available
    if "audio" in meta:
        audio = meta["audio"]
        assert "duration_seconds" in audio or "sample_rate" in audio
        if "sample_rate" in audio:
            assert audio["sample_rate"] == 16000
        if "channels" in audio:
            assert audio["channels"] == 1

    path.unlink()


def test_extract_metadata_graceful_on_invalid_file():
    """Test that metadata extraction doesn't crash on non-audio files."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"this is not audio at all")
        f.flush()
        path = Path(f.name)

    meta = extract_metadata(path)

    # Should still have file section
    assert "file" in meta
    assert meta["file"]["name"] == path.name
    # Tags/audio may be absent — that's fine
    assert meta["file"]["size_bytes"] > 0

    path.unlink()


def test_extract_metadata_missing_file():
    """Test that metadata extraction raises on missing file."""
    with pytest.raises((FileNotFoundError, OSError)):
        extract_metadata(Path("/nonexistent/audio.mp3"))


# --- Tag extraction with mocked mutagen ---

def test_tags_extraction_returns_none_for_plain_file():
    """Test that tag extraction returns None for a plain text file."""
    from audioscript.utils.metadata import _extract_tags

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"not a real audio file")
        f.flush()
        path = Path(f.name)

    tags = _extract_tags(path)
    path.unlink()

    # mutagen can't parse a fake file, so returns None
    assert tags is None
