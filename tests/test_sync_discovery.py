"""Tests for the sync file discovery module."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioscript.sync.discovery import FileDiscovery, FileEntry


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def discovery(tmp_dir):
    """Create a FileDiscovery instance with a temp cache path."""
    cache_path = tmp_dir / ".cache.json"
    return FileDiscovery(cache_path)


# --- scan ---


def test_scan_finds_m4a_files(tmp_dir, discovery):
    """scan finds .m4a files in a directory."""
    (tmp_dir / "recording.m4a").write_bytes(b"x" * 2048)
    (tmp_dir / "notes.m4a").write_bytes(b"x" * 2048)
    entries = discovery.scan(tmp_dir, min_file_size=0)
    assert len(entries) == 2
    names = {e.path.name for e in entries}
    assert names == {"recording.m4a", "notes.m4a"}


def test_scan_respects_extensions_filter(tmp_dir, discovery):
    """scan only returns files matching the extensions filter."""
    (tmp_dir / "song.mp3").write_bytes(b"x" * 2048)
    (tmp_dir / "readme.txt").write_bytes(b"x" * 2048)
    entries = discovery.scan(tmp_dir, extensions={"mp3"}, min_file_size=0)
    assert len(entries) == 1
    assert entries[0].path.name == "song.mp3"


def test_scan_respects_min_file_size(tmp_dir, discovery):
    """scan skips files smaller than min_file_size."""
    (tmp_dir / "tiny.m4a").write_bytes(b"x" * 10)
    (tmp_dir / "big.m4a").write_bytes(b"x" * 5000)
    entries = discovery.scan(tmp_dir, min_file_size=1000)
    assert len(entries) == 1
    assert entries[0].path.name == "big.m4a"


def test_scan_respects_ignore_patterns(tmp_dir, discovery):
    """scan skips files matching ignore patterns."""
    (tmp_dir / "audio.m4a").write_bytes(b"x" * 2048)
    (tmp_dir / "temp.tmp").write_bytes(b"x" * 2048)
    entries = discovery.scan(tmp_dir, ignore_patterns=["*.tmp"], min_file_size=0)
    assert len(entries) == 1
    assert entries[0].path.name == "audio.m4a"


def test_scan_recursive_finds_subdirectory_files(tmp_dir, discovery):
    """scan with recursive=True finds files in subdirectories."""
    sub = tmp_dir / "subdir"
    sub.mkdir()
    (tmp_dir / "top.m4a").write_bytes(b"x" * 2048)
    (sub / "nested.m4a").write_bytes(b"x" * 2048)
    entries = discovery.scan(tmp_dir, recursive=True, min_file_size=0)
    assert len(entries) == 2
    names = {e.path.name for e in entries}
    assert names == {"top.m4a", "nested.m4a"}


def test_scan_non_recursive_skips_subdirectories(tmp_dir, discovery):
    """scan with recursive=False only finds top-level files."""
    sub = tmp_dir / "subdir"
    sub.mkdir()
    (tmp_dir / "top.m4a").write_bytes(b"x" * 2048)
    (sub / "nested.m4a").write_bytes(b"x" * 2048)
    entries = discovery.scan(tmp_dir, recursive=False, min_file_size=0)
    assert len(entries) == 1
    assert entries[0].path.name == "top.m4a"


# --- probe_availability ---


def test_probe_availability_classifies_readable_as_local(tmp_dir, discovery):
    """probe_availability classifies readable files as 'local'."""
    f = tmp_dir / "readable.m4a"
    f.write_bytes(b"x" * 2048)
    entry = FileEntry(path=f, size=2048, mtime=time.time())
    local, cloud = discovery.probe_availability([entry])
    assert len(local) == 1
    assert len(cloud) == 0
    assert local[0].status == "local"


def test_probe_availability_classifies_oserror_as_cloud(discovery):
    """probe_availability classifies OSError files as 'cloud'."""
    entry = FileEntry(path=Path("/nonexistent/cloud_file.m4a"), size=2048, mtime=time.time())
    local, cloud = discovery.probe_availability([entry])
    assert len(local) == 0
    assert len(cloud) == 1
    assert cloud[0].status == "cloud"


# --- compute_hashes ---


def test_compute_hashes_caches_by_size_mtime(tmp_dir, discovery):
    """compute_hashes caches hashes by (size, mtime) and reuses them."""
    f = tmp_dir / "audio.m4a"
    f.write_bytes(b"x" * 2048)
    stat = f.stat()

    entry = FileEntry(path=f, size=stat.st_size, mtime=stat.st_mtime)

    with patch("audioscript.sync.discovery.get_file_hash", return_value="abc123") as mock_hash:
        discovery.compute_hashes([entry])
        assert entry.hash == "abc123"
        assert mock_hash.call_count == 1

        # Second call with same size/mtime should use cache
        entry2 = FileEntry(path=f, size=stat.st_size, mtime=stat.st_mtime)
        discovery.compute_hashes([entry2])
        assert entry2.hash == "abc123"
        assert mock_hash.call_count == 1  # no additional call


def test_compute_hashes_recomputes_when_mtime_changes(tmp_dir, discovery):
    """compute_hashes recomputes when mtime changes."""
    f = tmp_dir / "audio.m4a"
    f.write_bytes(b"x" * 2048)
    stat = f.stat()

    entry1 = FileEntry(path=f, size=stat.st_size, mtime=stat.st_mtime)

    with patch("audioscript.sync.discovery.get_file_hash", return_value="hash1") as mock_hash:
        discovery.compute_hashes([entry1])
        assert entry1.hash == "hash1"
        assert mock_hash.call_count == 1

    # Different mtime triggers recomputation
    entry2 = FileEntry(path=f, size=stat.st_size, mtime=stat.st_mtime + 100)

    with patch("audioscript.sync.discovery.get_file_hash", return_value="hash2") as mock_hash:
        discovery.compute_hashes([entry2])
        assert entry2.hash == "hash2"
        assert mock_hash.call_count == 1


# --- diff_against_manifest ---


def test_diff_against_manifest_returns_non_processed(discovery):
    """diff_against_manifest returns entries not in the manifest."""
    manifest = MagicMock()
    manifest.is_processed.return_value = False

    entry = FileEntry(path=Path("/tmp/new.m4a"), size=2048, mtime=1.0, hash="abc123")
    result = discovery.diff_against_manifest([entry], manifest, "draft", "1.0")
    assert len(result) == 1
    assert result[0].status == "new"


def test_diff_against_manifest_skips_completed(discovery):
    """diff_against_manifest skips entries already completed in the manifest."""
    manifest = MagicMock()
    manifest.is_processed.return_value = True

    entry = FileEntry(path=Path("/tmp/done.m4a"), size=2048, mtime=1.0, hash="abc123")
    result = discovery.diff_against_manifest([entry], manifest, "draft", "1.0")
    assert len(result) == 0
    assert entry.status == "skip"
