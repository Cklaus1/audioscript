"""Tests for the sync engine module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioscript.config.settings import (
    AudioScriptConfig,
    SyncConfig,
    SyncMiNotesConfig,
    SyncOneDriveConfig,
    SyncSourceConfig,
    TranscriptionTier,
)
from audioscript.sync.discovery import FileEntry
from audioscript.sync.engine import SyncEngine, SyncReport


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def global_config():
    """Create a minimal global AudioScriptConfig."""
    return AudioScriptConfig(
        output_dir="./output",
        output_format="json",
        tier=TranscriptionTier.DRAFT,
    )


def _make_sync_config(sources=None, output_dir="./output", **kwargs):
    """Build a SyncConfig with sensible defaults."""
    return SyncConfig(
        sources=sources or [],
        output_dir=output_dir,
        output_format="json",
        **kwargs,
    )


# --- run_once ---


def test_run_once_empty_sources(global_config, tmp_dir):
    """run_once with empty sources returns empty report."""
    sync_config = _make_sync_config(sources=[], output_dir=str(tmp_dir))
    engine = SyncEngine(sync_config, global_config)
    report = engine.run_once()
    assert isinstance(report, SyncReport)
    assert len(report.sources) == 0
    assert report.total_scanned == 0


def test_run_once_calls_discovery_and_processor(global_config, tmp_dir):
    """run_once calls discovery.scan and processor.process_file for new files."""
    source = SyncSourceConfig(path=str(tmp_dir))
    sync_config = _make_sync_config(sources=[source], output_dir=str(tmp_dir), batch_size=0)

    entry = FileEntry(path=tmp_dir / "test.m4a", size=2048, mtime=1.0, hash="abc123", status="new")

    with patch("audioscript.sync.engine.resolve_sync_path", return_value=tmp_dir):
        with patch("audioscript.sync.engine.FileDiscovery") as MockDiscovery:
            mock_disc = MockDiscovery.return_value
            mock_disc.scan.return_value = [entry]
            mock_disc.probe_availability.return_value = ([entry], [])
            mock_disc.compute_hashes.return_value = [entry]
            mock_disc.diff_against_manifest.return_value = [entry]

            with patch("audioscript.sync.engine.AudioProcessor") as MockProcessor:
                mock_proc = MockProcessor.return_value
                mock_proc.process_file.return_value = True

                engine = SyncEngine(sync_config, global_config)
                report = engine.run_once()

                assert report.total_scanned == 1
                assert report.total_transcribed == 1
                mock_proc.process_file.assert_called_once()


def test_run_once_respects_batch_size(global_config, tmp_dir):
    """run_once respects batch_size limit."""
    source = SyncSourceConfig(path=str(tmp_dir))
    sync_config = _make_sync_config(sources=[source], output_dir=str(tmp_dir), batch_size=2)

    entries = [
        FileEntry(path=tmp_dir / f"file{i}.m4a", size=2048, mtime=float(i), hash=f"hash{i}", status="new")
        for i in range(5)
    ]

    with patch("audioscript.sync.engine.resolve_sync_path", return_value=tmp_dir):
        with patch("audioscript.sync.engine.FileDiscovery") as MockDiscovery:
            mock_disc = MockDiscovery.return_value
            mock_disc.scan.return_value = entries
            mock_disc.probe_availability.return_value = (entries, [])
            mock_disc.compute_hashes.return_value = entries
            mock_disc.diff_against_manifest.return_value = entries

            with patch("audioscript.sync.engine.AudioProcessor") as MockProcessor:
                mock_proc = MockProcessor.return_value
                mock_proc.process_file.return_value = True

                engine = SyncEngine(sync_config, global_config)
                report = engine.run_once()

                # Only 2 files should be processed due to batch_size
                assert mock_proc.process_file.call_count == 2


def test_run_once_skips_already_processed(global_config, tmp_dir):
    """run_once skips already-processed files."""
    source = SyncSourceConfig(path=str(tmp_dir))
    sync_config = _make_sync_config(sources=[source], output_dir=str(tmp_dir))

    entry = FileEntry(path=tmp_dir / "test.m4a", size=2048, mtime=1.0, hash="abc123", status="new")

    with patch("audioscript.sync.engine.resolve_sync_path", return_value=tmp_dir):
        with patch("audioscript.sync.engine.FileDiscovery") as MockDiscovery:
            mock_disc = MockDiscovery.return_value
            mock_disc.scan.return_value = [entry]
            mock_disc.probe_availability.return_value = ([entry], [])
            mock_disc.compute_hashes.return_value = [entry]
            # diff returns empty list (all already processed)
            mock_disc.diff_against_manifest.return_value = []

            with patch("audioscript.sync.engine.AudioProcessor") as MockProcessor:
                mock_proc = MockProcessor.return_value

                engine = SyncEngine(sync_config, global_config)
                report = engine.run_once()

                assert report.total_new == 0
                mock_proc.process_file.assert_not_called()


# --- _merge_source_settings ---


def test_merge_source_settings_applies_tier_override(global_config, tmp_dir):
    """_merge_source_settings applies per-source tier override."""
    sync_config = _make_sync_config(output_dir=str(tmp_dir))
    engine = SyncEngine(sync_config, global_config)

    source = SyncSourceConfig(path=str(tmp_dir), tier=TranscriptionTier.HIGH_QUALITY)
    merged = engine._merge_source_settings(source)

    assert merged.tier == TranscriptionTier.HIGH_QUALITY


def test_merge_source_settings_applies_minotes_export(global_config, tmp_dir):
    """_merge_source_settings applies MiNotes export when enabled."""
    sync_config = _make_sync_config(
        output_dir=str(tmp_dir),
        minotes=SyncMiNotesConfig(enabled=True),
    )
    engine = SyncEngine(sync_config, global_config)

    source = SyncSourceConfig(path=str(tmp_dir))
    merged = engine._merge_source_settings(source)

    assert merged.export == "minotes"


# --- _is_slow_filesystem ---


def test_is_slow_filesystem_detects_mnt_paths(global_config, tmp_dir):
    """_is_slow_filesystem detects /mnt/ paths in WSL."""
    sync_config = _make_sync_config(output_dir=str(tmp_dir))
    engine = SyncEngine(sync_config, global_config)

    with patch("audioscript.sync.engine.is_wsl", return_value=True):
        assert engine._is_slow_filesystem(Path("/mnt/c/Users")) is True

    with patch("audioscript.sync.engine.is_wsl", return_value=False):
        assert engine._is_slow_filesystem(Path("/mnt/c/Users")) is False


# --- _trigger_download ---


def test_trigger_download_calls_attrib_exe(global_config, tmp_dir):
    """_trigger_download calls attrib.exe via subprocess in WSL."""
    sync_config = _make_sync_config(output_dir=str(tmp_dir))
    engine = SyncEngine(sync_config, global_config)

    wslpath_result = MagicMock()
    wslpath_result.returncode = 0
    wslpath_result.stdout = r"C:\Users\test\audio.m4a" + "\n"

    attrib_result = MagicMock()
    attrib_result.returncode = 0

    with patch("audioscript.sync.engine.is_wsl", return_value=True):
        with patch("subprocess.run", side_effect=[wslpath_result, attrib_result]) as mock_run:
            engine._trigger_download(Path("/mnt/c/Users/test/audio.m4a"))

            assert mock_run.call_count == 2
            # First call: wslpath
            assert mock_run.call_args_list[0][0][0][0] == "wslpath"
            # Second call: attrib.exe
            assert mock_run.call_args_list[1][0][0][0] == "attrib.exe"
