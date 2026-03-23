"""Tests for the MiNotes exporter module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioscript.exporters.minotes_exporter import MiNotesExporter


# --- is_available ---

def test_is_available_true_when_cli_exists():
    """is_available returns True when minotes CLI returns 0."""
    mock_result = MagicMock(returncode=0)
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        exporter = MiNotesExporter(sync_dir="/tmp/test-sync")
        assert exporter.is_available() is True
        mock_run.assert_called_once()


def test_is_available_false_when_not_found():
    """is_available returns False when FileNotFoundError."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        exporter = MiNotesExporter(sync_dir="/tmp/test-sync")
        assert exporter.is_available() is False


# --- ensure_registered ---

def test_ensure_registered_calls_subprocess():
    """ensure_registered calls subprocess with plugin register args."""
    with patch("subprocess.run") as mock_run:
        exporter = MiNotesExporter(sync_dir="/tmp/test-sync")
        exporter.ensure_registered()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "minotes" in args
        assert "plugin" in args
        assert "register" in args
        assert "--name" in args
        assert "audioscript" in args


def test_ensure_registered_is_idempotent():
    """ensure_registered only calls subprocess once."""
    with patch("subprocess.run") as mock_run:
        exporter = MiNotesExporter(sync_dir="/tmp/test-sync")
        exporter.ensure_registered()
        exporter.ensure_registered()
        assert mock_run.call_count == 1


# --- export ---

def test_export_writes_markdown_to_sync_dir():
    """export writes markdown file to sync_dir/transcripts/."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("subprocess.run"):
            exporter = MiNotesExporter(sync_dir=tmp)
            audio_path = Path("/tmp/interview.mp3")
            result_dict = {"language": "en"}
            output = exporter.export("# Hello", audio_path, result_dict)
            assert output.exists()
            assert output.parent.name == "transcripts"
            assert output.name == "interview.md"
            assert output.read_text() == "# Hello"


def test_export_updates_sync_state():
    """export updates the sync state file."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("subprocess.run"):
            exporter = MiNotesExporter(sync_dir=tmp)
            audio_path = Path("/tmp/meeting.mp3")
            exporter.export("# Content", audio_path, {"language": "en"})
            state_file = Path(tmp) / ".audioscript_sync_state.json"
            assert state_file.exists()
            state = json.loads(state_file.read_text())
            assert "meeting" in state
            assert "synced_at" in state["meeting"]


# --- is_already_exported ---

def test_is_already_exported_true_after_export():
    """is_already_exported returns True after a file has been exported."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("subprocess.run"):
            exporter = MiNotesExporter(sync_dir=tmp)
            audio_path = Path("/tmp/talk.mp3")
            exporter.export("# Talk", audio_path, {})
            assert exporter.is_already_exported(audio_path) is True


def test_is_already_exported_false_for_unknown():
    """is_already_exported returns False for unknown files."""
    with tempfile.TemporaryDirectory() as tmp:
        exporter = MiNotesExporter(sync_dir=tmp)
        assert exporter.is_already_exported(Path("/tmp/unknown.mp3")) is False


# --- journal_entry ---

def test_journal_entry_calls_subprocess():
    """journal_entry calls subprocess with correct args."""
    with patch("subprocess.run") as mock_run:
        exporter = MiNotesExporter(sync_dir="/tmp/test-sync")
        exporter.journal_entry(Path("/tmp/test.mp3"), "A short summary.")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "minotes" in args
        assert "block" in args
        assert "create" in args
        assert "--page" in args
        assert "journal" in args
