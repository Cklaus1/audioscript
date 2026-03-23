"""Tests for the WhisperTranscriber."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioscript.processors.whisper_transcriber import WhisperTranscriber
from audioscript.processors.backend_protocol import TranscriptionResult


@pytest.fixture
def mock_whisper():
    """Mock the whisper module."""
    with patch("audioscript.processors.whisper_transcriber.whisper") as mock_mod:
        mock_mod.available_models.return_value = [
            "tiny", "tiny.en", "base", "base.en", "small", "medium",
            "large", "large-v2", "large-v3", "turbo",
        ]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Hello world.",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": " Hello world."},
            ],
        }
        mock_mod.load_model.return_value = mock_model
        # For detect_language
        mock_mod.load_audio.return_value = MagicMock()
        mock_mod.pad_or_trim.return_value = MagicMock()
        mock_mod.log_mel_spectrogram.return_value = MagicMock(to=MagicMock(return_value=MagicMock()))
        yield mock_mod, mock_model


# --- Initialization ---

def test_init_default_model(mock_whisper):
    t = WhisperTranscriber(tier="draft")
    assert t.model_name == "base"


def test_init_balanced_tier(mock_whisper):
    t = WhisperTranscriber(tier="balanced")
    assert t.model_name == "turbo"


def test_init_high_quality_tier(mock_whisper):
    t = WhisperTranscriber(tier="high_quality")
    assert t.model_name == "large-v3"


def test_init_explicit_model(mock_whisper):
    t = WhisperTranscriber(model_name="medium", tier="draft")
    assert t.model_name == "medium"


def test_init_invalid_model(mock_whisper):
    with pytest.raises(ValueError, match="Unknown model"):
        WhisperTranscriber(model_name="nonexistent")


def test_init_en_model(mock_whisper):
    t = WhisperTranscriber(model_name="base.en")
    assert t.model_name == "base.en"


def test_init_download_root(mock_whisper):
    t = WhisperTranscriber(download_root="/tmp/models")
    assert t.download_root == "/tmp/models"


# --- Protocol compliance ---

def test_backend_name(mock_whisper):
    t = WhisperTranscriber()
    assert t.backend_name == "whisper"


def test_supports_confidence(mock_whisper):
    t = WhisperTranscriber()
    assert t.supports_confidence is False


# --- Model loading ---

def test_lazy_model_loading(mock_whisper):
    mock_mod, mock_model = mock_whisper
    t = WhisperTranscriber()
    assert t.model is None
    t.load_model()
    mock_mod.load_model.assert_called_once()
    assert t.model is not None


def test_model_loads_once(mock_whisper):
    mock_mod, _ = mock_whisper
    t = WhisperTranscriber()
    t.load_model()
    t.load_model()
    assert mock_mod.load_model.call_count == 1


def test_download_root_passed(mock_whisper):
    mock_mod, _ = mock_whisper
    t = WhisperTranscriber(download_root="/custom/cache")
    t.load_model()
    call_kwargs = mock_mod.load_model.call_args
    assert call_kwargs.kwargs.get("download_root") == "/custom/cache"


# --- Transcription ---

def test_transcribe_basic(mock_whisper):
    _, mock_model = mock_whisper
    t = WhisperTranscriber()
    result = t.transcribe("/tmp/audio.mp3")
    mock_model.transcribe.assert_called_once()
    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello world."
    assert result.language == "en"
    assert len(result.segments) == 1
    assert result.backend == "whisper"


def test_transcribe_passes_params(mock_whisper):
    _, mock_model = mock_whisper
    t = WhisperTranscriber()
    t.transcribe(
        "/tmp/audio.mp3",
        language="en", beam_size=10, word_timestamps=True,
        temperature=(0.0, 0.4), best_of=5,
    )
    call_kwargs = mock_model.transcribe.call_args
    assert call_kwargs.kwargs["language"] == "en"
    assert call_kwargs.kwargs["beam_size"] == 10
    assert call_kwargs.kwargs["word_timestamps"] is True
    assert call_kwargs.kwargs["temperature"] == (0.0, 0.4)
    assert call_kwargs.kwargs["best_of"] == 5


def test_transcribe_with_checkpoint(mock_whisper):
    _, mock_model = mock_whisper
    t = WhisperTranscriber()
    checkpoint = json.dumps({"text": "previous context"})
    t.transcribe("/tmp/audio.mp3", checkpoint=checkpoint)
    call_kwargs = mock_model.transcribe.call_args
    assert call_kwargs.kwargs["initial_prompt"] == "previous context"


def test_transcribe_invalid_checkpoint(mock_whisper):
    _, mock_model = mock_whisper
    t = WhisperTranscriber()
    t.transcribe("/tmp/audio.mp3", checkpoint="not json")
    call_kwargs = mock_model.transcribe.call_args
    assert call_kwargs.kwargs["initial_prompt"] is None


def test_transcribe_path_object(mock_whisper):
    _, mock_model = mock_whisper
    t = WhisperTranscriber()
    t.transcribe(Path("/tmp/audio.mp3"))
    call_args = mock_model.transcribe.call_args
    assert isinstance(call_args[0][0], str)


def test_transcribe_to_dict(mock_whisper):
    """Test that TranscriptionResult.to_dict() returns the expected format."""
    t = WhisperTranscriber()
    result = t.transcribe("/tmp/audio.mp3")
    d = result.to_dict()
    assert "text" in d
    assert "segments" in d
    assert "language" in d
    assert "backend" in d
    assert d["backend"] == "whisper"


# --- Duplicate segment filtering ---

def test_filters_duplicate_segments(mock_whisper):
    _, mock_model = mock_whisper
    mock_model.transcribe.return_value = {
        "text": "Hello Hello Goodbye",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": " Hello"},
            {"start": 1.0, "end": 2.0, "text": " Hello"},  # duplicate
            {"start": 2.0, "end": 3.0, "text": " Goodbye"},
        ],
    }
    t = WhisperTranscriber()
    result = t.transcribe("/tmp/audio.mp3")
    assert len(result.segments) == 2
    assert "Hello" in result.text
    assert "Goodbye" in result.text


def test_filters_empty_segments(mock_whisper):
    _, mock_model = mock_whisper
    mock_model.transcribe.return_value = {
        "text": "Hello",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": " Hello"},
        ],
    }
    t = WhisperTranscriber()
    result = t.transcribe("/tmp/audio.mp3")
    assert len(result.segments) == 1


# --- Save/load methods ---

def test_save_results(mock_whisper):
    t = WhisperTranscriber()
    transcription = {"text": "Hello", "segments": [{"id": 0}]}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        t.save_results(transcription, out)
        with open(out) as f:
            data = json.load(f)
        assert data["text"] == "Hello"
        assert len(data["segments"]) == 1


def test_save_results_no_segments(mock_whisper):
    t = WhisperTranscriber()
    transcription = {"text": "Hello", "segments": [{"id": 0}]}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        t.save_results(transcription, out, include_segments=False)
        with open(out) as f:
            data = json.load(f)
        assert "segments" not in data


def test_save_summary(mock_whisper):
    t = WhisperTranscriber()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "summary.txt"
        t.save_summary("This is a summary.", out)
        assert out.read_text() == "This is a summary."


# --- Summary generation ---

def test_generate_summary_short(mock_whisper):
    t = WhisperTranscriber()
    result = t.generate_summary({"text": "Short text."})
    assert result == "Short text."


def test_generate_summary_truncates(mock_whisper):
    t = WhisperTranscriber()
    long_text = " ".join(f"word{i}" for i in range(50))
    result = t.generate_summary({"text": long_text})
    assert result.endswith("...")
    assert len(result.split()) <= 26  # 25 words + "..."


def test_generate_summary_empty(mock_whisper):
    t = WhisperTranscriber()
    assert t.generate_summary({}) == ""
    assert t.generate_summary({"text": ""}) == ""


# --- Checkpoint ---

def test_create_checkpoint(mock_whisper):
    t = WhisperTranscriber()
    cp = t.create_checkpoint({"text": "Hello world"})
    data = json.loads(cp)
    assert data["text"] == "Hello world"


def test_create_checkpoint_empty(mock_whisper):
    t = WhisperTranscriber()
    cp = t.create_checkpoint({})
    data = json.loads(cp)
    assert data["text"] == ""


# --- Formatted output ---

def test_save_formatted_output(mock_whisper):
    mock_mod, _ = mock_whisper
    mock_writer = MagicMock()
    with patch("audioscript.processors.whisper_transcriber.get_writer", return_value=mock_writer):
        t = WhisperTranscriber()
        result = {"text": "Hello", "segments": []}
        with tempfile.TemporaryDirectory() as tmp:
            t.save_formatted_output(result, "/tmp/audio.mp3", tmp, "srt")
            mock_writer.assert_called_once()
