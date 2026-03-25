"""Tests for the backend protocol and factory."""

from unittest.mock import MagicMock, patch

from audioscript.processors.backend_protocol import (
    TranscriptionResult,
    TranscriptionSegment,
)


# --- TranscriptionSegment ---

def test_segment_to_dict_omits_none():
    """to_dict() omits fields that are None."""
    seg = TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hello")
    d = seg.to_dict()
    assert "id" in d
    assert "start" in d
    assert "end" in d
    assert "text" in d
    assert "confidence" not in d
    assert "avg_logprob" not in d
    assert "words" not in d


def test_segment_to_dict_includes_set_values():
    """to_dict() includes fields that are set."""
    seg = TranscriptionSegment(
        id=1, start=0.0, end=2.0, text="Hi", confidence=0.9, avg_logprob=-0.3,
    )
    d = seg.to_dict()
    assert d["confidence"] == 0.9
    assert d["avg_logprob"] == -0.3


# --- TranscriptionResult ---

def test_result_to_dict_structure():
    """to_dict() returns correct top-level structure."""
    seg = TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hello")
    result = TranscriptionResult(
        text="Hello", language="en", segments=[seg], backend="faster-whisper",
    )
    d = result.to_dict()
    assert d["text"] == "Hello"
    assert d["language"] == "en"
    assert d["backend"] == "faster-whisper"
    assert len(d["segments"]) == 1
    assert "raw" not in d


# --- create_transcriber factory ---

def test_create_transcriber_returns_faster_whisper():
    """create_transcriber returns FasterWhisperTranscriber."""
    mock_settings = MagicMock()
    mock_settings.backend = "faster-whisper"
    mock_settings.model = None
    mock_settings.tier.value = "draft"
    mock_settings.download_root = None

    with patch("audioscript.processors.faster_whisper_transcriber.FasterWhisperTranscriber.__init__", return_value=None):
        from audioscript.processors import create_transcriber
        transcriber = create_transcriber(mock_settings)

    from audioscript.processors.faster_whisper_transcriber import FasterWhisperTranscriber
    assert isinstance(transcriber, FasterWhisperTranscriber)
