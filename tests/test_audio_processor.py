"""Tests for the audio processor."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioscript.config.settings import AudioScriptConfig, TranscriptionTier
from audioscript.processors.audio_processor import AudioProcessor
from audioscript.processors.backend_protocol import TranscriptionResult, TranscriptionSegment
from audioscript.utils.file_utils import ProcessingManifest


def _make_result(text: str = "hello world") -> TranscriptionResult:
    """Create a mock TranscriptionResult."""
    return TranscriptionResult(
        text=text,
        language="en",
        segments=[
            TranscriptionSegment(id=0, start=0.0, end=2.0, text=text),
        ],
        backend="faster-whisper",
    )


def _make_settings(**overrides) -> AudioScriptConfig:
    """Create an AudioScriptConfig with sensible test defaults."""
    defaults = {
        "tier": TranscriptionTier.DRAFT,
        "version": "1.0",
        "force": False,
        "clean_audio": False,
        "summarize": False,
        "no_retry": True,
        "max_retries": 3,
        "output_dir": "./output",
        "backend": "faster-whisper",
        "hallucination_filter": "off",
    }
    defaults.update(overrides)
    return AudioScriptConfig(**defaults)


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
        temp_file.write(b"dummy audio content")
        temp_file.flush()
        yield Path(temp_file.name)


@pytest.fixture
def mock_manifest():
    """Create a mock ProcessingManifest."""
    manifest = MagicMock(spec=ProcessingManifest)
    manifest.is_processed.return_value = False
    manifest.get_status.return_value = None
    manifest.get_checkpoint.return_value = None
    return manifest


@pytest.fixture
def mock_transcriber():
    """Create a mock TranscriberBackend that returns valid results."""
    transcriber = MagicMock()
    transcriber.transcribe.return_value = _make_result()
    transcriber.backend_name = "faster-whisper"
    transcriber.supports_confidence = False
    return transcriber


def test_initialization(mock_manifest):
    """Test initializing the audio processor."""
    settings = _make_settings()
    processor = AudioProcessor(settings, mock_manifest)

    assert processor.settings == settings
    assert processor.manifest == mock_manifest


def test_skip_already_processed(temp_audio_file, mock_manifest):
    """Test that files are skipped if already processed."""
    mock_manifest.is_processed.return_value = True
    settings = _make_settings()
    processor = AudioProcessor(settings, mock_manifest)

    result = processor.process_file(temp_audio_file)

    assert result is True
    mock_manifest.is_processed.assert_called_once()
    mock_manifest.update_file_status.assert_not_called()


def test_force_reprocess(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that force flag causes files to be reprocessed."""
    mock_manifest.is_processed.return_value = True
    settings = _make_settings(force=True)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        result = processor.process_file(temp_audio_file)

    assert result is True
    mock_transcriber.transcribe.assert_called_once()
    calls = mock_manifest.update_file_status.call_args_list
    statuses = [c[0][1] for c in calls]
    assert "completed" in statuses


def test_transcription_saves_results(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that transcription results are saved."""
    settings = _make_settings()
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results") as mock_save:
        result = processor.process_file(temp_audio_file)

    assert result is True
    mock_transcriber.transcribe.assert_called_once()
    mock_save.assert_called_once()


def test_clean_audio_flag(temp_audio_file, mock_manifest, mock_transcriber):
    """Test processing a file with clean_audio flag."""
    settings = _make_settings(clean_audio=True)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"), \
         patch("audioscript.processors.audio_cleaner.clean_audio",
               return_value=(temp_audio_file, {"snr_before": 15.0, "snr_after": 25.0, "skipped": False, "level": "moderate"})) as mock_clean:
        result = processor.process_file(temp_audio_file)

    assert result is True
    mock_clean.assert_called_once()


def test_summarize_flag(temp_audio_file, mock_manifest, mock_transcriber):
    """Test processing a file with summarize flag."""
    settings = _make_settings(summarize=True)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"), \
         patch("audioscript.processors.audio_processor.open", MagicMock(), create=True):
        result = processor.process_file(temp_audio_file)

    assert result is True


def test_high_quality_tier(temp_audio_file, mock_manifest, mock_transcriber):
    """Test processing a file with high_quality tier."""
    settings = _make_settings(tier=TranscriptionTier.HIGH_QUALITY)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        result = processor.process_file(temp_audio_file)

    assert result is True
    mock_transcriber.transcribe.assert_called_once()


def test_balanced_tier(temp_audio_file, mock_manifest, mock_transcriber):
    """Test processing a file with balanced tier."""
    settings = _make_settings(tier=TranscriptionTier.BALANCED)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        result = processor.process_file(temp_audio_file)

    assert result is True
    mock_transcriber.transcribe.assert_called_once()


def test_error_no_retry(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that no_retry=True fails immediately on error."""
    mock_transcriber.transcribe.side_effect = RuntimeError("model failed")
    settings = _make_settings(no_retry=True)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"):
        result = processor.process_file(temp_audio_file)

    assert result is False
    assert mock_transcriber.transcribe.call_count == 1
    calls = mock_manifest.update_file_status.call_args_list
    statuses = [c[0][1] for c in calls]
    assert "error" in statuses


def test_error_with_retry_succeeds(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that a transient error is retried and succeeds."""
    mock_transcriber.transcribe.side_effect = [
        RuntimeError("transient error"),
        _make_result("success"),
    ]
    settings = _make_settings(no_retry=False, max_retries=3)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"), \
         patch("time.sleep"):
        result = processor.process_file(temp_audio_file)

    assert result is True
    assert mock_transcriber.transcribe.call_count == 2


def test_retry_exhaustion(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that retries are bounded and eventually fail."""
    mock_transcriber.transcribe.side_effect = RuntimeError("persistent error")
    settings = _make_settings(no_retry=False, max_retries=2)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), patch("time.sleep"):
        result = processor.process_file(temp_audio_file)

    assert result is False
    assert mock_transcriber.transcribe.call_count == 3


def test_word_timestamps_passed(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that word_timestamps setting is passed through to transcriber."""
    settings = _make_settings(word_timestamps=True)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        processor.process_file(temp_audio_file)

    call_kwargs = mock_transcriber.transcribe.call_args
    assert call_kwargs.kwargs.get("word_timestamps") is True


def test_temperature_fallback_passed(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that temperature fallback tuple is passed through."""
    settings = _make_settings(temperature="0.0,0.4,0.8")
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        processor.process_file(temp_audio_file)

    call_kwargs = mock_transcriber.transcribe.call_args
    assert call_kwargs.kwargs.get("temperature") == (0.0, 0.4, 0.8)


def test_beam_size_passed(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that beam_size setting is passed through."""
    settings = _make_settings(beam_size=10)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        processor.process_file(temp_audio_file)

    call_kwargs = mock_transcriber.transcribe.call_args
    assert call_kwargs.kwargs.get("beam_size") == 10


def test_output_format_markdown(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that markdown output format triggers _save_markdown."""
    settings = _make_settings(output_format="markdown")
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"), \
         patch.object(processor, "_save_markdown") as mock_md:
        processor.process_file(temp_audio_file)

    mock_md.assert_called_once()


def test_language_passed(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that language setting is passed through."""
    settings = _make_settings(language="en")
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        processor.process_file(temp_audio_file)

    call_kwargs = mock_transcriber.transcribe.call_args
    assert call_kwargs.kwargs.get("language") == "en"


def test_hallucination_threshold_passed(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that hallucination_silence_threshold is passed through."""
    settings = _make_settings(hallucination_silence_threshold=2.0)
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        processor.process_file(temp_audio_file)

    call_kwargs = mock_transcriber.transcribe.call_args
    assert call_kwargs.kwargs.get("hallucination_silence_threshold") == 2.0


def test_diarize_flag(temp_audio_file, mock_manifest, mock_transcriber):
    """Test that --diarize triggers diarization and enables word_timestamps."""
    mock_diarizer = MagicMock()
    mock_diarizer.diarize.return_value = {
        "segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
        "num_speakers": 1,
        "speakers": ["SPEAKER_00"],
    }
    mock_diarizer.assign_speakers.return_value = {
        "text": "hello world",
        "segments": [{"start": 0.0, "end": 5.0, "text": "hello world", "speaker": "SPEAKER_00"}],
        "diarization": {"num_speakers": 1, "speakers": ["SPEAKER_00"]},
    }

    settings = _make_settings(diarize=True, hf_token="test-token")
    processor = AudioProcessor(settings, mock_manifest)
    processor._transcriber = mock_transcriber
    processor._diarizer = mock_diarizer

    with patch("pathlib.Path.mkdir"), \
         patch("audioscript.processors.audio_processor._save_results"):
        result = processor.process_file(temp_audio_file)

    assert result is True
    # word_timestamps should be forced on for diarization
    call_kwargs = mock_transcriber.transcribe.call_args
    assert call_kwargs.kwargs.get("word_timestamps") is True
    # Diarizer should be called
    mock_diarizer.diarize.assert_called_once()
    mock_diarizer.assign_speakers.assert_called_once()
    mock_diarizer.save_rttm.assert_called_once()
