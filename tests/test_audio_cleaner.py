"""Tests for the audio cleaner module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audioscript.processors.audio_cleaner import (
    CLEAN_PARAMS,
    compute_snr,
)


# ---------------------------------------------------------------------------
# compute_snr tests
# ---------------------------------------------------------------------------


def test_compute_snr_with_pure_silence():
    """SNR of silence should be 0 (too few distinct energy levels)."""
    audio = np.zeros(8000)  # 0.5s at 16kHz — only 10 frames, returns 0.0
    assert compute_snr(audio, 16000) == 0.0


def test_compute_snr_with_short_audio():
    """Very short audio with fewer than 10 frames returns 0."""
    audio = np.zeros(100)
    assert compute_snr(audio, 16000) == 0.0


def test_compute_snr_positive_for_noisy_signal():
    """A signal with loud and quiet parts should have positive SNR."""
    sr = 16000
    t_loud = np.linspace(0, 0.5, sr // 2, endpoint=False)
    loud = 0.9 * np.sin(2 * np.pi * 440 * t_loud)
    quiet = 0.001 * np.random.default_rng(42).standard_normal(sr // 2)
    audio = np.concatenate([loud, quiet])
    snr = compute_snr(audio, sr)
    assert snr > 10.0, f"Expected SNR > 10 dB, got {snr}"


def test_compute_snr_returns_float():
    """compute_snr should always return a float."""
    sr = 16000
    audio = np.random.default_rng(0).standard_normal(sr)
    result = compute_snr(audio, sr)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Helpers for mocking heavy audio dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dirs():
    """Provide temporary input and output directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.wav"
        output_path = Path(tmpdir) / "out" / "cleaned.wav"
        input_path.write_bytes(b"RIFF" + b"\x00" * 100)
        yield input_path, output_path


@pytest.fixture
def mock_audio_libs():
    """Mock librosa, noisereduce, and soundfile in sys.modules.

    Yields a dict with keys 'librosa', 'nr', 'sf' containing the mocks.
    """
    mock_librosa = MagicMock()
    mock_nr = MagicMock()
    mock_sf = MagicMock()

    with patch.dict(sys.modules, {
        "librosa": mock_librosa,
        "noisereduce": mock_nr,
        "soundfile": mock_sf,
    }):
        yield {"librosa": mock_librosa, "nr": mock_nr, "sf": mock_sf}


def _make_mock_audio():
    """Return a short numpy array for use as mock audio data."""
    return np.random.default_rng(7).standard_normal(16000).astype(np.float32)


def _import_clean_audio():
    """Import clean_audio fresh so it picks up mocked sys.modules."""
    from audioscript.processors.audio_cleaner import clean_audio
    return clean_audio


# ---------------------------------------------------------------------------
# clean_audio tests
# ---------------------------------------------------------------------------


@patch("audioscript.processors.audio_cleaner.shutil.copy2")
@patch("audioscript.processors.audio_cleaner.compute_snr")
def test_clean_audio_skips_when_snr_above_threshold(mock_snr, mock_copy, temp_dirs, mock_audio_libs):
    """When SNR >= threshold, file is copied and skipped=True."""
    input_path, output_path = temp_dirs
    mock_snr.return_value = 35.0
    mock_audio = _make_mock_audio()
    mock_audio_libs["librosa"].load.return_value = (mock_audio, 16000)

    clean_audio = _import_clean_audio()
    result_path, stats = clean_audio(input_path, output_path, snr_threshold=30.0)

    mock_copy.assert_called_once_with(input_path, output_path)
    assert stats["skipped"] is True
    assert stats["snr_after"] is None
    assert stats["snr_before"] == 35.0
    assert stats["level"] == "moderate"


@patch("audioscript.processors.audio_cleaner.compute_snr")
def test_clean_audio_calls_noisereduce_moderate(mock_snr, temp_dirs, mock_audio_libs):
    """Default moderate level passes correct params to nr.reduce_noise."""
    input_path, output_path = temp_dirs
    mock_snr.side_effect = [15.0, 25.0]
    mock_audio = _make_mock_audio()
    cleaned_audio = _make_mock_audio()
    mock_audio_libs["librosa"].load.return_value = (mock_audio, 16000)
    mock_audio_libs["nr"].reduce_noise.return_value = cleaned_audio

    clean_audio = _import_clean_audio()
    result_path, stats = clean_audio(input_path, output_path, level="moderate")

    mock_audio_libs["nr"].reduce_noise.assert_called_once_with(
        y=mock_audio, sr=16000, prop_decrease=0.75, stationary=True
    )
    mock_audio_libs["sf"].write.assert_called_once_with(str(output_path), cleaned_audio, 16000)
    assert stats["skipped"] is False
    assert stats["snr_before"] == 15.0
    assert stats["snr_after"] == 25.0
    assert stats["level"] == "moderate"


@patch("audioscript.processors.audio_cleaner.compute_snr")
def test_clean_audio_calls_noisereduce_light(mock_snr, temp_dirs, mock_audio_libs):
    """Light level passes prop_decrease=0.5 and stationary=True."""
    input_path, output_path = temp_dirs
    mock_snr.side_effect = [10.0, 20.0]
    mock_audio = _make_mock_audio()
    mock_audio_libs["librosa"].load.return_value = (mock_audio, 16000)
    mock_audio_libs["nr"].reduce_noise.return_value = mock_audio

    clean_audio = _import_clean_audio()
    clean_audio(input_path, output_path, level="light")

    mock_audio_libs["nr"].reduce_noise.assert_called_once_with(
        y=mock_audio, sr=16000, prop_decrease=0.5, stationary=True
    )


@patch("audioscript.processors.audio_cleaner.compute_snr")
def test_clean_audio_calls_noisereduce_aggressive(mock_snr, temp_dirs, mock_audio_libs):
    """Aggressive level passes prop_decrease=1.0 and stationary=False."""
    input_path, output_path = temp_dirs
    mock_snr.side_effect = [5.0, 18.0]
    mock_audio = _make_mock_audio()
    mock_audio_libs["librosa"].load.return_value = (mock_audio, 16000)
    mock_audio_libs["nr"].reduce_noise.return_value = mock_audio

    clean_audio = _import_clean_audio()
    clean_audio(input_path, output_path, level="aggressive")

    mock_audio_libs["nr"].reduce_noise.assert_called_once_with(
        y=mock_audio, sr=16000, prop_decrease=1.0, stationary=False
    )


@patch("audioscript.processors.audio_cleaner.compute_snr")
def test_clean_audio_unknown_level_falls_back_to_moderate(mock_snr, temp_dirs, mock_audio_libs):
    """An unknown level should fall back to moderate params."""
    input_path, output_path = temp_dirs
    mock_snr.side_effect = [10.0, 20.0]
    mock_audio = _make_mock_audio()
    mock_audio_libs["librosa"].load.return_value = (mock_audio, 16000)
    mock_audio_libs["nr"].reduce_noise.return_value = mock_audio

    clean_audio = _import_clean_audio()
    _, stats = clean_audio(input_path, output_path, level="unknown_level")

    mock_audio_libs["nr"].reduce_noise.assert_called_once_with(
        y=mock_audio, sr=16000, prop_decrease=0.75, stationary=True
    )
    assert stats["level"] == "unknown_level"


@patch("audioscript.processors.audio_cleaner.compute_snr")
def test_clean_audio_stats_dict_keys_and_types(mock_snr, temp_dirs, mock_audio_libs):
    """Stats dict should have all required keys with correct types."""
    input_path, output_path = temp_dirs
    mock_snr.side_effect = [12.0, 22.0]
    mock_audio = _make_mock_audio()
    mock_audio_libs["librosa"].load.return_value = (mock_audio, 16000)
    mock_audio_libs["nr"].reduce_noise.return_value = mock_audio

    clean_audio = _import_clean_audio()
    result_path, stats = clean_audio(input_path, output_path)

    assert isinstance(result_path, Path)
    assert set(stats.keys()) == {"snr_before", "snr_after", "skipped", "level"}
    assert isinstance(stats["snr_before"], float)
    assert isinstance(stats["snr_after"], float)
    assert isinstance(stats["skipped"], bool)
    assert isinstance(stats["level"], str)


@patch("audioscript.processors.audio_cleaner.shutil.copy2")
@patch("audioscript.processors.audio_cleaner.compute_snr")
def test_clean_audio_skip_stats_dict_keys_and_types(mock_snr, mock_copy, temp_dirs, mock_audio_libs):
    """Stats dict for skipped cleaning should have snr_after=None."""
    input_path, output_path = temp_dirs
    mock_snr.return_value = 40.0
    mock_audio = _make_mock_audio()
    mock_audio_libs["librosa"].load.return_value = (mock_audio, 16000)

    clean_audio = _import_clean_audio()
    _, stats = clean_audio(input_path, output_path, snr_threshold=30.0)

    assert stats["snr_after"] is None
    assert stats["skipped"] is True


@patch("audioscript.processors.audio_cleaner.compute_snr")
def test_clean_audio_creates_output_directory(mock_snr, temp_dirs, mock_audio_libs):
    """Output parent directory is created if it doesn't exist."""
    input_path, output_path = temp_dirs
    nested_output = output_path.parent / "deep" / "nested" / "cleaned.wav"
    mock_snr.side_effect = [10.0, 20.0]
    mock_audio = _make_mock_audio()
    mock_audio_libs["librosa"].load.return_value = (mock_audio, 16000)
    mock_audio_libs["nr"].reduce_noise.return_value = mock_audio

    clean_audio = _import_clean_audio()
    result_path, _ = clean_audio(input_path, nested_output)

    assert result_path == nested_output
    assert nested_output.parent.exists()


# ---------------------------------------------------------------------------
# CLEAN_PARAMS mapping tests
# ---------------------------------------------------------------------------


def test_clean_params_light():
    """Light params should have prop_decrease=0.5, stationary=True."""
    assert CLEAN_PARAMS["light"] == {"prop_decrease": 0.5, "stationary": True}


def test_clean_params_moderate():
    """Moderate params should have prop_decrease=0.75, stationary=True."""
    assert CLEAN_PARAMS["moderate"] == {"prop_decrease": 0.75, "stationary": True}


def test_clean_params_aggressive():
    """Aggressive params should have prop_decrease=1.0, stationary=False."""
    assert CLEAN_PARAMS["aggressive"] == {"prop_decrease": 1.0, "stationary": False}


def test_clean_params_has_all_levels():
    """CLEAN_PARAMS should contain exactly three levels."""
    assert set(CLEAN_PARAMS.keys()) == {"light", "moderate", "aggressive"}
