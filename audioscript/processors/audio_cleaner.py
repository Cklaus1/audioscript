"""Audio cleaning via spectral noise reduction."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CLEAN_PARAMS: dict[str, dict[str, Any]] = {
    "light": {"prop_decrease": 0.5, "stationary": True},
    "moderate": {"prop_decrease": 0.75, "stationary": True},
    "aggressive": {"prop_decrease": 1.0, "stationary": False},
}


def compute_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate signal-to-noise ratio in dB using RMS-based method.

    Splits audio into short frames, uses the top 10% of frame energies
    as signal estimate and bottom 10% as noise estimate.
    """
    audio = audio.astype(np.float32)  # Prevent int16 overflow in squaring
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop

    # Vectorized RMS energy per frame (100x faster than Python loop)
    num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    # Pad audio to ensure full frames
    padded = np.pad(audio, (0, max(0, num_frames * hop_length + frame_length - len(audio))))
    # Create strided view of frames
    indices = np.arange(num_frames)[:, None] * hop_length + np.arange(frame_length)
    frames = padded[indices]
    energies = np.sqrt(np.mean(frames ** 2, axis=1))

    if len(energies) < 10:
        return 0.0

    sorted_energies = np.sort(energies)
    n = len(sorted_energies)
    noise_rms = np.mean(sorted_energies[:max(1, n // 10)]) + 1e-10
    signal_rms = np.mean(sorted_energies[-(max(1, n // 10)):]) + 1e-10

    return float(20 * np.log10(signal_rms / noise_rms))


def clean_audio(
    audio_path: str | Path,
    output_path: str | Path,
    level: str = "moderate",
    snr_threshold: float = 30.0,
) -> tuple[Path, dict[str, Any]]:
    """Clean audio using noisereduce spectral gating.

    Args:
        audio_path: Input audio file path.
        output_path: Where to write cleaned audio.
        level: Cleaning aggressiveness: light, moderate, aggressive.
        snr_threshold: Skip cleaning if SNR is above this (dB).

    Returns:
        Tuple of (output_path, stats_dict) where stats_dict contains:
        - snr_before: float (dB)
        - snr_after: float | None (dB, None if skipped)
        - skipped: bool
        - level: str
    """
    import librosa
    import noisereduce as nr
    import soundfile as sf

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    snr_before = compute_snr(audio, sr)

    # Skip if audio is already clean enough
    if snr_before >= snr_threshold:
        logger.info("SNR %.1f dB >= threshold %.1f dB, skipping cleaning", snr_before, snr_threshold)
        shutil.copy2(audio_path, output_path)
        return output_path, {
            "snr_before": round(snr_before, 1),
            "snr_after": None,
            "skipped": True,
            "level": level,
        }

    # Apply noise reduction
    params = CLEAN_PARAMS.get(level, CLEAN_PARAMS["moderate"])
    cleaned = nr.reduce_noise(y=audio, sr=sr, **params)

    # Write cleaned audio
    sf.write(str(output_path), cleaned, sr)
    snr_after = compute_snr(cleaned, sr)

    logger.info(
        "Cleaned audio: SNR %.1f → %.1f dB (level=%s)",
        snr_before, snr_after, level,
    )

    return output_path, {
        "snr_before": round(snr_before, 1),
        "snr_after": round(snr_after, 1),
        "skipped": False,
        "level": level,
    }
