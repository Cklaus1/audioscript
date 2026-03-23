"""Multi-layer hallucination detection for transcription segments."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from audioscript.processors.backend_protocol import TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class HallucinationReport:
    """Per-segment hallucination analysis result."""

    risk: str  # "none" | "low" | "medium" | "high"
    flags: list[str] = field(default_factory=list)
    confidence: float | None = None


def score_confidence(segments: list[TranscriptionSegment]) -> list[float | None]:
    """Layer 2: Extract confidence scores from segments.

    Uses exp(avg_logprob) when available (faster-whisper), None otherwise.
    """
    import math

    scores: list[float | None] = []
    for seg in segments:
        if seg.confidence is not None:
            scores.append(seg.confidence)
        elif seg.avg_logprob is not None:
            scores.append(min(1.0, max(0.0, math.exp(seg.avg_logprob))))
        else:
            scores.append(None)
    return scores


def detect_repetition(
    segments: list[TranscriptionSegment],
    overlap_threshold: float = 0.6,
    ngram_size: int = 3,
) -> list[bool]:
    """Layer 3: Detect repetitive segments via n-gram overlap.

    Compares n-grams between consecutive segments. Flags segments
    where overlap exceeds the threshold.
    """
    if not segments:
        return []

    def _ngrams(text: str, n: int) -> Counter[tuple[str, ...]]:
        words = re.findall(r'\w+', text.lower())
        if len(words) < n:
            return Counter()
        return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))

    flags: list[bool] = [False]  # First segment can't be a repeat

    for i in range(1, len(segments)):
        prev_ngrams = _ngrams(segments[i - 1].text, ngram_size)
        curr_ngrams = _ngrams(segments[i].text, ngram_size)

        if not prev_ngrams or not curr_ngrams:
            flags.append(False)
            continue

        # Compute overlap as Jaccard-like similarity
        common = sum((prev_ngrams & curr_ngrams).values())
        total = sum(curr_ngrams.values())

        overlap = common / total if total > 0 else 0.0
        flags.append(overlap >= overlap_threshold)

    return flags


def validate_energy(
    audio_path: str,
    segments: list[TranscriptionSegment],
    energy_threshold: float = 0.01,
) -> list[bool]:
    """Layer 4: Cross-check text against audio energy per segment.

    Returns per-segment flag: True if suspicious (text from near-silent audio).
    """
    try:
        import librosa
        import numpy as np

        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(audio) / sr

        flags: list[bool] = []
        for seg in segments:
            start_sample = int(seg.start * sr)
            end_sample = min(int(seg.end * sr), len(audio))

            if start_sample >= end_sample:
                flags.append(True)  # Invalid range
                continue

            chunk = audio[start_sample:end_sample]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            flags.append(rms < energy_threshold)

        return flags
    except ImportError:
        logger.debug("librosa not available for energy validation")
        return [False] * len(segments)
    except Exception as e:
        logger.warning("Energy validation failed: %s", e)
        return [False] * len(segments)


def analyze(
    segments: list[TranscriptionSegment],
    audio_path: str | None = None,
    min_confidence: float = 0.4,
) -> list[HallucinationReport]:
    """Multi-layer hallucination analysis.

    Combines confidence scoring, repetition detection, and energy validation
    into a per-segment HallucinationReport.
    """
    if not segments:
        return []

    # Layer 2: Confidence
    confidences = score_confidence(segments)

    # Layer 3: Repetition
    repetitions = detect_repetition(segments)

    # Layer 4: Energy (only if audio path provided)
    energies = (
        validate_energy(audio_path, segments) if audio_path else [False] * len(segments)
    )

    reports: list[HallucinationReport] = []

    for i, seg in enumerate(segments):
        flags: list[str] = []
        conf = confidences[i]

        # Check confidence
        if conf is not None and conf < min_confidence:
            flags.append("low_confidence")

        # Check repetition
        if repetitions[i]:
            flags.append("repetition")

        # Check energy
        if energies[i]:
            flags.append("low_energy")

        # Determine risk level
        if len(flags) >= 3:
            risk = "high"
        elif len(flags) == 2:
            risk = "medium"
        elif len(flags) == 1:
            risk = "low"
        else:
            risk = "none"

        reports.append(HallucinationReport(
            risk=risk,
            flags=flags,
            confidence=conf,
        ))

    return reports


def apply_filter(
    segments: list[TranscriptionSegment],
    reports: list[HallucinationReport],
    mode: str = "auto",
) -> list[TranscriptionSegment]:
    """Apply hallucination filtering to segments.

    Modes:
    - "auto": Remove segments with risk="high"
    - "flag": Keep all segments, add hallucination_risk info
    - "off": No-op
    """
    if mode == "off" or not reports:
        return segments

    filtered: list[TranscriptionSegment] = []
    for seg, report in zip(segments, reports):
        if mode == "auto" and report.risk == "high":
            logger.info(
                "Filtered hallucinated segment [%.1f-%.1f]: %s (flags: %s)",
                seg.start, seg.end, seg.text[:50], report.flags,
            )
            continue
        filtered.append(seg)

    return filtered
