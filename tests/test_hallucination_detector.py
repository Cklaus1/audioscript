"""Tests for the hallucination detector module."""

import pytest

from audioscript.processors.backend_protocol import TranscriptionSegment
from audioscript.processors.hallucination_detector import (
    HallucinationReport,
    analyze,
    apply_filter,
    detect_repetition,
    score_confidence,
)


def _seg(text="Hello world", confidence=None, avg_logprob=None, **kw):
    """Helper to create a TranscriptionSegment."""
    return TranscriptionSegment(
        id=kw.get("id", 0),
        start=kw.get("start", 0.0),
        end=kw.get("end", 1.0),
        text=text,
        confidence=confidence,
        avg_logprob=avg_logprob,
    )


# --- score_confidence ---

def test_score_confidence_from_confidence_field():
    """Returns confidence directly when present on segment."""
    segments = [_seg(confidence=0.95)]
    scores = score_confidence(segments)
    assert scores == [0.95]


def test_score_confidence_from_avg_logprob():
    """Computes exp(avg_logprob) when confidence is None."""
    import math
    segments = [_seg(avg_logprob=-0.5)]
    scores = score_confidence(segments)
    assert scores[0] == pytest.approx(math.exp(-0.5), abs=1e-6)


def test_score_confidence_returns_none_when_unavailable():
    """Returns None when neither confidence nor avg_logprob is set."""
    segments = [_seg()]
    scores = score_confidence(segments)
    assert scores == [None]


# --- detect_repetition ---

def test_detect_repetition_flags_identical_consecutive():
    """Consecutive identical segments are flagged as repetition."""
    segments = [
        _seg(text="The quick brown fox jumps over the lazy dog"),
        _seg(text="The quick brown fox jumps over the lazy dog"),
    ]
    flags = detect_repetition(segments)
    assert flags[0] is False
    assert flags[1] is True


def test_detect_repetition_no_flag_first_segment():
    """First segment is never flagged."""
    segments = [_seg(text="Hello world this is a test")]
    flags = detect_repetition(segments)
    assert flags == [False]


def test_detect_repetition_handles_short_segments():
    """Short segments (fewer words than ngram_size) don't cause errors."""
    segments = [
        _seg(text="Hi"),
        _seg(text="Hi"),
    ]
    flags = detect_repetition(segments)
    assert len(flags) == 2
    # Short segments produce empty ngrams so no flag
    assert flags[1] is False


# --- analyze ---

def test_analyze_returns_reports():
    """analyze returns one HallucinationReport per segment."""
    segments = [_seg(confidence=0.9), _seg(confidence=0.8)]
    reports = analyze(segments)
    assert len(reports) == 2
    assert all(isinstance(r, HallucinationReport) for r in reports)


def test_analyze_risk_none_no_flags():
    """Segment with no flags gets risk='none'."""
    segments = [_seg(confidence=0.9)]
    reports = analyze(segments)
    assert reports[0].risk == "none"
    assert reports[0].flags == []


def test_analyze_risk_low_one_flag():
    """Segment with 1 flag gets risk='low'."""
    segments = [_seg(confidence=0.1)]  # low confidence
    reports = analyze(segments, min_confidence=0.4)
    assert reports[0].risk == "low"
    assert "low_confidence" in reports[0].flags


def test_analyze_risk_medium_two_flags():
    """Segment with 2 flags gets risk='medium'."""
    # low confidence + repetition
    text = "The quick brown fox jumps over the lazy dog"
    segments = [
        _seg(text=text, confidence=0.9),
        _seg(text=text, confidence=0.1),  # low confidence + repetition
    ]
    reports = analyze(segments, min_confidence=0.4)
    assert reports[1].risk == "medium"
    assert len(reports[1].flags) == 2


def test_analyze_risk_high_three_flags():
    """Segment with 3+ flags gets risk='high'."""
    text = "The quick brown fox jumps over the lazy dog"
    segments = [
        _seg(text=text, confidence=0.9),
        _seg(text=text, confidence=0.1),
    ]
    # Mock energy validation to add a third flag
    from unittest.mock import patch
    with patch(
        "audioscript.processors.hallucination_detector.validate_energy",
        return_value=[False, True],
    ):
        reports = analyze(segments, audio_path="/tmp/fake.mp3", min_confidence=0.4)
    assert reports[1].risk == "high"
    assert len(reports[1].flags) >= 3


# --- apply_filter ---

def test_apply_filter_auto_removes_high_risk():
    """Auto mode removes segments with high risk."""
    segments = [_seg(text="ok"), _seg(text="bad")]
    reports = [
        HallucinationReport(risk="none", flags=[]),
        HallucinationReport(risk="high", flags=["a", "b", "c"]),
    ]
    filtered = apply_filter(segments, reports, mode="auto")
    assert len(filtered) == 1
    assert filtered[0].text == "ok"


def test_apply_filter_flag_keeps_all():
    """Flag mode keeps all segments."""
    segments = [_seg(text="ok"), _seg(text="bad")]
    reports = [
        HallucinationReport(risk="none", flags=[]),
        HallucinationReport(risk="high", flags=["a", "b", "c"]),
    ]
    filtered = apply_filter(segments, reports, mode="flag")
    assert len(filtered) == 2


def test_apply_filter_off_is_noop():
    """Off mode returns segments unchanged."""
    segments = [_seg(text="ok"), _seg(text="bad")]
    reports = [
        HallucinationReport(risk="high", flags=["a", "b", "c"]),
        HallucinationReport(risk="high", flags=["a", "b", "c"]),
    ]
    filtered = apply_filter(segments, reports, mode="off")
    assert len(filtered) == 2
