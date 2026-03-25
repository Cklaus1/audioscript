"""Typed data structures for processor outputs.

Replaces untyped Dict[str, Any] with TypedDicts for type safety
and IDE autocomplete across the codebase.
"""

from __future__ import annotations

from typing import Any, TypedDict


class DiarizationSegment(TypedDict, total=False):
    start: float
    end: float
    speaker: str


class DiarizationOverlap(TypedDict, total=False):
    overlap_duration: float
    total_speech_duration: float
    total_audio_duration: float
    overlap_percentage: float


class DiarizationResult(TypedDict, total=False):
    """Result from SpeakerDiarizer.diarize()."""

    segments: list[DiarizationSegment]
    num_speakers: int
    speakers: list[str]
    speaker_embeddings: dict[str, Any]  # {SPEAKER_00: ndarray}
    overlap: DiarizationOverlap


class SpeakerResolvedInfo(TypedDict, total=False):
    """Per-speaker resolution info in transcript output."""

    local_label: str
    speaker_cluster_id: str
    display_name: str | None
    status: str
    confidence: float
    source: str
    is_new: bool


class TranscriptDiarization(TypedDict, total=False):
    """Diarization metadata in transcript output dict."""

    num_speakers: int
    speakers: list[str]
    speakers_resolved: list[SpeakerResolvedInfo]
    identified: dict[str, str]
    evaluation: dict[str, float]
    overlap: DiarizationOverlap


class LLMAnalysis(TypedDict, total=False):
    """Structured output from LLM transcript analysis."""

    title: str
    summary: str
    classification: str
    speakers: list[dict[str, Any]]
    action_items: list[dict[str, Any]]
    topics: list[str]
    key_decisions: list[str]
    questions_raised: list[str]
