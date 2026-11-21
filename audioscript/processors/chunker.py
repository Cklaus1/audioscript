"""Audio chunking for long recordings.

Splits long transcripts into topic-coherent chunks for better
diarization accuracy and LLM analysis (avoids token limits).

Strategy: silence-based chunking with topic-aware boundaries.
1. Find natural break points (long pauses between segments)
2. Group consecutive segments into chunks of target duration
3. Prefer splitting at speaker changes or topic shifts
4. Each chunk gets independent LLM analysis
5. Results are merged back into a unified transcript
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TARGET_CHUNK_MINUTES = 30
DEFAULT_MIN_PAUSE_SECONDS = 3.0
DEFAULT_MAX_CHUNK_CHARS = 80_000  # Leave room under 100K LLM limit


@dataclass
class TranscriptChunk:
    """A chunk of a transcript with its segments and metadata."""

    chunk_id: int
    start_time: float
    end_time: float
    segments: list[dict[str, Any]]
    text: str
    char_count: int
    speaker_labels: list[str] = field(default_factory=list)


def find_split_points(
    segments: list[dict[str, Any]],
    min_pause: float = DEFAULT_MIN_PAUSE_SECONDS,
) -> list[float]:
    """Find natural break points in the transcript.

    Returns timestamps where pauses >= min_pause occur between segments.
    These are the best places to split without cutting mid-speech.
    """
    splits: list[float] = []

    for i in range(1, len(segments)):
        prev_end = segments[i - 1].get("end", 0)
        curr_start = segments[i].get("start", 0)
        gap = curr_start - prev_end

        if gap >= min_pause:
            # Split at the midpoint of the gap
            splits.append((prev_end + curr_start) / 2)

    return splits


def _is_wide(ch: str) -> bool:
    o = ord(ch)
    return (0x1100 <= o <= 0x11FF or 0x3040 <= o <= 0x30FF or 0x3130 <= o <= 0x318F
            or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF
            or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF)


def estimate_segment_tokens(seg: dict[str, Any]) -> int:
    """Conservative prompt-token estimate for one rendered line "[mm:ss] SPEAKER_NN: text".

    Korean/Japanese/Chinese characters cost about one token each; other text about one per
    three characters; the timestamp and speaker label about ten. A character cap alone let
    Korean recordings overflow a 16k-token local model.
    """
    text = seg.get("text", "") or ""
    wide = sum(1 for ch in text if _is_wide(ch))
    return wide + (len(text) - wide + 2) // 3 + 10


def chunk_transcript(
    segments: list[dict[str, Any]],
    target_minutes: float = DEFAULT_TARGET_CHUNK_MINUTES,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    min_pause: float = DEFAULT_MIN_PAUSE_SECONDS,
    size_fn: Callable[[dict[str, Any]], int] | None = None,
) -> list[TranscriptChunk]:
    """Split a transcript into chunks at natural break points.

    Args:
        segments: Transcript segments with start, end, text, speaker fields.
        target_minutes: Target chunk duration in minutes.
        max_chars: Maximum size per chunk, in characters, or in the units of size_fn.
        size_fn: Size of one segment (default: its character count). Pass
            estimate_segment_tokens to budget by prompt tokens.
        min_pause: Minimum pause to consider as a split point.

    Returns:
        List of TranscriptChunks. Short transcripts return a single chunk.
    """
    if not segments:
        return []

    total_duration = segments[-1].get("end", 0) - segments[0].get("start", 0)
    total_chars = sum(len(seg.get("text", "")) for seg in segments)
    size = size_fn or (lambda seg: len(seg.get("text", "")))
    total_size = sum(size(seg) for seg in segments)

    # No chunking needed for short transcripts
    if total_duration < target_minutes * 60 * 1.5 and total_size < max_chars:
        return [TranscriptChunk(
            chunk_id=0,
            start_time=segments[0].get("start", 0),
            end_time=segments[-1].get("end", 0),
            segments=segments,
            text=" ".join(seg.get("text", "").strip() for seg in segments),
            char_count=total_chars,
            speaker_labels=list({seg.get("speaker", "") for seg in segments if seg.get("speaker")}),
        )]

    # Find natural split points
    splits = find_split_points(segments, min_pause=min_pause)

    if not splits:
        # No natural pauses — fall back to fixed time splits
        target_secs = target_minutes * 60
        t = segments[0].get("start", 0) + target_secs
        while t < segments[-1].get("end", 0):
            splits.append(t)
            t += target_secs

    # Build chunks by grouping segments between split points
    target_secs = target_minutes * 60
    chunks: list[TranscriptChunk] = []
    current_segments: list[dict[str, Any]] = []
    current_chars = 0
    current_size = 0
    chunk_start = segments[0].get("start", 0)
    chunk_id = 0

    for seg in segments:
        seg_chars = len(seg.get("text", ""))
        seg_size = size(seg)
        seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2
        current_duration = seg.get("end", 0) - chunk_start

        # Check if we should split here
        should_split = False
        if current_duration >= target_secs:
            # Past target duration — split at next natural break
            for split_time in splits:
                if chunk_start < split_time <= seg.get("start", 0):
                    should_split = True
                    break
        if current_size + seg_size > max_chars:
            should_split = True

        if should_split and current_segments:
            # Emit current chunk
            chunks.append(TranscriptChunk(
                chunk_id=chunk_id,
                start_time=chunk_start,
                end_time=current_segments[-1].get("end", 0),
                segments=current_segments,
                text=" ".join(s.get("text", "").strip() for s in current_segments),
                char_count=current_chars,
                speaker_labels=list({s.get("speaker", "") for s in current_segments if s.get("speaker")}),
            ))
            chunk_id += 1
            current_segments = []
            current_chars = 0
            current_size = 0
            chunk_start = seg.get("start", 0)

        current_segments.append(seg)
        current_chars += seg_chars
        current_size += seg_size

    # Final chunk
    if current_segments:
        chunks.append(TranscriptChunk(
            chunk_id=chunk_id,
            start_time=chunk_start,
            end_time=current_segments[-1].get("end", 0),
            segments=current_segments,
            text=" ".join(s.get("text", "").strip() for s in current_segments),
            char_count=current_chars,
            speaker_labels=list({s.get("speaker", "") for s in current_segments if s.get("speaker")}),
        ))

    logger.info(
        "Chunked %d segments (%.0f min, %d chars) into %d chunks",
        len(segments), total_duration / 60, total_chars, len(chunks),
    )

    return chunks


def merge_chunk_analyses(
    chunks: list[TranscriptChunk],
    chunk_analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge LLM analyses from multiple chunks into a unified result.

    Combines titles, summaries, action items, topics, speakers,
    decisions, and questions from all chunks.
    """
    if not chunk_analyses:
        return {}

    if len(chunk_analyses) == 1:
        return chunk_analyses[0]

    # Use the first chunk's title as the overall title, or combine
    titles = [a.get("title", "") for a in chunk_analyses if a.get("title")]
    combined_title = titles[0] if titles else ""

    # Combine summaries
    summaries = [a.get("summary", "") for a in chunk_analyses if a.get("summary")]
    combined_summary = " ".join(summaries)

    # Use most common classification
    classifications = [a.get("classification", "") for a in chunk_analyses if a.get("classification")]
    if classifications:
        from collections import Counter
        combined_classification = Counter(classifications).most_common(1)[0][0]
    else:
        combined_classification = "other"

    # Merge lists (dedup where possible)
    all_actions = []
    for a in chunk_analyses:
        all_actions.extend(a.get("action_items", []))

    all_topics = []
    seen_topics = set()
    for a in chunk_analyses:
        for t in a.get("topics", []):
            if t.lower() not in seen_topics:
                seen_topics.add(t.lower())
                all_topics.append(t)

    all_decisions = []
    for a in chunk_analyses:
        all_decisions.extend(a.get("key_decisions", []))

    all_questions = []
    for a in chunk_analyses:
        all_questions.extend(a.get("questions_raised", []))

    # Merge speakers (dedup by name)
    all_speakers = []
    seen_names = set()
    for a in chunk_analyses:
        for s in a.get("speakers", []):
            name = s.get("likely_name", "")
            if name and name.lower() not in seen_names:
                seen_names.add(name.lower())
                all_speakers.append(s)

    # Per-chunk details for DeepScript (preserves topic timeline)
    chunk_details = []
    for chunk, analysis in zip(chunks, chunk_analyses):
        chunk_details.append({
            "chunk_id": chunk.chunk_id,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "duration_seconds": chunk.end_time - chunk.start_time,
            "speaker_labels": chunk.speaker_labels,
            "title": analysis.get("title"),
            "summary": analysis.get("summary"),
            "classification": analysis.get("classification"),
            "topics": analysis.get("topics", []),
            "action_items": analysis.get("action_items", []),
            "speakers_identified": [
                s.get("likely_name") for s in analysis.get("speakers", []) if s.get("likely_name")
            ],
        })

    return {
        "title": combined_title,
        "summary": combined_summary,
        "classification": combined_classification,
        "speakers": all_speakers,
        "action_items": all_actions,
        "topics": all_topics[:10],  # Cap at 10
        "key_decisions": all_decisions,
        "questions_raised": all_questions,
        "chunked": True,
        "chunk_count": len(chunk_analyses),
        "chunks": chunk_details,  # Per-chunk details for DeepScript topic timeline
    }
