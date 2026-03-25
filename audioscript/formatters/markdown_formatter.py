"""Obsidian-compatible markdown output formatter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_duration(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def format_frontmatter(
    audio_path: Path,
    metadata: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> str:
    """Generate YAML frontmatter (Obsidian-compatible).

    Returns the frontmatter block including the --- delimiters.
    """
    result = result or {}
    metadata = metadata or {}
    audio_meta = metadata.get("audio", {})
    file_meta = metadata.get("file", {})
    format_tags = audio_meta.get("format_tags", {})

    title = f"Transcript: {audio_path.stem}"

    # Use recording creation_time from audio metadata, fall back to file mtime, then today
    date = datetime.now().strftime("%Y-%m-%d")
    if format_tags.get("creation_time"):
        try:
            date = format_tags["creation_time"][:10]  # "2025-06-05T04:31:43..." → "2025-06-05"
        except (IndexError, TypeError):
            pass
    elif file_meta.get("modified"):
        try:
            date = str(file_meta["modified"])[:10]
        except (IndexError, TypeError):
            pass

    lines = ["---"]
    lines.append(f"title: \"{title}\"")
    lines.append(f"date: {date}")
    lines.append("source: audioscript")

    # Duration
    duration = audio_meta.get("duration_seconds")
    if duration:
        lines.append(f"duration: {duration}")

    # Language
    language = result.get("language")
    if language:
        lines.append(f"language: {language}")

    # Backend
    backend = result.get("backend")
    if backend:
        lines.append(f"backend: {backend}")

    # Speakers (with cluster IDs and names)
    diar = result.get("diarization", {})
    resolved = diar.get("speakers_resolved", [])
    if resolved:
        speaker_labels = []
        for s in resolved:
            name = s.get("display_name") or s.get("speaker_cluster_id", "")
            cluster = s.get("speaker_cluster_id", "")
            if name != cluster:
                speaker_labels.append(f"{name} ({cluster})")
            else:
                speaker_labels.append(cluster)
        lines.append(f"speakers: [{', '.join(speaker_labels)}]")
        lines.append(f"speaker_count: {len(resolved)}")
    elif diar.get("speakers"):
        speakers_list = ", ".join(diar["speakers"])
        lines.append(f"speakers: [{speakers_list}]")

    # Tags
    tags = ["transcript", "audioscript"]
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("---")

    return "\n".join(lines)


def format_metadata_table(
    audio_path: Path,
    metadata: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> str:
    """Generate a metadata table in markdown."""
    result = result or {}
    metadata = metadata or {}
    audio_meta = metadata.get("audio", {})
    file_meta = metadata.get("file", {})

    rows: list[tuple[str, str]] = []

    # Duration
    duration = audio_meta.get("duration_seconds")
    if duration:
        rows.append(("Duration", _format_duration(duration)))

    # Language
    if result.get("language"):
        rows.append(("Language", result["language"]))

    # Backend
    if result.get("backend"):
        rows.append(("Backend", result["backend"]))

    # File info
    if file_meta.get("size_human"):
        rows.append(("File Size", file_meta["size_human"]))
    if audio_meta.get("codec"):
        rows.append(("Codec", audio_meta["codec"]))
    if audio_meta.get("sample_rate"):
        rows.append(("Sample Rate", f"{audio_meta['sample_rate']} Hz"))

    # Speakers
    diar = result.get("diarization", {})
    if diar.get("num_speakers"):
        rows.append(("Speakers", str(diar["num_speakers"])))

    if not rows:
        return ""

    lines = ["## Metadata", "", "| Property | Value |", "|----------|-------|"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")

    return "\n".join(lines)


def format_transcript_body(
    segments: list[dict[str, Any]],
    include_timestamps: bool = True,
    include_speakers: bool = True,
    min_confidence: float = 0.4,
) -> str:
    """Format transcript segments as markdown.

    Groups consecutive segments by speaker into sections.
    Low-confidence segments are prefixed with [?].
    """
    if not segments:
        return "*No transcript available.*"

    lines: list[str] = ["## Transcript", ""]
    current_speaker = None

    for seg in segments:
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Low confidence marker
        confidence = seg.get("confidence")
        prefix = ""
        if confidence is not None and confidence < min_confidence:
            prefix = "[?] "

        # Speaker change → new section
        if include_speakers and speaker and speaker != current_speaker:
            current_speaker = speaker
            ts = ""
            if include_timestamps and "start" in seg:
                ts = f" — {_format_timestamp(seg['start'])}"
            speaker_name = speaker if isinstance(speaker, str) else str(speaker)
            lines.append(f"### {speaker_name}{ts}")
            lines.append("")

        # Timestamp inline (when no speakers or same speaker)
        if include_timestamps and "start" in seg and (not include_speakers or not speaker):
            lines.append(f"**[{_format_timestamp(seg['start'])}]** {prefix}{text}")
        else:
            lines.append(f"{prefix}{text}")

        lines.append("")

    return "\n".join(lines)


def format_summary(summary: str) -> str:
    """Format a summary section."""
    return f"## Summary\n\n{summary}"


def render_markdown(
    result_dict: dict[str, Any],
    audio_path: Path,
    metadata: dict[str, Any] | None = None,
    summary: str | None = None,
) -> str:
    """Assemble a complete markdown document from transcription results.

    Produces an Obsidian-compatible markdown file with:
    - YAML frontmatter
    - Metadata table
    - Summary (if provided)
    - Speaker-labeled transcript
    """
    parts: list[str] = []

    # Frontmatter
    parts.append(format_frontmatter(audio_path, metadata, result_dict))

    # Title
    parts.append(f"# Transcript: {audio_path.stem}")

    # Metadata table
    meta_table = format_metadata_table(audio_path, metadata, result_dict)
    if meta_table:
        parts.append(meta_table)

    # Summary
    if summary:
        parts.append(format_summary(summary))

    # Transcript body
    segments = result_dict.get("segments", [])
    has_speakers = any(seg.get("speaker") for seg in segments)
    body = format_transcript_body(
        segments,
        include_speakers=has_speakers,
    )
    parts.append(body)

    return "\n\n".join(parts) + "\n"
