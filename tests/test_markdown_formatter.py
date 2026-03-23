"""Tests for the markdown formatter module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from audioscript.formatters.markdown_formatter import (
    format_frontmatter,
    format_metadata_table,
    format_summary,
    format_transcript_body,
    render_markdown,
)


# --- format_frontmatter ---

def test_frontmatter_has_yaml_delimiters():
    """Frontmatter starts and ends with --- delimiters."""
    result = format_frontmatter(Path("interview.mp3"))
    lines = result.strip().splitlines()
    assert lines[0] == "---"
    assert lines[-1] == "---"


def test_frontmatter_contains_title_date_source_tags():
    """Frontmatter includes title, date, source, and tags."""
    result = format_frontmatter(Path("interview.mp3"))
    assert 'title: "Transcript: interview"' in result
    assert "date:" in result
    assert "source: audioscript" in result
    assert "tags: [transcript, audioscript]" in result


def test_frontmatter_includes_language_and_backend():
    """Frontmatter includes language and backend when present in result."""
    result_dict = {"language": "en", "backend": "faster-whisper"}
    output = format_frontmatter(Path("talk.mp3"), result=result_dict)
    assert "language: en" in output
    assert "backend: faster-whisper" in output


def test_frontmatter_omits_language_when_missing():
    """Frontmatter omits language/backend when not in result."""
    output = format_frontmatter(Path("talk.mp3"), result={})
    assert "language:" not in output
    assert "backend:" not in output


# --- format_metadata_table ---

def test_metadata_table_has_markdown_format():
    """Metadata table has correct markdown table headers and pipes."""
    metadata = {"audio": {"duration_seconds": 120.5}}
    result_dict = {"language": "en", "backend": "whisper"}
    output = format_metadata_table(Path("a.mp3"), metadata, result_dict)
    assert "## Metadata" in output
    assert "| Property | Value |" in output
    assert "|----------|-------|" in output
    assert "| Duration |" in output
    assert "| Language | en |" in output
    assert "| Backend | whisper |" in output


def test_metadata_table_empty_when_no_data():
    """Returns empty string when no metadata available."""
    output = format_metadata_table(Path("a.mp3"), {}, {})
    assert output == ""


# --- format_transcript_body ---

def test_transcript_body_groups_by_speaker():
    """Consecutive segments by same speaker are grouped under one heading."""
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello there."},
        {"speaker": "Alice", "start": 2.0, "end": 4.0, "text": "How are you?"},
        {"speaker": "Bob", "start": 4.0, "end": 6.0, "text": "I'm fine."},
    ]
    output = format_transcript_body(segments)
    assert output.count("### Alice") == 1
    assert output.count("### Bob") == 1


def test_transcript_body_includes_timestamps():
    """Speaker headings include timestamps."""
    segments = [
        {"speaker": "Alice", "start": 65.0, "end": 70.0, "text": "Hello."},
    ]
    output = format_transcript_body(segments, include_timestamps=True)
    assert "01:05" in output


def test_low_confidence_segments_get_prefix():
    """Low-confidence segments are prefixed with [?]."""
    segments = [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": "Maybe.", "confidence": 0.2},
    ]
    output = format_transcript_body(segments, min_confidence=0.4)
    assert "[?] Maybe." in output


def test_no_speaker_mode_uses_timestamp_format():
    """When include_speakers is False, timestamps appear inline."""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello."},
        {"start": 2.0, "end": 4.0, "text": "World."},
    ]
    output = format_transcript_body(
        segments, include_timestamps=True, include_speakers=False,
    )
    assert "**[00:00]** Hello." in output
    assert "**[00:02]** World." in output
    assert "###" not in output


def test_empty_segments_returns_no_transcript():
    """Empty segments list returns a placeholder message."""
    output = format_transcript_body([])
    assert output == "*No transcript available.*"


# --- format_summary ---

def test_format_summary():
    """Summary section has heading and text."""
    output = format_summary("This is a summary.")
    assert "## Summary" in output
    assert "This is a summary." in output


# --- render_markdown ---

def test_render_markdown_assembles_all_sections():
    """render_markdown produces frontmatter, title, metadata, and transcript."""
    result_dict = {
        "language": "en",
        "backend": "whisper",
        "segments": [
            {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello."},
        ],
    }
    metadata = {"audio": {"duration_seconds": 60}}
    output = render_markdown(result_dict, Path("test.mp3"), metadata)
    assert "---" in output
    assert "# Transcript: test" in output
    assert "## Metadata" in output
    assert "## Transcript" in output
    assert "## Summary" not in output


def test_render_markdown_with_summary():
    """render_markdown includes summary section when provided."""
    result_dict = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello."},
        ],
    }
    output = render_markdown(result_dict, Path("test.mp3"), summary="A brief summary.")
    assert "## Summary" in output
    assert "A brief summary." in output
