"""Tests for the path validation module."""

import pytest

from audioscript.utils.validate import (
    PathValidationError,
    validate_safe_input,
    validate_safe_output_dir,
    validate_safe_path,
    validate_safe_file_path,
)


# --- validate_safe_path ---

def test_safe_path_relative():
    """Test that relative paths are accepted."""
    p = validate_safe_path("audio/test.mp3")
    assert str(p) == "audio/test.mp3"


def test_safe_path_simple_file():
    p = validate_safe_path("test.mp3")
    assert str(p) == "test.mp3"


def test_safe_path_rejects_empty():
    with pytest.raises(PathValidationError, match="Empty"):
        validate_safe_path("")


def test_safe_path_rejects_whitespace_only():
    with pytest.raises(PathValidationError, match="Empty"):
        validate_safe_path("   ")


def test_safe_path_rejects_absolute():
    with pytest.raises(PathValidationError, match="Absolute"):
        validate_safe_path("/etc/passwd")


def test_safe_path_rejects_traversal():
    with pytest.raises(PathValidationError, match="traversal"):
        validate_safe_path("../../../etc/passwd")


def test_safe_path_rejects_traversal_mid():
    with pytest.raises(PathValidationError, match="traversal"):
        validate_safe_path("audio/../../../etc/passwd")


def test_safe_path_rejects_control_chars():
    with pytest.raises(PathValidationError, match="Control characters"):
        validate_safe_path("audio/\x00test.mp3")


def test_safe_path_rejects_null_byte():
    with pytest.raises(PathValidationError, match="Control characters"):
        validate_safe_path("test\x00.mp3")


def test_safe_path_rejects_tab_injection():
    with pytest.raises(PathValidationError, match="Control characters"):
        validate_safe_path("test\x01.mp3")


def test_safe_path_allows_spaces():
    """Spaces in filenames are OK."""
    p = validate_safe_path("my audio/test file.mp3")
    assert "my audio" in str(p)


def test_safe_path_allows_unicode():
    p = validate_safe_path("audio/tëst_日本語.mp3")
    assert "tëst" in str(p)


# --- validate_safe_output_dir ---

def test_safe_output_dir_rejects_glob():
    with pytest.raises(PathValidationError, match="Glob characters"):
        validate_safe_output_dir("output/*")


def test_safe_output_dir_rejects_question_mark():
    with pytest.raises(PathValidationError, match="Glob characters"):
        validate_safe_output_dir("output/dir?")


def test_safe_output_dir_accepts_plain():
    p = validate_safe_output_dir("output/transcripts")
    assert str(p) == "output/transcripts"


def test_safe_output_dir_rejects_traversal():
    with pytest.raises(PathValidationError, match="traversal"):
        validate_safe_output_dir("../../.ssh")


def test_safe_output_dir_rejects_absolute():
    with pytest.raises(PathValidationError, match="Absolute"):
        validate_safe_output_dir("/tmp/output")


# --- validate_safe_input ---

def test_safe_input_allows_glob():
    """Input paths may contain glob chars."""
    result = validate_safe_input("audio/*.mp3")
    assert result == "audio/*.mp3"


def test_safe_input_allows_recursive_glob():
    result = validate_safe_input("audio/**/*.mp3")
    assert result == "audio/**/*.mp3"


def test_safe_input_rejects_traversal():
    with pytest.raises(PathValidationError, match="traversal"):
        validate_safe_input("../../etc/*.mp3")


def test_safe_input_rejects_absolute():
    with pytest.raises(PathValidationError, match="Absolute"):
        validate_safe_input("/etc/audio/*.mp3")


def test_safe_input_rejects_control_chars():
    with pytest.raises(PathValidationError, match="Control characters"):
        validate_safe_input("audio/\x00*.mp3")


def test_safe_input_rejects_empty():
    with pytest.raises(PathValidationError, match="Empty"):
        validate_safe_input("")


# --- validate_safe_file_path ---

def test_safe_file_path_relative():
    p = validate_safe_file_path("data/speakers.json", label="speaker database")
    assert str(p) == "data/speakers.json"


def test_safe_file_path_rejects_traversal():
    with pytest.raises(PathValidationError, match="traversal"):
        validate_safe_file_path("../secret/data.json", label="speaker database")


# --- PathValidationError hint ---

def test_error_has_hint():
    try:
        validate_safe_path("/absolute/path")
    except PathValidationError as e:
        assert e.hint is not None
        assert "relative" in e.hint.lower()
