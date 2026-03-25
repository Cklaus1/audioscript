"""Tests for the WSL detection and path translation module."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from audioscript.sync.wsl import (
    is_windows_path,
    is_wsl,
    resolve_sync_path,
    translate_path,
)


@pytest.fixture(autouse=True)
def clear_wsl_cache():
    """Clear the is_wsl functools.cache between every test."""
    is_wsl.cache_clear()
    yield
    is_wsl.cache_clear()


# --- is_wsl ---


def test_is_wsl_true_when_env_set():
    """is_wsl returns True when WSL_DISTRO_NAME is set."""
    with patch.dict("os.environ", {"WSL_DISTRO_NAME": "Ubuntu"}):
        assert is_wsl() is True


def test_is_wsl_true_when_proc_version_contains_microsoft():
    """is_wsl returns True when /proc/version contains 'microsoft'."""
    with patch.dict("os.environ", {}, clear=True):
        with patch(
            "builtins.open",
            mock_open(read_data="Linux version 5.10.0 microsoft-standard-WSL2"),
        ):
            assert is_wsl() is True


def test_is_wsl_false_when_neither_condition():
    """is_wsl returns False when neither env var nor /proc/version match."""
    with patch.dict("os.environ", {}, clear=True):
        with patch(
            "builtins.open",
            mock_open(read_data="Linux version 5.15.0-generic"),
        ):
            assert is_wsl() is False


# --- is_windows_path ---


def test_is_windows_path_detects_backslash():
    """is_windows_path returns True for C:\\ style paths."""
    assert is_windows_path(r"C:\Users\test") is True


def test_is_windows_path_detects_forward_slash():
    """is_windows_path returns True for D:/ style paths."""
    assert is_windows_path("D:/Documents/audio") is True


def test_is_windows_path_rejects_unix():
    """is_windows_path returns False for Unix paths."""
    assert is_windows_path("/home/user") is False


# --- translate_path ---


def test_translate_path_returns_unix_as_is():
    """translate_path returns Unix paths unchanged."""
    assert translate_path("/home/user/audio") == "/home/user/audio"


def test_translate_path_uses_wslpath(clear_wsl_cache):
    """translate_path uses wslpath subprocess when available."""
    with patch("audioscript.sync.wsl.is_wsl", return_value=True):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/mnt/c/Users/test\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = translate_path(r"C:\Users\test")
            assert result == "/mnt/c/Users/test"
            mock_run.assert_called_once()


def test_translate_path_falls_back_to_regex(clear_wsl_cache):
    """translate_path uses regex fallback when wslpath is missing."""
    with patch("audioscript.sync.wsl.is_wsl", return_value=True):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = translate_path(r"C:\Users\test\audio")
            assert result == "/mnt/c/Users/test/audio"


def test_translate_path_handles_spaces(clear_wsl_cache):
    """translate_path preserves spaces in paths via regex fallback."""
    with patch("audioscript.sync.wsl.is_wsl", return_value=True):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = translate_path(r"C:\Users\My User\Sound Recordings")
            assert result == "/mnt/c/Users/My User/Sound Recordings"


# --- resolve_sync_path ---


def test_resolve_sync_path_raises_for_missing_dir():
    """resolve_sync_path raises FileNotFoundError for missing dirs."""
    with patch("audioscript.sync.wsl.translate_path", return_value="/tmp/nonexistent"):
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Source directory not found"):
                resolve_sync_path("/tmp/nonexistent")


def test_resolve_sync_path_returns_path_for_existing_dir():
    """resolve_sync_path returns Path for existing directories."""
    with patch("audioscript.sync.wsl.translate_path", return_value="/tmp/exists"):
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_dir", return_value=True):
                result = resolve_sync_path("/tmp/exists")
                assert isinstance(result, Path)
                assert str(result) == "/tmp/exists"
