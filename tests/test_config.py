"""Tests for the configuration module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from audioscript.config.settings import (
    AudioScriptConfig,
    TranscriptionTier,
    get_settings,
    load_yaml_config,
    merge_configs,
)


def test_load_yaml_config():
    """Test loading configuration from YAML file."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as temp_file:
        config = {
            "output_dir": "./test_output",
            "tier": "high_quality",
            "force": True,
        }
        yaml.dump(config, temp_file)
        temp_file.flush()

        loaded = load_yaml_config(Path(temp_file.name))

        assert loaded["output_dir"] == "./test_output"
        assert loaded["tier"] == "high_quality"
        assert loaded["force"] is True


def test_load_yaml_config_missing_file():
    """Test that a missing config file returns an empty dict."""
    loaded = load_yaml_config(Path("/nonexistent/config.yaml"))
    assert loaded == {}


def test_load_yaml_config_invalid_yaml():
    """Test that an invalid YAML file logs a warning and returns empty dict."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as temp_file:
        temp_file.write(": invalid: yaml: {{{}}}::::\n")
        temp_file.flush()

        loaded = load_yaml_config(Path(temp_file.name))
        assert loaded == {}


def test_merge_configs():
    """Test merging CLI arguments with file configuration."""
    file_config = {
        "output_dir": "./file_output",
        "tier": "draft",
        "force": False,
        "version": "1.0",
    }
    cli_args = {
        "output_dir": "./cli_output",
        "tier": "high_quality",
        "force": None,
        "model": "test-model",
    }

    merged = merge_configs(cli_args, file_config)

    assert merged["output_dir"] == "./cli_output"
    assert merged["tier"] == "high_quality"
    assert merged["force"] is False
    assert merged["version"] == "1.0"
    assert merged["model"] == "test-model"


def test_merge_configs_none_booleans_preserve_yaml():
    """Verify that None CLI booleans don't override YAML values."""
    file_config = {"force": True, "clean_audio": True, "summarize": True}
    cli_args = {"force": None, "clean_audio": None, "summarize": None}

    merged = merge_configs(cli_args, file_config)

    assert merged["force"] is True
    assert merged["clean_audio"] is True
    assert merged["summarize"] is True


def test_get_settings_returns_config_object():
    """Test that get_settings returns an AudioScriptConfig instance."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as temp_file:
        config = {
            "output_dir": "./test_output",
            "tier": "draft",
            "version": "2.0",
        }
        yaml.dump(config, temp_file)
        temp_file.flush()

        cli_args = {"input": "./test.mp3", "tier": "high_quality"}

        settings = get_settings(cli_args, Path(temp_file.name))

        assert isinstance(settings, AudioScriptConfig)
        assert settings.input == "./test.mp3"
        assert settings.output_dir == "./test_output"
        assert settings.tier == TranscriptionTier.HIGH_QUALITY
        assert settings.version == "2.0"


def test_audio_script_config_defaults():
    """Test the AudioScriptConfig Pydantic model defaults."""
    config = AudioScriptConfig()

    assert config.output_dir == "./output"
    assert config.tier == TranscriptionTier.DRAFT
    assert config.version == "1.0"
    assert config.clean_audio is False
    assert config.summarize is False
    assert config.force is False
    assert config.model is None
    assert config.no_retry is False
    assert config.max_retries == 3
    # New defaults
    assert config.language is None
    assert config.temperature == "0.0,0.2,0.4,0.6,0.8,1.0"
    assert config.word_timestamps is False
    assert config.hallucination_silence_threshold is None
    assert config.beam_size is None
    assert config.best_of is None
    assert config.clip_timestamps is None
    assert config.carry_initial_prompt is False
    assert config.condition_on_previous_text is True
    assert config.suppress_tokens == "-1"
    assert config.suppress_blank is True
    assert config.fp16 is True
    assert config.patience is None
    assert config.length_penalty is None
    assert config.output_format == "json"
    assert config.highlight_words is False
    assert config.max_line_width is None
    assert config.max_line_count is None
    assert config.max_words_per_line is None
    assert config.download_root is None


def test_audio_script_config_custom():
    """Test the AudioScriptConfig Pydantic model with custom values."""
    config = AudioScriptConfig(
        input="./test.mp3",
        output_dir="./custom_output",
        tier=TranscriptionTier.HIGH_QUALITY,
        version="2.0",
        clean_audio=True,
        summarize=True,
        force=True,
        model="custom-model",
        no_retry=True,
        max_retries=5,
        word_timestamps=True,
        beam_size=10,
    )

    assert config.input == "./test.mp3"
    assert config.output_dir == "./custom_output"
    assert config.tier == TranscriptionTier.HIGH_QUALITY
    assert config.version == "2.0"
    assert config.clean_audio is True
    assert config.summarize is True
    assert config.force is True
    assert config.model == "custom-model"
    assert config.no_retry is True
    assert config.max_retries == 5
    assert config.word_timestamps is True
    assert config.beam_size == 10


def test_balanced_tier():
    """Test the new balanced tier option."""
    config = AudioScriptConfig(tier=TranscriptionTier.BALANCED)
    assert config.tier == TranscriptionTier.BALANCED
    assert config.tier.value == "balanced"


def test_parse_temperature_single():
    """Test parsing a single temperature value."""
    config = AudioScriptConfig(temperature="0")
    assert config.parse_temperature() == 0.0


def test_parse_temperature_tuple():
    """Test parsing comma-separated temperature values."""
    config = AudioScriptConfig(temperature="0.0,0.2,0.4")
    assert config.parse_temperature() == (0.0, 0.2, 0.4)


def test_parse_temperature_default():
    """Test that the default temperature produces a fallback tuple."""
    config = AudioScriptConfig()
    result = config.parse_temperature()
    assert isinstance(result, tuple)
    assert len(result) == 6
    assert result[0] == 0.0


def test_parse_clip_timestamps_none():
    """Test parsing clip_timestamps when not set."""
    config = AudioScriptConfig()
    assert config.parse_clip_timestamps() == "0"


def test_parse_clip_timestamps_values():
    """Test parsing comma-separated clip timestamps."""
    config = AudioScriptConfig(clip_timestamps="30,60,90")
    result = config.parse_clip_timestamps()
    assert result == [30.0, 60.0, 90.0]


def test_output_format_validation():
    """Test that invalid output formats are rejected."""
    with pytest.raises(ValueError, match="Invalid output format"):
        AudioScriptConfig(output_format="mp4")


def test_output_format_valid():
    """Test that all valid output formats are accepted."""
    for fmt in ["json", "txt", "vtt", "srt", "tsv", "all"]:
        config = AudioScriptConfig(output_format=fmt)
        assert config.output_format == fmt


# --- C3: Range validation tests ---

def test_similarity_threshold_out_of_range():
    """Test that similarity threshold outside 0-1 is rejected."""
    with pytest.raises(ValueError):
        AudioScriptConfig(speaker_similarity_threshold=5.0)
    with pytest.raises(ValueError):
        AudioScriptConfig(speaker_similarity_threshold=-0.1)


def test_similarity_threshold_valid_range():
    assert AudioScriptConfig(speaker_similarity_threshold=0.0).speaker_similarity_threshold == 0.0
    assert AudioScriptConfig(speaker_similarity_threshold=1.0).speaker_similarity_threshold == 1.0
    assert AudioScriptConfig(speaker_similarity_threshold=0.85).speaker_similarity_threshold == 0.85


def test_min_speakers_gt_max_speakers():
    """Test that min_speakers > max_speakers is rejected."""
    with pytest.raises(ValueError, match="min_speakers.*max_speakers"):
        AudioScriptConfig(min_speakers=5, max_speakers=2)


def test_num_speakers_conflicts_with_range():
    """Test that num_speakers outside min/max range is rejected."""
    with pytest.raises(ValueError, match="num_speakers.*min_speakers"):
        AudioScriptConfig(num_speakers=1, min_speakers=3)
    with pytest.raises(ValueError, match="num_speakers.*max_speakers"):
        AudioScriptConfig(num_speakers=10, max_speakers=5)


def test_speaker_counts_valid():
    config = AudioScriptConfig(num_speakers=3, min_speakers=2, max_speakers=5)
    assert config.num_speakers == 3


def test_beam_size_positive():
    with pytest.raises(ValueError):
        AudioScriptConfig(beam_size=0)
    with pytest.raises(ValueError):
        AudioScriptConfig(beam_size=-1)
    assert AudioScriptConfig(beam_size=5).beam_size == 5


def test_best_of_positive():
    with pytest.raises(ValueError):
        AudioScriptConfig(best_of=0)
    assert AudioScriptConfig(best_of=3).best_of == 3


def test_batch_sizes_positive():
    with pytest.raises(ValueError):
        AudioScriptConfig(segmentation_batch_size=0)
    with pytest.raises(ValueError):
        AudioScriptConfig(embedding_batch_size=-1)


def test_diarize_requires_token():
    """Test that diarize=True without HF token is rejected at config level."""
    import os
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="HuggingFace token"):
            AudioScriptConfig(diarize=True, hf_token=None)


def test_diarize_with_env_token():
    """Test that diarize=True succeeds with HF_TOKEN env var."""
    import os
    with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
        config = AudioScriptConfig(diarize=True)
        assert config.diarize is True
