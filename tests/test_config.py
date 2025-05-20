"""Tests for the configuration module."""

import os
import tempfile
from pathlib import Path

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
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as temp_file:
        config = {
            "output_dir": "./test_output",
            "tier": "high_quality",
            "force": True,
        }
        yaml.dump(config, temp_file)
        temp_file.flush()
        
        # Load the config
        loaded_config = load_yaml_config(Path(temp_file.name))
        
        # Check that the config was loaded correctly
        assert loaded_config["output_dir"] == "./test_output"
        assert loaded_config["tier"] == "high_quality"
        assert loaded_config["force"] is True


def test_merge_configs():
    """Test merging CLI arguments with file configuration."""
    # File config
    file_config = {
        "output_dir": "./file_output",
        "tier": "draft",
        "force": False,
        "version": "1.0",
    }
    
    # CLI args (should override file config)
    cli_args = {
        "output_dir": "./cli_output",
        "tier": "high_quality",
        "force": None,  # None values should not override
        "model": "test-model",  # New values should be added
    }
    
    # Merge configs
    merged_config = merge_configs(cli_args, file_config)
    
    # Check that CLI args override file config
    assert merged_config["output_dir"] == "./cli_output"
    assert merged_config["tier"] == "high_quality"
    assert merged_config["force"] is False  # Not overridden by None
    assert merged_config["version"] == "1.0"  # Preserved from file
    assert merged_config["model"] == "test-model"  # Added from CLI


def test_get_settings():
    """Test getting settings by merging CLI args and config file."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as temp_file:
        config = {
            "output_dir": "./test_output",
            "tier": "draft",
            "version": "2.0",
        }
        yaml.dump(config, temp_file)
        temp_file.flush()
        
        # CLI args
        cli_args = {
            "input": "./test.mp3",
            "tier": "high_quality",
        }
        
        # Get settings
        settings = get_settings(cli_args, Path(temp_file.name))
        
        # Check merged settings
        assert settings["input"] == "./test.mp3"  # From CLI
        assert settings["output_dir"] == "./test_output"  # From file
        assert settings["tier"] == "high_quality"  # CLI overrides file
        assert settings["version"] == "2.0"  # From file


def test_audio_script_config_model():
    """Test the AudioScriptConfig Pydantic model."""
    # Create a config with default values
    config = AudioScriptConfig()
    
    # Check default values
    assert config.output_dir == "./output"
    assert config.tier == TranscriptionTier.DRAFT
    assert config.version == "1.0"
    assert config.clean_audio is False
    assert config.summarize is False
    assert config.force is False
    assert config.model is None
    assert config.no_retry is False
    
    # Create a config with custom values
    custom_config = AudioScriptConfig(
        input="./test.mp3",
        output_dir="./custom_output",
        tier=TranscriptionTier.HIGH_QUALITY,
        version="2.0",
        clean_audio=True,
        summarize=True,
        force=True,
        model="custom-model",
        no_retry=True,
    )
    
    # Check custom values
    assert custom_config.input == "./test.mp3"
    assert custom_config.output_dir == "./custom_output"
    assert custom_config.tier == TranscriptionTier.HIGH_QUALITY
    assert custom_config.version == "2.0"
    assert custom_config.clean_audio is True
    assert custom_config.summarize is True
    assert custom_config.force is True
    assert custom_config.model == "custom-model"
    assert custom_config.no_retry is True