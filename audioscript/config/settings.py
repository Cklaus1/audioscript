"""Settings management for AudioScript."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field


class TranscriptionTier(str, Enum):
    """Transcription quality tier options."""

    DRAFT = "draft"
    HIGH_QUALITY = "high_quality"


class AudioScriptConfig(BaseModel):
    """AudioScript configuration model."""

    input: Optional[str] = None
    output_dir: str = Field(default="./output")
    tier: TranscriptionTier = Field(default=TranscriptionTier.DRAFT)
    version: str = Field(default="1.0")
    clean_audio: bool = Field(default=False)
    summarize: bool = Field(default=False)
    force: bool = Field(default=False)
    model: Optional[str] = None
    no_retry: bool = Field(default=False)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dict containing the configuration
    """
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception:
        # If there's an error reading the config, return empty dict
        return {}


def merge_configs(cli_args: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI arguments with file configuration.

    CLI arguments take precedence over file configuration.

    Args:
        cli_args: Dictionary of CLI arguments
        file_config: Dictionary of configuration from YAML file

    Returns:
        Dict containing the merged configuration
    """
    # Start with file config as the base
    merged_config = {**file_config}

    # Override with CLI args (but only for non-None values)
    for key, value in cli_args.items():
        if value is not None:
            merged_config[key] = value

    return merged_config


def get_settings(
    cli_args: Dict[str, Any], config_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Get application settings by merging CLI args and config file.

    Args:
        cli_args: Dictionary of CLI arguments
        config_path: Path to config file (defaults to .audioscript.yaml in current dir)

    Returns:
        Dict containing the merged settings
    """
    # Default config path if not provided
    if config_path is None:
        config_path = Path(".audioscript.yaml")

    # Load config from file
    file_config = load_yaml_config(config_path)

    # Merge configs (CLI args override file config)
    merged_config = merge_configs(cli_args, file_config)

    # Validate using Pydantic model
    config_model = AudioScriptConfig(**merged_config)

    # Return as dict
    return config_model.model_dump()