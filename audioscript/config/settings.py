"""Settings management for AudioScript."""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS = {"json", "txt", "vtt", "srt", "tsv", "all", "markdown"}
VALID_BACKENDS = {"faster-whisper"}
VALID_CLEAN_LEVELS = {"light", "moderate", "aggressive"}
VALID_HALLUCINATION_FILTERS = {"auto", "flag", "off"}
VALID_RETRY_STRATEGIES = {"smart", "always", "never"}
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_VAD_MODEL = "pyannote/segmentation-3.0"
DEFAULT_TEMPERATURE = "0.0,0.2,0.4,0.6,0.8,1.0"


class TranscriptionTier(str, Enum):
    """Transcription quality tier options."""

    DRAFT = "draft"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


class AudioScriptConfig(BaseModel):
    """AudioScript configuration model."""

    # Core options
    input: str | None = None
    output_dir: str = Field(default="./output")
    tier: TranscriptionTier = Field(default=TranscriptionTier.DRAFT)
    version: str = Field(default="1.0")
    clean_audio: bool = Field(default=False)
    summarize: bool = Field(default=False)
    force: bool = Field(default=False)
    model: str | None = None
    no_retry: bool = Field(default=False)
    max_retries: int = Field(default=3, ge=0)

    # Speaker diarization options
    diarize: bool = Field(default=False)
    hf_token: str | None = Field(default=None)
    diarization_model: str = Field(default=DEFAULT_DIARIZATION_MODEL)
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)
    allow_overlap: bool = Field(default=False)
    speaker_db: str | None = Field(default=None)
    speaker_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    vad: bool = Field(default=False)
    reference_rttm: str | None = Field(default=None)
    segmentation_batch_size: int | None = Field(default=None, ge=1)
    embedding_batch_size: int | None = Field(default=None, ge=1)

    # Whisper transcription options
    language: str | None = Field(default=None)
    temperature: str = Field(default=DEFAULT_TEMPERATURE)
    word_timestamps: bool = Field(default=False)
    hallucination_silence_threshold: float | None = Field(default=None, gt=0)
    beam_size: int | None = Field(default=None, ge=1)
    best_of: int | None = Field(default=None, ge=1)
    clip_timestamps: str | None = Field(default=None)
    carry_initial_prompt: bool = Field(default=False)
    condition_on_previous_text: bool = Field(default=True)

    # Decode options
    suppress_tokens: str = Field(default="-1")
    suppress_blank: bool = Field(default=True)
    fp16: bool = Field(default=True)
    patience: float | None = Field(default=None, gt=0)
    length_penalty: float | None = Field(default=None)

    # Output options
    output_format: str = Field(default="json")
    highlight_words: bool = Field(default=False)
    max_line_width: int | None = Field(default=None, ge=1)
    max_line_count: int | None = Field(default=None, ge=1)
    max_words_per_line: int | None = Field(default=None, ge=1)

    # Metadata options
    metadata: bool = Field(default=False)

    # Model options
    download_root: str | None = Field(default=None)

    # Backend options
    backend: str = Field(default="faster-whisper")

    # Audio cleaning options
    clean_level: str = Field(default="moderate")

    # Hallucination detection options
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    hallucination_filter: str = Field(default="auto")

    # Error handling options
    retry_strategy: str = Field(default="smart")

    # Export options
    export: str | None = Field(default=None)
    minotes_sync_dir: str | None = Field(default=None)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        if v not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output format '{v}'. Must be one of: {', '.join(sorted(VALID_OUTPUT_FORMATS))}"
            )
        return v

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{v}'. Must be one of: {', '.join(sorted(VALID_BACKENDS))}"
            )
        return v

    @field_validator("clean_level")
    @classmethod
    def validate_clean_level(cls, v: str) -> str:
        if v not in VALID_CLEAN_LEVELS:
            raise ValueError(
                f"Invalid clean level '{v}'. Must be one of: {', '.join(sorted(VALID_CLEAN_LEVELS))}"
            )
        return v

    @field_validator("hallucination_filter")
    @classmethod
    def validate_hallucination_filter(cls, v: str) -> str:
        if v not in VALID_HALLUCINATION_FILTERS:
            raise ValueError(
                f"Invalid hallucination filter '{v}'. Must be one of: {', '.join(sorted(VALID_HALLUCINATION_FILTERS))}"
            )
        return v

    @field_validator("retry_strategy")
    @classmethod
    def validate_retry_strategy(cls, v: str) -> str:
        if v not in VALID_RETRY_STRATEGIES:
            raise ValueError(
                f"Invalid retry strategy '{v}'. Must be one of: {', '.join(sorted(VALID_RETRY_STRATEGIES))}"
            )
        return v

    @model_validator(mode="after")
    def validate_speaker_counts(self) -> "AudioScriptConfig":
        """Ensure min_speakers <= max_speakers and num_speakers is consistent."""
        if self.min_speakers is not None and self.max_speakers is not None:
            if self.min_speakers > self.max_speakers:
                raise ValueError(
                    f"min_speakers ({self.min_speakers}) must be <= max_speakers ({self.max_speakers})"
                )
        if self.num_speakers is not None:
            if self.min_speakers is not None and self.num_speakers < self.min_speakers:
                raise ValueError(
                    f"num_speakers ({self.num_speakers}) conflicts with min_speakers ({self.min_speakers})"
                )
            if self.max_speakers is not None and self.num_speakers > self.max_speakers:
                raise ValueError(
                    f"num_speakers ({self.num_speakers}) conflicts with max_speakers ({self.max_speakers})"
                )
        return self

    @model_validator(mode="after")
    def validate_diarize_token(self) -> "AudioScriptConfig":
        """Warn early if diarize is set but no token is available."""
        if self.diarize:
            token = self.hf_token or os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError(
                    "diarize=True requires a HuggingFace token. "
                    "Set --hf-token, hf_token in config, or HF_TOKEN env var."
                )
        return self

    def parse_temperature(self) -> float | tuple[float, ...]:
        """Parse temperature string into a float or tuple of floats."""
        parts = [float(t.strip()) for t in self.temperature.split(",")]
        if len(parts) == 1:
            return parts[0]
        return tuple(parts)

    def parse_clip_timestamps(self) -> str | list[float]:
        """Parse clip_timestamps string into a list of floats."""
        if self.clip_timestamps is None:
            return "0"
        parts = [float(t.strip()) for t in self.clip_timestamps.split(",")]
        if len(parts) == 1 and parts[0] == 0.0:
            return "0"
        return parts


# --- Sync configuration models ---


class SyncSourceConfig(BaseModel):
    """Per-source directory overrides for sync."""

    path: str
    tier: TranscriptionTier | None = None
    model: str | None = None
    diarize: bool | None = None
    export: str | None = None
    output_format: str | None = None
    summarize: bool | None = None


class SyncOneDriveConfig(BaseModel):
    """OneDrive Files On-Demand handling config."""

    auto_download: bool = Field(default=True)
    download_timeout: int = Field(default=300)
    download_poll_interval: int = Field(default=5)
    staging_dir: str | None = Field(default=None)
    cleanup_staging: bool = Field(default=True)


class SyncMiNotesConfig(BaseModel):
    """MiNotes-specific sync settings."""

    enabled: bool = Field(default=False)
    folder: str = Field(default="Transcripts")
    journal: bool = Field(default=True)


class SyncConfig(BaseModel):
    """Configuration for the audioscript sync command."""

    sources: list[SyncSourceConfig] = Field(default_factory=list)
    extensions: list[str] = Field(
        default_factory=lambda: ["m4a", "mp3", "wav", "flac", "ogg", "opus", "webm", "mp4", "wma", "aac"]
    )
    recursive: bool = Field(default=True)
    ignore_patterns: list[str] = Field(default_factory=lambda: ["*.tmp", ".*"])
    poll_interval: int = Field(default=300)
    batch_size: int = Field(default=10)
    delay_between: float = Field(default=2.0)
    skip_older_than: int | None = Field(default=None)
    min_file_size: int = Field(default=1024)
    max_file_size: int | None = Field(default=None)
    output_dir: str = Field(default="./transcripts/sync")
    output_format: str = Field(default="markdown")
    onedrive: SyncOneDriveConfig = Field(default_factory=SyncOneDriveConfig)
    minotes: SyncMiNotesConfig = Field(default_factory=SyncMiNotesConfig)


def load_sync_config(config_path: Path | None = None) -> SyncConfig:
    """Load sync configuration from .audioscript.yaml."""
    if config_path is None:
        config_path = Path(".audioscript.yaml")
    file_config = load_yaml_config(config_path)
    sync_data = file_config.get("sync", {})
    if not sync_data:
        return SyncConfig()
    return SyncConfig(**sync_data)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config or {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return {}
    except OSError as e:
        logger.warning("Failed to read config file %s: %s", config_path, e)
        return {}


def merge_configs(cli_args: dict[str, Any], file_config: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI arguments with file configuration.

    Only non-None CLI values override file config.
    """
    merged = {**file_config}
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged


def get_settings(
    cli_args: dict[str, Any], config_path: Path | None = None,
) -> AudioScriptConfig:
    """Get application settings by merging CLI args and config file."""
    if config_path is None:
        config_path = Path(".audioscript.yaml")
    file_config = load_yaml_config(config_path)
    merged = merge_configs(cli_args, file_config)
    return AudioScriptConfig(**merged)
