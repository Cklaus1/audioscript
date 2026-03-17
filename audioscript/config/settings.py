"""Settings management for AudioScript."""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS = {"json", "txt", "vtt", "srt", "tsv", "all"}
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
    input: Optional[str] = None
    output_dir: str = Field(default="./output")
    tier: TranscriptionTier = Field(default=TranscriptionTier.DRAFT)
    version: str = Field(default="1.0")
    clean_audio: bool = Field(default=False)
    summarize: bool = Field(default=False)
    force: bool = Field(default=False)
    model: Optional[str] = None
    no_retry: bool = Field(default=False)
    max_retries: int = Field(default=3, ge=0)

    # Speaker diarization options
    diarize: bool = Field(default=False)
    hf_token: Optional[str] = Field(default=None)
    diarization_model: str = Field(default=DEFAULT_DIARIZATION_MODEL)
    num_speakers: Optional[int] = Field(default=None, ge=1)
    min_speakers: Optional[int] = Field(default=None, ge=1)
    max_speakers: Optional[int] = Field(default=None, ge=1)
    allow_overlap: bool = Field(default=False)
    speaker_db: Optional[str] = Field(default=None)
    speaker_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    vad: bool = Field(default=False)
    reference_rttm: Optional[str] = Field(default=None)
    segmentation_batch_size: Optional[int] = Field(default=None, ge=1)
    embedding_batch_size: Optional[int] = Field(default=None, ge=1)

    # Whisper transcription options
    language: Optional[str] = Field(default=None)
    temperature: str = Field(default=DEFAULT_TEMPERATURE)
    word_timestamps: bool = Field(default=False)
    hallucination_silence_threshold: Optional[float] = Field(default=None, gt=0)
    beam_size: Optional[int] = Field(default=None, ge=1)
    best_of: Optional[int] = Field(default=None, ge=1)
    clip_timestamps: Optional[str] = Field(default=None)
    carry_initial_prompt: bool = Field(default=False)
    condition_on_previous_text: bool = Field(default=True)

    # Decode options
    suppress_tokens: str = Field(default="-1")
    suppress_blank: bool = Field(default=True)
    fp16: bool = Field(default=True)
    patience: Optional[float] = Field(default=None, gt=0)
    length_penalty: Optional[float] = Field(default=None)

    # Output options
    output_format: str = Field(default="json")
    highlight_words: bool = Field(default=False)
    max_line_width: Optional[int] = Field(default=None, ge=1)
    max_line_count: Optional[int] = Field(default=None, ge=1)
    max_words_per_line: Optional[int] = Field(default=None, ge=1)

    # Metadata options
    metadata: bool = Field(default=False)

    # Model options
    download_root: Optional[str] = Field(default=None)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        if v not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output format '{v}'. Must be one of: {', '.join(sorted(VALID_OUTPUT_FORMATS))}"
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

    def parse_temperature(self) -> Union[float, Tuple[float, ...]]:
        """Parse temperature string into a float or tuple of floats."""
        parts = [float(t.strip()) for t in self.temperature.split(",")]
        if len(parts) == 1:
            return parts[0]
        return tuple(parts)

    def parse_clip_timestamps(self) -> Union[str, List[float]]:
        """Parse clip_timestamps string into a list of floats."""
        if self.clip_timestamps is None:
            return "0"
        parts = [float(t.strip()) for t in self.clip_timestamps.split(",")]
        if len(parts) == 1 and parts[0] == 0.0:
            return "0"
        return parts


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
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


def merge_configs(cli_args: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI arguments with file configuration.

    Only non-None CLI values override file config.
    """
    merged = {**file_config}
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged


def get_settings(
    cli_args: Dict[str, Any], config_path: Optional[Path] = None
) -> AudioScriptConfig:
    """Get application settings by merging CLI args and config file."""
    if config_path is None:
        config_path = Path(".audioscript.yaml")
    file_config = load_yaml_config(config_path)
    merged = merge_configs(cli_args, file_config)
    return AudioScriptConfig(**merged)
