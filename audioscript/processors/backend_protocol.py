"""Transcriber backend protocol and shared data structures."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionSegment:
    """Normalized segment from any transcription backend."""

    id: int
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] | None = None
    confidence: float | None = None
    no_speech_prob: float | None = None
    temperature: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, omitting None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class TranscriptionResult:
    """Normalized result from any transcription backend."""

    text: str
    language: str
    segments: list[TranscriptionSegment]
    backend: str
    raw: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict matching the existing JSON output schema."""
        return {
            "text": self.text,
            "language": self.language,
            "segments": [seg.to_dict() for seg in self.segments],
            "backend": self.backend,
        }


class TranscriberBackend(abc.ABC):
    """Abstract base class that all transcription backends must implement."""

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load the transcription model into memory."""

    @abc.abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe an audio file and return normalized results."""

    @abc.abstractmethod
    def detect_language(self, audio_path: str | Path) -> dict[str, Any]:
        """Detect the language of an audio file without full transcription."""

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier string."""

    @property
    @abc.abstractmethod
    def supports_confidence(self) -> bool:
        """Whether this backend provides per-segment confidence scores."""
