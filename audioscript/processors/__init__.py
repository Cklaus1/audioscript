"""Audio processors and transcription backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audioscript.config.settings import AudioScriptConfig
    from audioscript.processors.backend_protocol import TranscriberBackend

logger = logging.getLogger(__name__)


def create_transcriber(settings: AudioScriptConfig) -> TranscriberBackend:
    """Create a faster-whisper transcription backend."""
    from audioscript.processors.faster_whisper_transcriber import (
        FasterWhisperTranscriber,
    )

    return FasterWhisperTranscriber(
        model_name=settings.model,
        tier=settings.tier.value,
        download_root=settings.download_root,
    )
