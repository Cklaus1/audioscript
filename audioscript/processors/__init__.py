"""Audio processors and transcription backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audioscript.config.settings import AudioScriptConfig
    from audioscript.processors.backend_protocol import TranscriberBackend

logger = logging.getLogger(__name__)


def create_transcriber(settings: AudioScriptConfig) -> TranscriberBackend:
    """Factory: create the right transcription backend based on settings.

    Falls back to vanilla Whisper if faster-whisper is not installed.
    """
    if settings.backend == "faster-whisper":
        try:
            from audioscript.processors.faster_whisper_transcriber import (
                FasterWhisperTranscriber,
            )

            return FasterWhisperTranscriber(
                model_name=settings.model,
                tier=settings.tier.value,
                download_root=settings.download_root,
            )
        except ImportError:
            logger.warning(
                "faster-whisper not installed, falling back to vanilla Whisper. "
                "Install with: pip install faster-whisper"
            )
            from audioscript.processors.whisper_transcriber import WhisperTranscriber

            return WhisperTranscriber(
                model_name=settings.model,
                tier=settings.tier.value,
                download_root=settings.download_root,
            )
    else:
        from audioscript.processors.whisper_transcriber import WhisperTranscriber

        return WhisperTranscriber(
            model_name=settings.model,
            tier=settings.tier.value,
            download_root=settings.download_root,
        )
