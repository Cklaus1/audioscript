"""Faster-whisper transcription backend using CTranslate2."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from audioscript.processors.backend_protocol import (
    TranscriberBackend,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(TranscriberBackend):
    """Transcription backend using faster-whisper (CTranslate2).

    Provides 4-6x speedup over vanilla Whisper with reduced VRAM usage,
    built-in Silero VAD, and per-token log probability access for
    confidence scoring.
    """

    TIER_TO_MODEL = {
        "draft": "base",
        "balanced": "turbo",
        "high_quality": "large-v3",
    }

    TIER_TO_COMPUTE = {
        "draft": "int8",
        "balanced": "float16",
        "high_quality": "float16",
    }

    def __init__(
        self,
        model_name: str | None = None,
        tier: str = "draft",
        device: str | None = None,
        compute_type: str | None = None,
        download_root: str | None = None,
    ):
        if model_name is None:
            model_name = self.TIER_TO_MODEL.get(tier, "base")

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type is None:
            compute_type = self.TIER_TO_COMPUTE.get(tier, "int8")
            # CPU doesn't support float16
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.tier = tier
        self.download_root = download_root
        self.model = None

    def load_model(self) -> None:
        """Load the faster-whisper model if not already loaded."""
        if self.model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model '%s' on %s (compute=%s)...",
                self.model_name, self.device, self.compute_type,
            )
            kwargs: dict[str, Any] = {
                "model_size_or_path": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
            }
            if self.download_root is not None:
                kwargs["download_root"] = self.download_root
            self.model = WhisperModel(**kwargs)
            logger.info("Model loaded successfully.")

    @property
    def backend_name(self) -> str:
        return "faster-whisper"

    @property
    def supports_confidence(self) -> bool:
        return True

    def transcribe(self, audio_path: str | Path, **kwargs: Any) -> TranscriptionResult:
        """Transcribe audio using faster-whisper.

        Adapts the faster-whisper generator-based API to our normalized
        TranscriptionResult format. Extracts confidence scores from
        avg_logprob for hallucination detection.
        """
        self.load_model()

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        # Extract and adapt kwargs for faster-whisper API
        checkpoint = kwargs.pop("checkpoint", None)
        initial_prompt = kwargs.pop("initial_prompt", None)
        if checkpoint and not initial_prompt:
            try:
                checkpoint_data = json.loads(checkpoint)
                if "text" in checkpoint_data:
                    initial_prompt = checkpoint_data["text"]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse checkpoint, proceeding without it.")

        # Map common kwargs to faster-whisper parameter names
        fw_kwargs: dict[str, Any] = {}

        if initial_prompt is not None:
            fw_kwargs["initial_prompt"] = initial_prompt

        # Direct passthrough params
        for key in (
            "language", "task", "temperature", "beam_size", "best_of",
            "patience", "length_penalty", "condition_on_previous_text",
            "word_timestamps", "vad_filter",
        ):
            if key in kwargs and kwargs[key] is not None:
                fw_kwargs[key] = kwargs[key]

        # Adapt parameter names that differ
        if "suppress_tokens" in kwargs and kwargs["suppress_tokens"] is not None:
            tokens_str = kwargs["suppress_tokens"]
            if isinstance(tokens_str, str):
                fw_kwargs["suppress_tokens"] = [int(t) for t in tokens_str.split(",")]
            else:
                fw_kwargs["suppress_tokens"] = tokens_str

        if "suppress_blank" in kwargs and kwargs["suppress_blank"] is not None:
            fw_kwargs["suppress_blank"] = kwargs["suppress_blank"]

        if "hallucination_silence_threshold" in kwargs and kwargs["hallucination_silence_threshold"] is not None:
            fw_kwargs["hallucination_silence_threshold"] = kwargs["hallucination_silence_threshold"]

        if "clip_timestamps" in kwargs and kwargs["clip_timestamps"] is not None:
            clip_ts = kwargs["clip_timestamps"]
            if isinstance(clip_ts, str) and clip_ts != "0":
                fw_kwargs["clip_timestamps"] = [float(t) for t in clip_ts.split(",")]
            elif isinstance(clip_ts, list):
                fw_kwargs["clip_timestamps"] = clip_ts

        # Consume the generator
        segments_gen, info = self.model.transcribe(audio_path, **fw_kwargs)
        raw_segments = list(segments_gen)

        # Convert to normalized segments with duplicate filtering
        norm_segments: list[TranscriptionSegment] = []
        prev_text = None
        for i, seg in enumerate(raw_segments):
            text = seg.text.strip()
            if not text or text == prev_text:
                continue

            # Extract word-level data if available
            words = None
            if hasattr(seg, "words") and seg.words:
                words = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        **({"probability": w.probability} if hasattr(w, "probability") else {}),
                    }
                    for w in seg.words
                ]

            # Compute confidence from avg_logprob (clamped to 0-1)
            confidence = None
            if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                confidence = min(1.0, max(0.0, math.exp(seg.avg_logprob)))

            norm_segments.append(TranscriptionSegment(
                id=seg.id if hasattr(seg, "id") else i,
                start=seg.start,
                end=seg.end,
                text=text,
                words=words,
                confidence=confidence,
                no_speech_prob=seg.no_speech_prob if hasattr(seg, "no_speech_prob") else None,
                avg_logprob=seg.avg_logprob if hasattr(seg, "avg_logprob") else None,
                compression_ratio=seg.compression_ratio if hasattr(seg, "compression_ratio") else None,
            ))
            prev_text = text

        full_text = " ".join(seg.text for seg in norm_segments)

        return TranscriptionResult(
            text=full_text,
            language=info.language if hasattr(info, "language") else "",
            segments=norm_segments,
            backend=self.backend_name,
            raw={"segments": raw_segments, "info": info},
        )

    def detect_language(self, audio_path: str | Path) -> dict[str, Any]:
        """Detect the language of an audio file without full transcription."""
        self.load_model()

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        # Use detect_language_multi_segment for efficiency (no full transcription)
        try:
            info = self.model.detect_language(audio_path)
            # Returns list of (language, probability) tuples
            if info:
                best_lang, best_prob = info[0]
                return {
                    "language": best_lang,
                    "language_probability": best_prob,
                    "all_probabilities": dict(info[:10]),
                }
        except (AttributeError, TypeError):
            pass

        # Fallback: transcribe first segment only
        segments, info = self.model.transcribe(audio_path, task="transcribe")
        # Consume just the first segment to release resources
        for _ in segments:
            break

        return {
            "language": info.language if hasattr(info, "language") else "",
            "language_probability": info.language_probability if hasattr(info, "language_probability") else 0.0,
        }
