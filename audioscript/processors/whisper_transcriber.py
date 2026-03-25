"""Whisper model based transcription implementation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Lazy imports — these are heavy (~10GB) and only needed when this backend is used
torch = None  # type: ignore
whisper = None  # type: ignore
get_writer = None  # type: ignore


def _ensure_whisper_imports() -> None:
    """Import torch/whisper on first use."""
    global torch, whisper, get_writer
    if torch is None:
        try:
            import torch as _torch
            import whisper as _whisper
            from whisper.utils import get_writer as _get_writer
            torch = _torch
            whisper = _whisper
            get_writer = _get_writer
        except ImportError as e:
            raise ImportError(
                f"Whisper backend requires torch and openai-whisper: {e}. "
                "Use faster-whisper backend instead (default)."
            ) from e

from audioscript.processors.backend_protocol import (
    TranscriberBackend,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)


class WhisperTranscriber(TranscriberBackend):
    """Transcription backend using OpenAI's Whisper model."""

    TIER_TO_MODEL = {
        "draft": "base",
        "balanced": "turbo",
        "high_quality": "large-v3",
    }

    def __init__(
        self,
        model_name: str | None = None,
        tier: str = "draft",
        device: str | None = None,
        download_root: str | None = None,
    ):
        _ensure_whisper_imports()

        if model_name is None:
            model_name = self.TIER_TO_MODEL.get(tier, "base")

        available = whisper.available_models()
        if model_name not in available:
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {', '.join(available)}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.tier = tier
        self.download_root = download_root
        self.model = None

    def load_model(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            logger.info("Loading Whisper model '%s' on %s...", self.model_name, self.device)
            kwargs: dict[str, Any] = {"name": self.model_name, "device": self.device}
            if self.download_root is not None:
                kwargs["download_root"] = self.download_root
            self.model = whisper.load_model(**kwargs)
            logger.info("Model loaded successfully.")

    @property
    def backend_name(self) -> str:
        return "whisper"

    @property
    def supports_confidence(self) -> bool:
        return False

    def transcribe(self, audio_path: str | Path, **kwargs: Any) -> TranscriptionResult:
        """Transcribe audio using the Whisper model.

        Returns a normalized TranscriptionResult. For backwards compatibility,
        use .to_dict() to get the old dict format.
        """
        self.load_model()

        # Extract checkpoint and convert to initial_prompt
        checkpoint = kwargs.pop("checkpoint", None)
        initial_prompt = kwargs.pop("initial_prompt", None)
        if checkpoint and not initial_prompt:
            try:
                checkpoint_data = json.loads(checkpoint)
                if "text" in checkpoint_data:
                    initial_prompt = checkpoint_data["text"]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse checkpoint, proceeding without it.")

        # Strip kwargs not accepted by whisper.transcribe
        kwargs.pop("vad_filter", None)

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        raw = self.model.transcribe(
            audio_path,
            initial_prompt=initial_prompt,
            **kwargs,
        )

        # Post-process: remove duplicate consecutive segments
        segments = []
        prev_text = None
        for seg in raw.get("segments", []):
            current_text = seg.get("text", "").strip()
            if not current_text or current_text == prev_text:
                continue
            segments.append(seg)
            prev_text = current_text

        # Convert to normalized segments
        norm_segments = [
            TranscriptionSegment(
                id=seg.get("id", i),
                start=seg["start"],
                end=seg["end"],
                text=seg.get("text", "").strip(),
                words=seg.get("words"),
                confidence=None,  # vanilla Whisper doesn't expose clean confidence
                no_speech_prob=seg.get("no_speech_prob"),
                temperature=seg.get("temperature"),
                avg_logprob=seg.get("avg_logprob"),
                compression_ratio=seg.get("compression_ratio"),
            )
            for i, seg in enumerate(segments)
        ]

        text = " ".join(seg.text for seg in norm_segments) if norm_segments else raw.get("text", "")

        result = TranscriptionResult(
            text=text,
            language=raw.get("language", ""),
            segments=norm_segments,
            backend=self.backend_name,
            raw=raw,
        )

        return result

    def detect_language(self, audio_path: str | Path) -> dict[str, Any]:
        """Detect the language of an audio file without full transcription."""
        self.load_model()

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        return {
            "language": detected_lang,
            "language_probability": probs[detected_lang],
            "all_probabilities": dict(sorted(probs.items(), key=lambda x: -x[1])[:10]),
        }

    # --- Utility methods (not part of TranscriberBackend protocol) ---

    def save_formatted_output(
        self,
        result: dict[str, Any],
        audio_path: str | Path,
        output_dir: str | Path,
        output_format: str = "json",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Save transcription results using Whisper's built-in format writers."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        writer = get_writer(output_format, str(output_dir))
        writer(result, audio_path, options or {})

    def generate_summary(self, transcription: dict[str, Any]) -> str:
        """Generate a basic extractive summary (first 25 words).

        For production use, replace with an LLM-based summarizer.
        """
        text = transcription.get("text", "")
        words = text.split()
        if len(words) <= 25:
            return text
        return " ".join(words[:25]) + "..."

    def save_results(
        self,
        transcription: dict[str, Any],
        output_path: str | Path,
        include_segments: bool = True,
    ) -> None:
        """Save transcription results to a JSON file."""
        output = dict(transcription)
        if not include_segments:
            output.pop("segments", None)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    def save_summary(self, summary: str, output_path: str | Path) -> None:
        """Save summary text to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)

    def create_checkpoint(self, transcription: dict[str, Any]) -> str:
        """Create a context-priming checkpoint from transcription text."""
        text = transcription.get("text", "")
        if isinstance(transcription, TranscriptionResult):
            text = transcription.text
        return json.dumps({"text": text})
