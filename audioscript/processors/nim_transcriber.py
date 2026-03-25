"""NIM ASR transcription backend using NVIDIA NIM microservice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audioscript.processors.backend_protocol import (
    TranscriberBackend,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)


class NimTranscriber(TranscriberBackend):
    """Transcription backend using NVIDIA NIM ASR microservice.

    Instead of loading a local Whisper model, this sends audio files to a
    NIM ASR HTTP endpoint (e.g. running in a Docker container or on a
    remote server) and parses the OpenAI-compatible response.

    The NIM endpoint is expected to serve ``POST /v1/audio/transcriptions``
    with multipart file upload, returning JSON with ``text`` and optional
    ``segments`` fields.
    """

    def __init__(
        self,
        nim_url: str = "https://integrate.api.nvidia.com/v1",
        model_name: str | None = None,
        timeout: int = 600,
    ):
        self.nim_url = nim_url.rstrip("/")
        self.model_name = model_name  # e.g. "nvidia/parakeet-ctc-1.1b-asr"
        self.timeout = timeout
        self._is_hosted = "nvidia.com" in self.nim_url

    def load_model(self) -> None:
        """No local model to load — the NIM container handles it."""

    def transcribe(
        self,
        audio_path: str | Path,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Send audio to NIM ASR endpoint and return normalized result.

        Args:
            audio_path: Path to audio file.
            **kwargs: Extra options forwarded as form data.
                ``language`` defaults to ``"en"``.

        Returns:
            TranscriptionResult with text, segments, and language.

        Raises:
            RuntimeError: If the NIM endpoint returns a non-200 response.
        """
        import os
        import requests

        audio_path = Path(audio_path)
        language = kwargs.get("language", "en")

        data: dict[str, Any] = {"language": language}
        if self.model_name:
            data["model"] = self.model_name

        # Auth headers for NVIDIA hosted API
        headers: dict[str, str] = {}
        if self._is_hosted:
            api_key = os.environ.get("NVIDIA_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        # Determine endpoint path
        endpoint = f"{self.nim_url}/audio/transcriptions"
        if "localhost" in self.nim_url or "127.0.0.1" in self.nim_url:
            endpoint = f"{self.nim_url}/v1/audio/transcriptions"

        with open(audio_path, "rb") as f:
            response = requests.post(
                endpoint,
                files={"file": (audio_path.name, f)},
                data=data,
                headers=headers,
                timeout=self.timeout,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"NIM ASR returned status {response.status_code}: "
                f"{response.text[:500]}"
            )

        body = response.json()
        text = body.get("text", "")
        raw_segments = body.get("segments", [])

        segments = []
        for idx, seg in enumerate(raw_segments):
            segments.append(
                TranscriptionSegment(
                    id=seg.get("id", idx),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    words=seg.get("words"),
                    confidence=seg.get("confidence"),
                    no_speech_prob=seg.get("no_speech_prob"),
                )
            )

        detected_language = body.get("language", language)

        logger.info(
            "NIM ASR transcription complete: %d segments, %d chars",
            len(segments),
            len(text),
        )

        return TranscriptionResult(
            text=text,
            language=detected_language,
            segments=segments,
            backend=self.backend_name,
            raw=body,
        )

    def detect_language(self, audio_path: str | Path) -> dict[str, Any]:
        """Detect language via the NIM endpoint."""
        import os
        import requests

        audio_path = Path(audio_path)
        data: dict[str, Any] = {}
        if self.model_name:
            data["model"] = self.model_name

        headers: dict[str, str] = {}
        if self._is_hosted:
            api_key = os.environ.get("NVIDIA_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        endpoint = f"{self.nim_url}/audio/transcriptions"
        if "localhost" in self.nim_url or "127.0.0.1" in self.nim_url:
            endpoint = f"{self.nim_url}/v1/audio/transcriptions"

        with open(audio_path, "rb") as f:
            response = requests.post(
                endpoint,
                files={"file": (audio_path.name, f)},
                data=data,
                headers=headers,
                timeout=self.timeout,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"NIM ASR language detection returned status {response.status_code}: "
                f"{response.text[:500]}"
            )

        body = response.json()
        language = body.get("language", "unknown")
        return {
            "language": language,
            "confidence": body.get("language_confidence"),
            "backend": self.backend_name,
        }

    @property
    def backend_name(self) -> str:
        return "nim-asr"

    @property
    def supports_confidence(self) -> bool:
        return True
