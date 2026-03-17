"""Whisper model based transcription implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import whisper
from whisper.utils import get_writer

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Handles transcription using OpenAI's Whisper model."""

    TIER_TO_MODEL = {
        "draft": "base",
        "balanced": "turbo",
        "high_quality": "large-v3",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        tier: str = "draft",
        device: Optional[str] = None,
        download_root: Optional[str] = None,
    ):
        """Initialize the WhisperTranscriber.

        Args:
            model_name: Specific model name to use (overrides tier-based selection).
                        Supports .en variants (e.g. 'base.en') for English-only.
            tier: Transcription quality tier ('draft', 'balanced', or 'high_quality')
            device: Device to run model on ('cpu', 'cuda', etc.). If None, auto-detect.
            download_root: Custom directory for caching model files. If None, uses default.
        """
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
            kwargs = {"name": self.model_name, "device": self.device}
            if self.download_root is not None:
                kwargs["download_root"] = self.download_root
            self.model = whisper.load_model(**kwargs)
            logger.info("Model loaded successfully.")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        checkpoint: Optional[str] = None,
        word_timestamps: bool = False,
        carry_initial_prompt: bool = False,
        hallucination_silence_threshold: Optional[float] = None,
        clip_timestamps: Union[str, List[float]] = "0",
        prepend_punctuations: str = "\"'\u201c\u00bf([{-",
        append_punctuations: str = "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}、",
        # Decode options
        beam_size: Optional[int] = None,
        best_of: Optional[int] = None,
        patience: Optional[float] = None,
        length_penalty: Optional[float] = None,
        suppress_tokens: Optional[str] = "-1",
        suppress_blank: bool = True,
        fp16: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe audio using the Whisper model.

        Supports temperature fallback (pass a tuple to retry with increasing
        randomness on failed segments), word-level timestamps, hallucination
        filtering, and time-range clipping.

        If a checkpoint is provided, its text is used as initial_prompt to prime
        the language model with prior context. This does NOT skip already-transcribed
        audio — Whisper always processes the full file.
        """
        self.load_model()

        # Use checkpoint text as initial prompt for context continuity
        initial_prompt = None
        if checkpoint:
            try:
                checkpoint_data = json.loads(checkpoint)
                if "text" in checkpoint_data:
                    initial_prompt = checkpoint_data["text"]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse checkpoint, proceeding without it.")

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=verbose,
            initial_prompt=initial_prompt,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
            carry_initial_prompt=carry_initial_prompt,
            hallucination_silence_threshold=hallucination_silence_threshold,
            clip_timestamps=clip_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            suppress_tokens=suppress_tokens,
            suppress_blank=suppress_blank,
            fp16=fp16,
        )

        # Post-process: remove duplicate consecutive segments
        if "segments" in result:
            filtered_segments = []
            prev_text = None

            for segment in result["segments"]:
                current_text = segment.get("text", "").strip()
                if not current_text or current_text == prev_text:
                    continue
                filtered_segments.append(segment)
                prev_text = current_text

            result["segments"] = filtered_segments
            if filtered_segments:
                result["text"] = " ".join(
                    seg.get("text", "").strip() for seg in filtered_segments
                )

        return result

    def detect_language(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Detect the language of an audio file without full transcription.

        Returns:
            Dict with 'language' (code) and 'language_probability' keys.
        """
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

    def save_formatted_output(
        self,
        result: Dict[str, Any],
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        output_format: str = "json",
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save transcription results using Whisper's built-in format writers.

        Args:
            result: Transcription result dict from transcribe()
            audio_path: Original audio file path (used for naming output files)
            output_dir: Directory to write output files
            output_format: One of 'json', 'txt', 'vtt', 'srt', 'tsv', 'all'
            options: Writer options (highlight_words, max_line_width, etc.)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        writer = get_writer(output_format, str(output_dir))
        writer(result, audio_path, options or {})

    def clean_audio(
        self, audio_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Clean audio file to improve transcription quality.

        Placeholder implementation — currently copies the file unchanged.
        Replace with librosa/ffmpeg-based noise reduction for production use.
        """
        if output_path is None:
            input_path = Path(audio_path)
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"

        import shutil

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(audio_path, output_path)

        return output_path

    def generate_summary(self, transcription: Dict[str, Any]) -> str:
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
        transcription: Dict[str, Any],
        output_path: Union[str, Path],
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

    def save_summary(self, summary: str, output_path: Union[str, Path]) -> None:
        """Save summary text to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)

    def create_checkpoint(self, transcription: Dict[str, Any]) -> str:
        """Create a context-priming checkpoint from the current transcription.

        Stores the transcribed text for use as initial_prompt in future runs.
        This provides language model context continuity, but does NOT enable
        skipping already-transcribed portions of audio.
        """
        return json.dumps({"text": transcription.get("text", "")})
