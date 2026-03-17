"""Audio processing and transcription functionality."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from audioscript.config.settings import AudioScriptConfig
from audioscript.processors.whisper_transcriber import WhisperTranscriber
from audioscript.utils.file_utils import ProcessingManifest, get_file_hash, get_output_path
from audioscript.utils.metadata import extract_metadata

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio processing and transcription."""

    def __init__(
        self,
        settings: AudioScriptConfig,
        manifest: ProcessingManifest,
        console: Optional[Console] = None,
    ) -> None:
        self.settings = settings
        self.manifest = manifest
        self.console = console or Console()
        self._transcriber: Optional[WhisperTranscriber] = None
        self._diarizer: Optional[Any] = None
        self._speaker_db: Optional[Any] = None

    def _get_transcriber(self) -> WhisperTranscriber:
        """Get or lazily initialize the Whisper transcriber."""
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber(
                model_name=self.settings.model,
                tier=self.settings.tier.value,
                download_root=self.settings.download_root,
            )
        return self._transcriber

    def _get_diarizer(self) -> Any:
        """Get or lazily initialize the speaker diarizer."""
        if self._diarizer is None:
            from audioscript.processors.diarizer import SpeakerDiarizer

            self._diarizer = SpeakerDiarizer(
                hf_token=self.settings.hf_token,
                model=self.settings.diarization_model,
                cache_dir=self.settings.download_root,
                segmentation_batch_size=self.settings.segmentation_batch_size,
                embedding_batch_size=self.settings.embedding_batch_size,
            )
        return self._diarizer

    def _get_speaker_db(self) -> Optional[Any]:
        """Get or lazily initialize the speaker database."""
        if self._speaker_db is None and self.settings.speaker_db:
            from audioscript.processors.diarizer import SpeakerDatabase

            db_path = Path(self.settings.speaker_db)
            if not db_path.exists():
                self.console.print(
                    f"[yellow]Warning:[/] Speaker database not found: {db_path}. "
                    "Starting with empty database."
                )
            self._speaker_db = SpeakerDatabase(self.settings.speaker_db)
            names = self._speaker_db.speaker_names
            if names:
                self.console.print(
                    f"Loaded speaker database with [bold]{len(names)}[/] speakers: "
                    f"{', '.join(names)}"
                )
        return self._speaker_db

    def process_file(self, file_path: Path) -> bool:
        """Process a single audio file.

        Returns True if processing succeeded, False otherwise.
        Retries up to max_retries times with exponential backoff unless
        no_retry is set.
        """
        file_hash = get_file_hash(file_path)

        # Skip already-processed files unless --force
        if (
            not self.settings.force
            and self.manifest.is_processed(
                file_hash, self.settings.tier.value, self.settings.version
            )
        ):
            self.console.print(f"Skipping already processed file: {file_path.name}")
            return True

        checkpoint = self.manifest.get_checkpoint(file_hash)
        max_attempts = 1 if self.settings.no_retry else self.settings.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            self.manifest.update_file_status(
                file_hash, "processing",
                self.settings.tier.value, self.settings.version, checkpoint,
            )

            try:
                transcriber = self._get_transcriber()
                output_dir = Path(self.settings.output_dir)
                transcription_path = get_output_path(file_path, output_dir, "json")

                # Clean audio if requested
                audio_to_process = file_path
                if self.settings.clean_audio:
                    self.console.print(
                        f"[yellow]Warning:[/] --clean-audio is a placeholder "
                        f"(copies file unchanged). Proceeding with: {file_path.name}"
                    )
                    cleaned_path = output_dir / "cleaned" / file_path.name
                    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
                    audio_to_process = transcriber.clean_audio(file_path, cleaned_path)

                # Run standalone VAD if requested (C1 fix: independent of --diarize)
                vad_clips: Optional[List[float]] = None
                if self.settings.vad:
                    diarizer = self._get_diarizer()
                    self.console.print(f"Running VAD: {file_path.name}")
                    vad_result = diarizer.detect_speech(audio_to_process)
                    self.console.print(
                        f"Speech: {vad_result['speech_percentage']}% "
                        f"({vad_result['total_speech_duration']}s / {vad_result['total_duration']}s)"
                    )
                    # Save VAD timeline
                    vad_path = get_output_path(file_path, output_dir, "vad.json")
                    vad_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(vad_path, "w") as f:
                        json.dump(vad_result, f, indent=2)

                    # H5 fix: Convert VAD speech regions to clip_timestamps
                    # so Whisper only processes speech (reduces hallucinations)
                    if vad_result["speech_segments"] and self.settings.clip_timestamps is None:
                        vad_clips = []
                        for seg in vad_result["speech_segments"]:
                            vad_clips.extend([seg["start"], seg["end"]])

                # Diarization requires word_timestamps for accurate speaker assignment
                word_timestamps = self.settings.word_timestamps
                if self.settings.diarize and not word_timestamps:
                    word_timestamps = True
                    self.console.print(
                        "[dim]Enabling word_timestamps (required for diarization)[/]"
                    )

                self.console.print(
                    f"Transcribing: {file_path.name} "
                    f"(tier={self.settings.tier.value}, attempt {attempt}/{max_attempts})"
                )

                # Determine clip_timestamps: user-provided > VAD-derived > default
                clip_ts = self.settings.parse_clip_timestamps()
                if vad_clips is not None:
                    clip_ts = vad_clips

                # Build transcription kwargs from settings
                result = transcriber.transcribe(
                    audio_to_process,
                    language=self.settings.language,
                    verbose=True,
                    checkpoint=checkpoint,
                    temperature=self.settings.parse_temperature(),
                    word_timestamps=word_timestamps,
                    hallucination_silence_threshold=self.settings.hallucination_silence_threshold,
                    beam_size=self.settings.beam_size,
                    best_of=self.settings.best_of,
                    clip_timestamps=clip_ts,
                    carry_initial_prompt=self.settings.carry_initial_prompt,
                    condition_on_previous_text=self.settings.condition_on_previous_text,
                    suppress_tokens=self.settings.suppress_tokens,
                    suppress_blank=self.settings.suppress_blank,
                    fp16=self.settings.fp16,
                    patience=self.settings.patience,
                    length_penalty=self.settings.length_penalty,
                )

                # C2 fix: Save checkpoint immediately after transcription
                # so retries don't lose transcription context
                new_checkpoint = transcriber.create_checkpoint(result)
                checkpoint = new_checkpoint  # Update for potential retry
                self.manifest.update_file_status(
                    file_hash, "transcribed",
                    self.settings.tier.value, self.settings.version, new_checkpoint,
                )

                # Embed audio file metadata if requested
                if self.settings.metadata:
                    try:
                        result["metadata"] = extract_metadata(file_path)
                    except Exception as meta_err:
                        logger.warning("Metadata extraction failed: %s", meta_err)

                # Save JSON before diarization (work is preserved even if diarization fails)
                transcriber.save_results(result, transcription_path)

                # Run speaker diarization if requested
                if self.settings.diarize:
                    try:
                        result = self._run_diarization(
                            result, audio_to_process, file_path, output_dir,
                        )
                        # Re-save JSON with diarization data
                        transcriber.save_results(result, transcription_path)
                    except Exception as diar_err:
                        self.console.print(
                            f"[yellow]Warning:[/] Diarization failed for {file_path.name}: {diar_err}. "
                            "Transcription saved without speaker labels."
                        )
                        logger.warning("Diarization failed: %s", diar_err, exc_info=True)

                # Save additional output formats if requested
                fmt = self.settings.output_format
                if fmt != "json":
                    writer_options = {
                        "highlight_words": self.settings.highlight_words,
                        "max_line_width": self.settings.max_line_width,
                        "max_line_count": self.settings.max_line_count,
                        "max_words_per_line": self.settings.max_words_per_line,
                    }
                    transcriber.save_formatted_output(
                        result, audio_to_process, output_dir,
                        output_format=fmt, options=writer_options,
                    )

                # Generate summary if requested
                if self.settings.summarize:
                    self.console.print(f"Generating summary: {file_path.name}")
                    summary_path = get_output_path(file_path, output_dir, "summary.txt")
                    summary = transcriber.generate_summary(result)
                    transcriber.save_summary(summary, summary_path)

                # Mark as completed
                self.manifest.update_file_status(
                    file_hash, "completed",
                    self.settings.tier.value, self.settings.version, new_checkpoint,
                )
                return True

            except Exception as e:
                self.manifest.update_file_status(
                    file_hash, "error",
                    self.settings.tier.value, self.settings.version,
                    checkpoint, str(e),
                )

                if attempt >= max_attempts:
                    self.console.print(f"[red]Failed[/] {file_path.name}: {e}")
                    return False

                self.console.print(
                    f"[yellow]Error[/] processing {file_path.name}: {e}. "
                    f"Retrying ({attempt}/{self.settings.max_retries})..."
                )
                time.sleep(min(2 ** (attempt - 1), 10))

        return False

    def _run_diarization(
        self,
        result: Dict[str, Any],
        audio_path: Path,
        file_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run diarization pipeline and merge with transcription result."""
        self.console.print(f"Diarizing: {file_path.name}")
        diarizer = self._get_diarizer()

        def diar_progress(step: str, completed: Optional[int], total: Optional[int]) -> None:
            if step:
                self.console.print(f"  [dim]{step}...[/]")

        diar_result = diarizer.diarize(
            audio_path,
            num_speakers=self.settings.num_speakers,
            min_speakers=self.settings.min_speakers,
            max_speakers=self.settings.max_speakers,
            allow_overlap=self.settings.allow_overlap,
            progress_callback=diar_progress,
        )

        # Speaker identification via database
        speaker_db = self._get_speaker_db()
        result = diarizer.assign_speakers(
            result, diar_result,
            speaker_db=speaker_db,
            similarity_threshold=self.settings.speaker_similarity_threshold,
            allow_overlap=self.settings.allow_overlap,
        )

        # Report
        self.console.print(f"Found [bold]{diar_result['num_speakers']}[/] speakers")
        overlap = diar_result.get("overlap", {})
        if overlap.get("overlap_percentage", 0) > 0:
            self.console.print(
                f"Overlapping speech: [bold]{overlap['overlap_percentage']}%[/]"
            )

        # Save RTTM
        rttm_path = get_output_path(file_path, output_dir, "rttm")
        diarizer.save_rttm(diar_result["segments"], rttm_path, file_id=file_path.stem)

        # Save speaker embeddings
        if diar_result.get("speaker_embeddings"):
            emb_path = get_output_path(file_path, output_dir, "embeddings.json")
            diarizer.save_embeddings(diar_result["speaker_embeddings"], emb_path)

        # H4 fix: Evaluate with error handling
        if self.settings.reference_rttm:
            ref_path = Path(self.settings.reference_rttm)
            if not ref_path.exists():
                self.console.print(
                    f"[yellow]Warning:[/] Reference RTTM not found: {ref_path}"
                )
            else:
                try:
                    metrics = diarizer.evaluate(
                        diar_result["segments"], ref_path, file_id=file_path.stem,
                    )
                    self.console.print(
                        f"DER: [bold]{metrics['diarization_error_rate']:.1%}[/] "
                        f"(missed={metrics['missed_speech']:.1%}, "
                        f"false_alarm={metrics['false_alarm']:.1%}, "
                        f"confusion={metrics['speaker_confusion']:.1%})"
                    )
                    result["diarization"]["evaluation"] = metrics
                except Exception as eval_err:
                    self.console.print(
                        f"[yellow]Warning:[/] DER evaluation failed: {eval_err}"
                    )

        return result
