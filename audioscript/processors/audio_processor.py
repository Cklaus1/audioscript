"""Audio processing and transcription functionality."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from audioscript.processors.whisper_transcriber import WhisperTranscriber
from audioscript.utils.file_utils import ProcessingManifest, get_file_hash, get_output_path


class AudioProcessor:
    """Handles audio processing and transcription."""

    def __init__(
        self,
        settings: Dict[str, Any],
        manifest: ProcessingManifest,
    ):
        """Initialize the audio processor.

        Args:
            settings: Configuration settings
            manifest: Processing manifest
        """
        self.settings = settings
        self.manifest = manifest
        self.transcriber = None

    def _get_transcriber(self) -> WhisperTranscriber:
        """Get or initialize the transcriber.

        Returns:
            Initialized WhisperTranscriber instance
        """
        if self.transcriber is None:
            # Initialize transcriber with settings
            self.transcriber = WhisperTranscriber(
                model_name=self.settings.get("model"),
                tier=self.settings["tier"],
            )
        return self.transcriber

    def process_file(self, file_path: Path) -> bool:
        """Process a single audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if processing was successful, False otherwise
        """
        # Get file hash for tracking
        file_hash = get_file_hash(file_path)

        # Check if file is already processed (unless force flag is set)
        if (
            not self.settings["force"] and
            self.manifest.is_processed(
                file_hash,
                self.settings["tier"],
                self.settings["version"]
            )
        ):
            print(f"Skipping already processed file: {file_path.name}")
            return True

        # Get the current status and checkpoint
        status = self.manifest.get_status(file_hash)
        checkpoint = self.manifest.get_checkpoint(file_hash)

        # Update status to processing
        self.manifest.update_file_status(
            file_hash,
            "processing",
            self.settings["tier"],
            self.settings["version"],
            checkpoint,
        )

        try:
            # Get the transcriber
            transcriber = self._get_transcriber()

            # Get output paths
            output_dir = Path(self.settings["output_dir"])
            transcription_path = get_output_path(file_path, output_dir, "json")

            # Clean audio if requested
            audio_to_process = file_path
            if self.settings["clean_audio"]:
                print(f"Cleaning audio file: {file_path.name}")
                cleaned_path = output_dir / "cleaned" / file_path.name
                audio_to_process = transcriber.clean_audio(file_path, cleaned_path)

            # Perform transcription based on tier
            print(f"Transcribing audio file: {file_path.name} with tier: {self.settings['tier']}")
            result = transcriber.transcribe(
                audio_to_process,
                verbose=True,
                checkpoint=checkpoint,
            )

            # Save transcription results
            transcriber.save_results(result, transcription_path)

            # Create new checkpoint for future use
            new_checkpoint = transcriber.create_checkpoint(result)

            # Generate summary if requested
            if self.settings["summarize"]:
                print(f"Generating summary for: {file_path.name}")
                summary_path = get_output_path(file_path, output_dir, "summary.txt")
                summary = transcriber.generate_summary(result)
                transcriber.save_summary(summary, summary_path)

            # Update status to completed
            self.manifest.update_file_status(
                file_hash,
                "completed",
                self.settings["tier"],
                self.settings["version"],
                new_checkpoint,
            )

            return True

        except Exception as e:
            # Update status to error
            self.manifest.update_file_status(
                file_hash,
                "error",
                self.settings["tier"],
                self.settings["version"],
                checkpoint,
                str(e),
            )

            # Retry if needed
            if not self.settings["no_retry"]:
                print(f"Error processing {file_path.name}: {e}. Will retry...")
                # Wait a bit before retrying
                time.sleep(1)
                return self.process_file(file_path)
            else:
                print(f"Error processing {file_path.name}: {e}")
                return False

    def _transcribe_draft(
        self,
        audio_path: Path,
        output_path: Path,
        checkpoint: Optional[str] = None,
    ) -> None:
        """Perform draft-quality transcription (legacy method).

        Args:
            audio_path: Path to the audio file
            output_path: Path to save the transcription
            checkpoint: Optional checkpoint for resuming
        """
        transcriber = self._get_transcriber()
        result = transcriber.transcribe(audio_path, checkpoint=checkpoint)
        transcriber.save_results(result, output_path)

    def _transcribe_high_quality(
        self,
        audio_path: Path,
        output_path: Path,
        checkpoint: Optional[str] = None,
    ) -> None:
        """Perform high-quality transcription (legacy method).

        Args:
            audio_path: Path to the audio file
            output_path: Path to save the transcription
            checkpoint: Optional checkpoint for resuming
        """
        transcriber = self._get_transcriber()
        result = transcriber.transcribe(audio_path, checkpoint=checkpoint)
        transcriber.save_results(result, output_path)

    def _generate_summary(self, transcription_path: Path, summary_path: Path) -> None:
        """Generate a summary of the transcription (legacy method).

        Args:
            transcription_path: Path to the transcription file
            summary_path: Path to save the summary
        """
        # Load the transcription
        with open(transcription_path, "r", encoding="utf-8") as f:
            transcription = json.load(f)

        # Generate and save the summary
        transcriber = self._get_transcriber()
        summary = transcriber.generate_summary(transcription)
        transcriber.save_summary(summary, summary_path)