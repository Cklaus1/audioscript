"""MiNotes export integration.

Hybrid approach:
- Writes markdown to a MiNotes sync directory (bulk content)
- Calls `minotes` CLI for metadata operations (properties, plugin registration)
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PLUGIN_NAME = "audioscript"
_PLUGIN_VERSION = "0.2.0"
_TRANSCRIPT_CLASS = "transcript"


class MiNotesExporter:
    """Export transcriptions to MiNotes via markdown sync + CLI."""

    def __init__(
        self,
        sync_dir: str | Path | None = None,
    ) -> None:
        if sync_dir:
            self.sync_dir = Path(sync_dir)
        else:
            self.sync_dir = self._default_sync_dir()
        self._registered = False
        self._class_registered = False

    @staticmethod
    def _default_sync_dir() -> Path:
        """Default sync directory for transcript markdown files."""
        return Path.home() / ".audioscript" / "minotes-sync"

    def is_available(self) -> bool:
        """Check if the minotes CLI is installed and on PATH."""
        try:
            result = subprocess.run(
                ["minotes", "--version"],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def ensure_registered(self) -> None:
        """Register audioscript as a MiNotes plugin (idempotent)."""
        if self._registered:
            return

        try:
            subprocess.run(
                [
                    "minotes", "plugin", "register",
                    "--name", _PLUGIN_NAME,
                    "--version", _PLUGIN_VERSION,
                    "--description", "Audio transcription integration",
                ],
                capture_output=True, timeout=10,
            )
            self._registered = True
            logger.info("Registered as MiNotes plugin: %s", _PLUGIN_NAME)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("Failed to register MiNotes plugin: %s", e)

    def ensure_transcript_class(self) -> None:
        """Register the transcript property class in MiNotes (idempotent)."""
        if self._class_registered:
            return

        schemas = [
            ("source_file", "text"),
            ("duration", "text"),
            ("language", "text"),
            ("speakers", "number"),
            ("transcribed_at", "datetime"),
            ("confidence_avg", "number"),
        ]

        for name, value_type in schemas:
            try:
                subprocess.run(
                    [
                        "minotes", "property", "schema", "create",
                        "--name", name,
                        "--type", value_type,
                        "--class", _TRANSCRIPT_CLASS,
                    ],
                    capture_output=True, timeout=10,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass  # Schema may already exist

        self._class_registered = True

    def export(
        self,
        markdown_content: str,
        audio_path: Path,
        result_dict: dict[str, Any],
    ) -> Path:
        """Export a transcription to MiNotes.

        1. Write markdown to sync directory
        2. Set properties via CLI
        3. Update sync state
        """
        transcript_dir = self.sync_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        output_name = f"{audio_path.stem}.md"
        output_path = transcript_dir / output_name
        output_path.write_text(markdown_content, encoding="utf-8")

        logger.info("Wrote transcript markdown to %s", output_path)

        # Set properties via CLI
        self._set_properties(audio_path, result_dict)

        # Update sync state
        self._update_state(audio_path.stem, output_path)

        return output_path

    def is_already_exported(self, audio_path: Path) -> bool:
        """Check if this audio file has already been exported."""
        state = self._load_state()
        return audio_path.stem in state

    def journal_entry(self, audio_path: Path, summary: str) -> None:
        """Add a journal entry linking to the transcript."""
        truncated = summary[:200] + "..." if len(summary) > 200 else summary
        text = f"Transcribed [[Transcript: {audio_path.stem}]]: {truncated}"

        try:
            subprocess.run(
                [
                    "minotes", "block", "create",
                    "--page", "journal",
                    "--content", text,
                ],
                capture_output=True, timeout=10,
            )
            logger.info("Added journal entry for %s", audio_path.stem)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("Failed to add journal entry: %s", e)

    def _set_properties(self, audio_path: Path, result_dict: dict[str, Any]) -> None:
        """Set transcript properties via minotes CLI."""
        page_title = f"Transcript: {audio_path.stem}"
        props = {
            "source_file": str(audio_path),
            "language": result_dict.get("language", ""),
            "transcribed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        diar = result_dict.get("diarization", {})
        if diar.get("num_speakers"):
            props["speakers"] = str(diar["num_speakers"])

        for key, value in props.items():
            if not value:
                continue
            try:
                subprocess.run(
                    [
                        "minotes", "property", "set",
                        "--page", page_title,
                        "--key", key,
                        "--value", value,
                    ],
                    capture_output=True, timeout=10,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    def _load_state(self) -> dict[str, Any]:
        """Load the sync state file."""
        state_file = self.sync_dir / ".audioscript_sync_state.json"
        if not state_file.exists():
            return {}
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _update_state(self, name: str, path: Path) -> None:
        """Update the sync state file atomically."""
        import os as _os
        import tempfile as _tf

        state = self._load_state()
        state[name] = {
            "path": str(path),
            "synced_at": time.time(),
        }
        state_file = self.sync_dir / ".audioscript_sync_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = _tf.mkstemp(dir=state_file.parent, prefix=".state_", suffix=".tmp")
        try:
            with _os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2)
            _os.replace(tmp_path, state_file)
        except BaseException:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
            raise
