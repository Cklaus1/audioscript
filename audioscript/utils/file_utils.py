"""File utility functions for AudioScript."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Buffer size for reading files in chunks (256 KB — reduces syscalls 30x on large audio)
_HASH_BUF_SIZE = 262144


def get_file_hash(file_path: Path) -> str:
    """Calculate a SHA-256 content hash for the file.

    Uses the actual file content so that renames/moves don't
    invalidate the hash and identical content is always detected.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(_HASH_BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def get_output_path(input_file: Path, output_dir: Path, ext: str = "json") -> Path:
    """Generate an output path for a processed file.

    Args:
        input_file: Path to the input file
        output_dir: Directory for output files
        ext: Output file extension (default: json)

    Returns:
        Path to the output file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{input_file.stem}.{ext}"
    return output_dir / output_filename


class ProcessingManifest:
    """Manages the tracking of processed files and their status.

    Writes are atomic (write to temp file, then os.replace) to prevent
    corruption from crashes or concurrent writers.
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.data = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        """Load the manifest file or create a new one if it doesn't exist."""
        if not self.manifest_path.exists():
            return {"version": "1.1", "files": {}}

        try:
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("Corrupt manifest file %s: %s", self.manifest_path, e)
            return {"version": "1.1", "files": {}}
        except OSError as e:
            logger.warning("Failed to read manifest %s: %s", self.manifest_path, e)
            return {"version": "1.1", "files": {}}

    def save(self) -> None:
        """Save the manifest atomically with file locking.

        Uses fcntl.flock to prevent concurrent writers from clobbering
        each other's updates. Writes to temp file then renames.
        """
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        lock_path = self.manifest_path.with_suffix(".lock")
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Reload from disk to get any concurrent updates before merging
            if self.manifest_path.exists():
                try:
                    with open(self.manifest_path, "r") as f:
                        disk_data = json.load(f)
                    # Merge: our in-memory changes take precedence
                    for file_hash, file_data in self.data.get("files", {}).items():
                        disk_data.setdefault("files", {})[file_hash] = file_data
                    self.data = disk_data
                except (json.JSONDecodeError, OSError):
                    pass  # Corrupt file — our in-memory data wins

            fd, tmp_path = tempfile.mkstemp(
                dir=self.manifest_path.parent,
                prefix=".manifest_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self.data, f, indent=2)
                os.replace(tmp_path, self.manifest_path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def is_processed(self, file_hash: str, tier: str, version: str) -> bool:
        """Check if a file has been processed at the given tier and version."""
        if file_hash not in self.data["files"]:
            return False

        file_data = self.data["files"][file_hash]
        if file_data.get("status") != "completed":
            return False

        return (
            file_data.get("tier") == tier
            and file_data.get("version") == version
        )

    def update_file_status(
        self,
        file_hash: str,
        status: str,
        tier: str,
        version: str,
        checkpoint: str | None = None,
        error: str | None = None,
        *,
        backend: str | None = None,
        confidence: float | None = None,
        hallucination_flags: int | None = None,
        error_category: str | None = None,
        filename: str | None = None,
        duration_seconds: float | None = None,
        word_count: int | None = None,
        language: str | None = None,
        flush: bool = True,
    ) -> None:
        """Update the status of a file in the manifest.

        Set flush=False for intermediate updates (processing, transcribed)
        to avoid unnecessary disk writes. Call with flush=True (default)
        for final status changes (completed, error).
        """
        if file_hash not in self.data["files"]:
            self.data["files"][file_hash] = {}

        self.data["files"][file_hash].update({
            "status": status,
            "tier": tier,
            "version": version,
            "last_updated": time.time(),
        })

        if checkpoint is not None:
            self.data["files"][file_hash]["checkpoint"] = checkpoint

        if error is not None:
            self.data["files"][file_hash]["error"] = error

        if backend is not None:
            self.data["files"][file_hash]["backend"] = backend

        if confidence is not None:
            self.data["files"][file_hash]["confidence"] = confidence

        if hallucination_flags is not None:
            self.data["files"][file_hash]["hallucination_flags"] = hallucination_flags

        if error_category is not None:
            self.data["files"][file_hash]["error_category"] = error_category

        if filename is not None:
            self.data["files"][file_hash]["filename"] = filename

        if duration_seconds is not None:
            self.data["files"][file_hash]["duration_seconds"] = duration_seconds

        if word_count is not None:
            self.data["files"][file_hash]["word_count"] = word_count

        if language is not None:
            self.data["files"][file_hash]["language"] = language

        if flush:
            self.save()

    def get_checkpoint(self, file_hash: str) -> str | None:
        """Get the checkpoint information for a file."""
        if file_hash not in self.data["files"]:
            return None
        return self.data["files"][file_hash].get("checkpoint")

    def get_status(self, file_hash: str) -> str | None:
        """Get the processing status of a file."""
        if file_hash not in self.data["files"]:
            return None
        return self.data["files"][file_hash].get("status")
