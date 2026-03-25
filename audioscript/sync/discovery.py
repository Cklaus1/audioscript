"""File discovery, mtime caching, and OneDrive availability probing."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audioscript.utils.file_utils import ProcessingManifest, get_file_hash

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {
    "m4a", "mp3", "wav", "flac", "ogg", "opus", "webm", "mp4", "wma", "aac",
}


@dataclass
class FileEntry:
    """Represents a discovered audio file with cached metadata."""

    path: Path
    size: int
    mtime: float
    hash: str | None = None
    status: str = "new"  # "new" | "local" | "cloud" | "error" | "skip"


class FileDiscovery:
    """Scans directories for audio files with mtime-based caching."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        """Load the mtime/hash cache from disk."""
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load sync cache: %s", e)
            return {}

    def _save_cache(self) -> None:
        """Persist the cache to disk atomically."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.cache_path.parent, prefix=".sync_cache_", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._cache, f, indent=2)
            os.replace(tmp_path, self.cache_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def scan(
        self,
        source_dir: Path,
        extensions: set[str] | list[str] | None = None,
        recursive: bool = True,
        ignore_patterns: list[str] | None = None,
        min_file_size: int = 0,
        max_file_size: int | None = None,
        skip_older_than: int | None = None,
    ) -> list[FileEntry]:
        """Scan a directory for audio files matching the criteria.

        Returns FileEntry list sorted by mtime (newest first).
        """
        extensions = set(extensions) if extensions else AUDIO_EXTENSIONS
        ignore_patterns = ignore_patterns or []
        entries: list[FileEntry] = []
        now = time.time()

        if recursive:
            walker = source_dir.rglob("*")
        else:
            walker = source_dir.glob("*")

        for filepath in walker:
            if not filepath.is_file():
                continue

            # Extension check
            ext = filepath.suffix.lstrip(".").lower()
            if ext not in extensions:
                continue

            # Ignore patterns
            rel_path = str(filepath.relative_to(source_dir))
            if any(fnmatch.fnmatch(rel_path, pat) for pat in ignore_patterns):
                continue
            if any(fnmatch.fnmatch(filepath.name, pat) for pat in ignore_patterns):
                continue

            # Size checks
            try:
                stat = filepath.stat()
            except OSError:
                continue

            if stat.st_size < min_file_size:
                continue
            if max_file_size is not None and stat.st_size > max_file_size:
                continue

            # Age check
            if skip_older_than is not None:
                age_days = (now - stat.st_mtime) / 86400
                if age_days > skip_older_than:
                    continue

            entries.append(FileEntry(
                path=filepath,
                size=stat.st_size,
                mtime=stat.st_mtime,
            ))

        # Sort newest first
        entries.sort(key=lambda e: e.mtime, reverse=True)
        return entries

    def probe_availability(
        self, entries: list[FileEntry],
    ) -> tuple[list[FileEntry], list[FileEntry]]:
        """Test which files are locally available vs cloud-only (OneDrive).

        Tries reading first 1KB. Returns (local, cloud_only).
        """
        local: list[FileEntry] = []
        cloud: list[FileEntry] = []

        def _probe_one(entry: FileEntry) -> tuple[FileEntry, str]:
            try:
                with open(entry.path, "rb") as f:
                    f.read(1024)
                return entry, "local"
            except PermissionError:
                logger.warning("Permission denied: %s", entry.path)
                return entry, "error"
            except OSError:
                return entry, "cloud"

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=16) as pool:
            for entry, status in pool.map(_probe_one, entries):
                entry.status = status
                if status == "local":
                    local.append(entry)
                elif status == "cloud":
                    cloud.append(entry)

        return local, cloud

    def compute_hashes(self, entries: list[FileEntry]) -> list[FileEntry]:
        """Compute SHA-256 hashes, using cache when (size, mtime) unchanged.

        Updates the cache after computing.
        """
        for entry in entries:
            cache_key = str(entry.path)
            cached = self._cache.get(cache_key)

            if (
                cached
                and cached.get("size") == entry.size
                and cached.get("mtime") == entry.mtime
                and cached.get("hash")
            ):
                # Cache hit — reuse hash
                entry.hash = cached["hash"]
            else:
                # Cache miss — compute hash
                try:
                    entry.hash = get_file_hash(entry.path)
                    self._cache[cache_key] = {
                        "size": entry.size,
                        "mtime": entry.mtime,
                        "hash": entry.hash,
                    }
                except (OSError, FileNotFoundError) as e:
                    logger.warning("Failed to hash %s: %s", entry.path, e)
                    entry.status = "error"

        self._save_cache()
        return entries

    def diff_against_manifest(
        self,
        entries: list[FileEntry],
        manifest: ProcessingManifest,
        tier: str,
        version: str,
    ) -> list[FileEntry]:
        """Return only entries not already completed in the manifest."""
        new_entries: list[FileEntry] = []

        for entry in entries:
            if entry.hash is None or entry.status == "error":
                continue

            if manifest.is_processed(entry.hash, tier, version):
                entry.status = "skip"
            else:
                entry.status = "new"
                new_entries.append(entry)

        return new_entries
