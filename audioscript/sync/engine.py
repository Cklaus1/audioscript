"""Sync engine — scan, diff, download, transcribe, export."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from audioscript.config.settings import (
    AudioScriptConfig,
    SyncConfig,
    SyncSourceConfig,
    TranscriptionTier,
)
from audioscript.processors.audio_processor import AudioProcessor
from audioscript.sync.discovery import FileDiscovery, FileEntry
from audioscript.sync.wsl import is_wsl, resolve_sync_path
from audioscript.utils.file_utils import ProcessingManifest

logger = logging.getLogger(__name__)


@dataclass
class SourceReport:
    """Report for a single source directory sync."""

    source: str
    scanned: int = 0
    local: int = 0
    cloud_only: int = 0
    downloaded: int = 0
    new: int = 0
    skipped: int = 0
    transcribed: int = 0
    failed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    speaker_summary: dict[str, Any] | None = None


@dataclass
class SyncReport:
    """Report for a full sync cycle."""

    sources: list[SourceReport] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total_scanned(self) -> int:
        return sum(s.scanned for s in self.sources)

    @property
    def total_new(self) -> int:
        return sum(s.new for s in self.sources)

    @property
    def total_transcribed(self) -> int:
        return sum(s.transcribed for s in self.sources)

    @property
    def total_failed(self) -> int:
        return sum(s.failed for s in self.sources)

    @property
    def summary(self) -> str:
        if self.total_new == 0:
            return f"Scanned {self.total_scanned} files, 0 new. Up to date."
        return (
            f"Scanned {self.total_scanned}, "
            f"{self.total_transcribed} transcribed, "
            f"{self.total_failed} failed "
            f"({self.elapsed_seconds:.1f}s)"
        )


class SyncEngine:
    """Orchestrates directory sync: scan → probe → download → transcribe → export."""

    def __init__(
        self,
        sync_config: SyncConfig,
        global_config: AudioScriptConfig,
        console: Console | None = None,
    ) -> None:
        self.sync_config = sync_config
        self.global_config = global_config
        self.console = console or Console()

    def run_once(
        self,
        force: bool = False,
        no_download: bool = False,
        download_only: bool = False,
    ) -> SyncReport:
        """Run a single sync cycle across all configured sources."""
        start = time.time()
        report = SyncReport()

        output_dir = Path(self.sync_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_path = output_dir / ".audioscript_sync_cache.json"
        manifest_path = output_dir / "manifest.json"
        manifest = ProcessingManifest(manifest_path)
        discovery = FileDiscovery(cache_path)

        for source in self.sync_config.sources:
            src_report = self._sync_source(
                source, discovery, manifest, output_dir,
                force=force, no_download=no_download,
                download_only=download_only,
            )
            report.sources.append(src_report)

        report.elapsed_seconds = time.time() - start
        return report

    def run_watch(self, poll_interval: int | None = None, **kwargs: Any) -> None:
        """Continuous polling loop. Ctrl+C to stop."""
        interval = poll_interval or self.sync_config.poll_interval
        cycle = 0

        self.console.print(
            f"[bold green]Watch mode[/] — polling every {interval}s. Ctrl+C to stop."
        )

        try:
            while True:
                cycle += 1
                timestamp = time.strftime("%H:%M:%S")
                self.console.print(f"\n[dim][{timestamp}] Scan #{cycle}[/]")

                report = self.run_once(**kwargs)
                self.console.print(f"[{timestamp}] {report.summary}")

                time.sleep(interval)
        except KeyboardInterrupt:
            self.console.print("\n[bold]Watch mode stopped.[/]")

    def _sync_source(
        self,
        source: SyncSourceConfig,
        discovery: FileDiscovery,
        manifest: ProcessingManifest,
        output_dir: Path,
        force: bool = False,
        no_download: bool = False,
        download_only: bool = False,
    ) -> SourceReport:
        """Sync a single source directory."""
        report = SourceReport(source=source.path)

        # Resolve path (WSL translation)
        try:
            resolved = resolve_sync_path(source.path)
        except (FileNotFoundError, NotADirectoryError) as e:
            self.console.print(f"[red]Error:[/] {e}")
            return report

        self.console.print(f"Scanning: {resolved}")

        # Scan
        entries = discovery.scan(
            resolved,
            extensions=set(self.sync_config.extensions),
            recursive=self.sync_config.recursive,
            ignore_patterns=self.sync_config.ignore_patterns,
            min_file_size=self.sync_config.min_file_size,
            max_file_size=self.sync_config.max_file_size,
            skip_older_than=self.sync_config.skip_older_than,
        )
        report.scanned = len(entries)

        if not entries:
            self.console.print("  No audio files found.")
            return report

        # Probe availability (OneDrive detection)
        local, cloud = discovery.probe_availability(entries)
        report.local = len(local)
        report.cloud_only = len(cloud)

        if cloud:
            self.console.print(
                f"  {len(local)} local, {len(cloud)} cloud-only (OneDrive)"
            )

            if not no_download and self.sync_config.onedrive.auto_download:
                downloaded = self._handle_onedrive(cloud, discovery)
                report.downloaded = len(downloaded)
                local.extend(downloaded)

        if download_only:
            self.console.print(f"  Download-only mode: {report.downloaded} triggered.")
            return report

        # Hash + diff
        local = discovery.compute_hashes(local)
        settings = self._merge_source_settings(source)
        tier = settings.tier.value
        version = settings.version

        if force:
            to_process = [e for e in local if e.hash and e.status != "error"]
        else:
            to_process = discovery.diff_against_manifest(local, manifest, tier, version)

        report.new = len(to_process)
        report.skipped = len(local) - len(to_process)

        if not to_process:
            self.console.print(
                f"  {report.scanned} scanned, 0 new, {report.skipped} skipped. Up to date."
            )
            return report

        self.console.print(
            f"  {report.new} new files to transcribe"
        )

        # Apply batch size
        batch_size = self.sync_config.batch_size
        if batch_size > 0:
            to_process = to_process[:batch_size]

        # Stage to local filesystem if needed
        staged_dir = None
        if self._is_slow_filesystem(resolved):
            staged_dir, to_process = self._stage_files(to_process)

        # Transcribe
        processor = AudioProcessor(settings, manifest, self.console)

        for i, entry in enumerate(to_process):
            success = processor.process_file(entry.path)
            result = {
                "file": str(entry.path.name),
                "status": "completed" if success else "failed",
            }
            report.results.append(result)

            if success:
                report.transcribed += 1
            else:
                report.failed += 1

            # Delay between files
            if self.sync_config.delay_between > 0 and i < len(to_process) - 1:
                time.sleep(self.sync_config.delay_between)

        # Clean up staging
        if staged_dir and self.sync_config.onedrive.cleanup_staging:
            shutil.rmtree(staged_dir, ignore_errors=True)

        self.console.print(
            f"  Done: {report.transcribed} transcribed, {report.failed} failed"
        )

        # Speaker summary (if diarization was used)
        try:
            from audioscript.speakers.identity_db import SpeakerIdentityDB
            from audioscript.speakers.reporter import UnknownSpeakerReporter

            id_db_path = Path(output_dir) / "speaker_identities.json"
            if id_db_path.exists():
                id_db = SpeakerIdentityDB(id_db_path)
                reporter = UnknownSpeakerReporter(id_db)
                summary = reporter.generate_summary()
                report.speaker_summary = summary
                self.console.print(
                    f"  Speakers: {summary['total_clusters']} total, "
                    f"{summary['confirmed']} confirmed, {summary['unknown']} unknown"
                )
        except Exception:
            pass

        return report

    def _handle_onedrive(
        self,
        cloud_files: list[FileEntry],
        discovery: FileDiscovery,
    ) -> list[FileEntry]:
        """Trigger OneDrive downloads and wait for files to become available."""
        self.console.print(
            f"  Triggering download for {len(cloud_files)} cloud-only files..."
        )

        # Trigger downloads via attrib.exe
        for entry in cloud_files:
            self._trigger_download(entry.path)

        # Wait for downloads
        timeout = self.sync_config.onedrive.download_timeout
        poll = self.sync_config.onedrive.download_poll_interval
        deadline = time.time() + timeout
        pending = list(cloud_files)
        downloaded: list[FileEntry] = []

        while pending and time.time() < deadline:
            still_pending: list[FileEntry] = []
            for entry in pending:
                try:
                    with open(entry.path, "rb") as f:
                        f.read(1024)
                    entry.status = "local"
                    downloaded.append(entry)
                except OSError:
                    still_pending.append(entry)

            pending = still_pending
            if pending:
                elapsed = int(time.time() - (deadline - timeout))
                self.console.print(
                    f"  [{elapsed}s] {len(downloaded)}/{len(cloud_files)} downloaded..."
                )
                time.sleep(poll)

        if pending:
            self.console.print(
                f"  [yellow]Warning:[/] {len(pending)} files still downloading "
                f"after {timeout}s timeout. Will process on next sync."
            )

        return downloaded

    def _trigger_download(self, path: Path) -> None:
        """Trigger OneDrive download for a single file via attrib.exe."""
        if not is_wsl():
            return

        try:
            # Get Windows path
            result = subprocess.run(
                ["wslpath", "-w", str(path)],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return

            win_path = result.stdout.strip()

            # attrib.exe -U +P removes Unpinned, adds Pinned → triggers download
            subprocess.run(
                ["attrib.exe", "-U", "+P", win_path],
                capture_output=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug("Failed to trigger download for %s: %s", path, e)

    def _is_slow_filesystem(self, path: Path) -> bool:
        """Check if path is on a slow WSL→Windows mount."""
        return is_wsl() and str(path).startswith("/mnt/")

    def _stage_files(
        self, entries: list[FileEntry],
    ) -> tuple[Path, list[FileEntry]]:
        """Copy files to local Linux filesystem for faster I/O.

        Returns (staging_dir, updated entries with local paths).
        """
        staging_dir = Path(
            self.sync_config.onedrive.staging_dir
            or tempfile.mkdtemp(prefix="audioscript-stage-")
        )
        staging_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"  Staging {len(entries)} files to {staging_dir}...")
        staged: list[FileEntry] = []

        for entry in entries:
            dest = staging_dir / entry.path.name
            try:
                shutil.copy2(entry.path, dest)
                staged.append(FileEntry(
                    path=dest,
                    size=entry.size,
                    mtime=entry.mtime,
                    hash=entry.hash,
                    status=entry.status,
                ))
            except (OSError, PermissionError) as e:
                logger.warning("Failed to stage %s: %s", entry.path, e)

        return staging_dir, staged

    def _merge_source_settings(
        self, source: SyncSourceConfig,
    ) -> AudioScriptConfig:
        """Merge per-source overrides with global config for transcription."""
        overrides: dict[str, Any] = {
            "output_dir": self.sync_config.output_dir,
            "output_format": self.sync_config.output_format,
        }

        # Apply source-level overrides
        if source.tier is not None:
            overrides["tier"] = source.tier
        if source.model is not None:
            overrides["model"] = source.model
        if source.diarize is not None:
            overrides["diarize"] = source.diarize
        if source.export is not None:
            overrides["export"] = source.export
        if source.output_format is not None:
            overrides["output_format"] = source.output_format
        if source.summarize is not None:
            overrides["summarize"] = source.summarize

        # MiNotes from sync config
        if self.sync_config.minotes.enabled:
            overrides["export"] = "minotes"

        # Sync defaults: always extract metadata and generate summary
        base = self.global_config.model_dump()
        base["metadata"] = True
        base["summarize"] = True
        base.update(overrides)
        return AudioScriptConfig(**base)
