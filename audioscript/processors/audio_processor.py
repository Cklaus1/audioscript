"""Audio processing and transcription functionality."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from audioscript.config.settings import AudioScriptConfig
from audioscript.processors import create_transcriber
from audioscript.processors.backend_protocol import TranscriberBackend, TranscriptionResult
from audioscript.utils.file_utils import ProcessingManifest, get_file_hash, get_output_path
from audioscript.utils.metadata import extract_metadata

logger = logging.getLogger(__name__)


def _save_results(data: dict[str, Any], output_path: Path) -> None:
    """Save transcription results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _create_checkpoint(result: TranscriptionResult | dict[str, Any]) -> str:
    """Create a context-priming checkpoint from transcription."""
    if isinstance(result, TranscriptionResult):
        return json.dumps({"text": result.text})
    return json.dumps({"text": result.get("text", "")})


def _generate_summary(result_dict: dict[str, Any]) -> str:
    """Generate a basic extractive summary (first 25 words).

    For production use, replace with an LLM-based summarizer.
    """
    text = result_dict.get("text", "")
    words = text.split()
    if len(words) <= 25:
        return text
    return " ".join(words[:25]) + "..."


class AudioProcessor:
    """Handles audio processing and transcription."""

    def __init__(
        self,
        settings: AudioScriptConfig,
        manifest: ProcessingManifest,
        console: Console | None = None,
    ) -> None:
        self.settings = settings
        self.manifest = manifest
        self.console = console or Console()
        self._transcriber: TranscriberBackend | None = None
        self._diarizer: Any = None
        self._speaker_db: Any = None
        self._identity_db: Any = None

    def _get_transcriber(self) -> TranscriberBackend:
        """Get or lazily initialize the transcription backend."""
        if self._transcriber is None:
            self._transcriber = create_transcriber(self.settings)
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

    def _get_speaker_db(self) -> Any:
        """Get or lazily initialize the legacy speaker database."""
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

    def _get_identity_db(self) -> Any:
        """Get or lazily initialize the speaker identity database."""
        if self._identity_db is None:
            from audioscript.speakers.identity_db import SpeakerIdentityDB

            # Use explicit path or default to output_dir
            db_path = getattr(self.settings, "speaker_identity_db", None)
            if db_path:
                db_path = Path(db_path)
            else:
                db_path = Path(self.settings.output_dir) / "speaker_identities.json"

            self._identity_db = SpeakerIdentityDB(db_path)
            self.console.print(
                f"Speaker identity DB: {db_path} "
                f"({self._identity_db.cluster_count} clusters, "
                f"{self._identity_db.confirmed_count} confirmed)"
            )
        return self._identity_db

    def process_file(self, file_path: Path) -> bool:
        """Process a single audio file.

        Returns True if processing succeeded, False otherwise.
        """
        from audioscript.utils.error_classification import classify_error, should_retry

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
                    try:
                        from audioscript.processors.audio_cleaner import (
                            clean_audio as do_clean,
                        )
                        cleaned_path = output_dir / "cleaned" / file_path.name
                        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
                        audio_to_process, clean_stats = do_clean(
                            file_path, cleaned_path,
                            level=self.settings.clean_level,
                        )
                        if clean_stats["skipped"]:
                            self.console.print(
                                f"Audio clean skipped (SNR {clean_stats['snr_before']} dB): {file_path.name}"
                            )
                        else:
                            self.console.print(
                                f"Cleaned audio: SNR {clean_stats['snr_before']} → {clean_stats['snr_after']} dB"
                            )
                        audio_to_process = Path(audio_to_process)
                    except ImportError:
                        self.console.print(
                            "[yellow]Warning:[/] noisereduce not installed. "
                            "Skipping audio cleaning."
                        )

                # Run standalone VAD if requested (independent of --diarize)
                # Uses faster-whisper's built-in Silero VAD
                use_builtin_vad = self.settings.vad

                # Diarization requires word_timestamps
                word_timestamps = self.settings.word_timestamps
                if self.settings.diarize and not word_timestamps:
                    word_timestamps = True
                    self.console.print(
                        "[dim]Enabling word_timestamps (required for diarization)[/]"
                    )

                self.console.print(
                    f"Transcribing: {file_path.name} "
                    f"(tier={self.settings.tier.value}, backend={transcriber.backend_name}, "
                    f"attempt {attempt}/{max_attempts})"
                )

                # Build transcription kwargs
                transcribe_kwargs: dict[str, Any] = {
                    "language": self.settings.language,
                    "temperature": self.settings.parse_temperature(),
                    "word_timestamps": word_timestamps,
                    "beam_size": self.settings.beam_size,
                    "best_of": self.settings.best_of,
                    "condition_on_previous_text": self.settings.condition_on_previous_text,
                    "checkpoint": checkpoint,
                    "suppress_tokens": self.settings.suppress_tokens,
                    "suppress_blank": self.settings.suppress_blank,
                    "clip_timestamps": self.settings.parse_clip_timestamps(),
                    "vad_filter": use_builtin_vad,
                }

                if self.settings.hallucination_silence_threshold is not None:
                    transcribe_kwargs["hallucination_silence_threshold"] = (
                        self.settings.hallucination_silence_threshold
                    )
                if self.settings.patience is not None:
                    transcribe_kwargs["patience"] = self.settings.patience
                if self.settings.length_penalty is not None:
                    transcribe_kwargs["length_penalty"] = self.settings.length_penalty

                result = transcriber.transcribe(audio_to_process, **transcribe_kwargs)

                # Save checkpoint immediately after transcription
                new_checkpoint = _create_checkpoint(result)
                checkpoint = new_checkpoint
                self.manifest.update_file_status(
                    file_hash, "transcribed",
                    self.settings.tier.value, self.settings.version, new_checkpoint,
                )

                # Convert to dict for JSON output and downstream processing
                result_dict = result.to_dict()

                # Hallucination detection
                avg_confidence = None
                hallucination_flag_count = None
                if self.settings.hallucination_filter != "off" and result.segments:
                    try:
                        from audioscript.processors.hallucination_detector import (
                            analyze,
                            apply_filter,
                        )
                        reports = analyze(
                            result.segments,
                            audio_path=str(audio_to_process),
                            min_confidence=self.settings.min_confidence or 0.4,
                        )
                        result.segments = apply_filter(
                            result.segments, reports,
                            mode=self.settings.hallucination_filter,
                        )
                        # Compute aggregate metrics
                        confidences = [r.confidence for r in reports if r.confidence is not None]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else None
                        hallucination_flag_count = sum(1 for r in reports if r.risk in ("medium", "high"))
                        # Rebuild dict after filtering
                        result_dict = result.to_dict()
                    except ImportError:
                        pass

                # Embed audio file metadata if requested
                if self.settings.metadata:
                    try:
                        result_dict["metadata"] = extract_metadata(file_path)
                    except Exception as meta_err:
                        logger.warning("Metadata extraction failed: %s", meta_err)

                # Save JSON before diarization
                _save_results(result_dict, transcription_path)

                # Run speaker diarization if requested
                if self.settings.diarize:
                    try:
                        result_dict = self._run_diarization(
                            result_dict, audio_to_process, file_path, output_dir,
                        )
                        # Re-save JSON with diarization data
                        _save_results(result_dict, transcription_path)
                    except Exception as diar_err:
                        self.console.print(
                            f"[yellow]Warning:[/] Diarization failed for {file_path.name}: {diar_err}. "
                            "Transcription saved without speaker labels."
                        )
                        logger.warning("Diarization failed: %s", diar_err, exc_info=True)

                # LLM analysis (if API key available) — runs BEFORE markdown generation
                llm_analysis = None
                try:
                    from audioscript.llm.analyzer import analyze_transcript, apply_llm_results
                    from audioscript.llm.cost_tracker import CostTracker
                    import os

                    if os.environ.get("ANTHROPIC_API_KEY"):
                        self.console.print(f"LLM analysis: {file_path.name}")
                        cost_log = output_dir / ".audioscript_llm_costs.jsonl"
                        tracker = CostTracker(cost_log)

                        llm_analysis = analyze_transcript(
                            transcript_text=result_dict.get("text", ""),
                            segments=result_dict.get("segments"),
                            metadata=result_dict.get("metadata"),
                            call_id=file_hash,
                            cost_tracker=tracker,
                        )

                        if llm_analysis:
                            result_dict = apply_llm_results(result_dict, llm_analysis)
                            self.console.print(
                                f"  Title: {llm_analysis.get('title', '?')}"
                            )
                            self.console.print(
                                f"  Classification: {llm_analysis.get('classification', '?')}"
                            )
                            speakers_found = llm_analysis.get("speakers", [])
                            for s in speakers_found:
                                if s.get("likely_name"):
                                    self.console.print(
                                        f"  Speaker: {s['label']} → {s['likely_name']} ({s.get('evidence', '')})"
                                    )

                            # Update speaker identity DB with LLM name hints
                            self._apply_llm_speaker_hints(llm_analysis, result_dict)

                            usage = tracker.session_summary()
                            self.console.print(
                                f"  Cost: ${usage['total_cost_usd']:.4f} "
                                f"({usage['total_input_tokens']}+{usage['total_output_tokens']} tokens)"
                            )

                            # Re-save JSON with LLM results
                            _save_results(result_dict, transcription_path)
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning("LLM analysis failed: %s", e)

                # Generate summary
                summary_text = None
                if llm_analysis and llm_analysis.get("summary"):
                    summary_text = llm_analysis["summary"]
                elif self.settings.summarize:
                    summary_text = _generate_summary(result_dict)

                if summary_text:
                    self.console.print(f"Generating summary: {file_path.name}")
                    summary_path = get_output_path(file_path, output_dir, "summary.txt")
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(summary_text)

                # Re-save JSON with all enrichment (LLM, hallucination, metadata)
                _save_results(result_dict, transcription_path)

                # Save markdown AFTER all enrichment (LLM title, summary, actions, topics)
                fmt = self.settings.output_format
                if fmt == "markdown":
                    self._save_markdown(result_dict, file_path, output_dir, summary=summary_text)

                # Export to MiNotes if requested
                if self.settings.export == "minotes":
                    self._export_minotes(result_dict, file_path, summary_text)

                # Mark as completed
                self.manifest.update_file_status(
                    file_hash, "completed",
                    self.settings.tier.value, self.settings.version, new_checkpoint,
                    backend=transcriber.backend_name,
                    confidence=avg_confidence,
                    hallucination_flags=hallucination_flag_count,
                )
                return True

            except Exception as e:
                error_category = classify_error(e).value
                self.manifest.update_file_status(
                    file_hash, "error",
                    self.settings.tier.value, self.settings.version,
                    checkpoint, str(e),
                    error_category=error_category,
                )

                if not should_retry(e, self.settings.retry_strategy, attempt, max_attempts):
                    self.console.print(f"[red]Failed[/] {file_path.name}: {e}")
                    return False

                self.console.print(
                    f"[yellow]Error[/] processing {file_path.name}: {e}. "
                    f"Retrying ({attempt}/{self.settings.max_retries})..."
                )
                time.sleep(min(2 ** (attempt - 1), 10))

        return False

    def _save_markdown(
        self, result_dict: dict[str, Any], file_path: Path, output_dir: Path,
        summary: str | None = None,
    ) -> None:
        """Save transcription as Obsidian-compatible markdown."""
        try:
            from audioscript.formatters.markdown_formatter import render_markdown

            md_content = render_markdown(
                result_dict, file_path,
                metadata=result_dict.get("metadata"),
                summary=summary,
            )
            md_path = get_output_path(file_path, output_dir, "md")
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md_content, encoding="utf-8")
        except ImportError:
            logger.warning("Markdown formatter not available")

    def _apply_llm_speaker_hints(
        self,
        llm_analysis: dict[str, Any],
        result_dict: dict[str, Any],
    ) -> None:
        """Apply LLM speaker name hints to the identity DB as evidence."""
        try:
            identity_db = self._get_identity_db()
            speakers = llm_analysis.get("speakers", [])

            for s in speakers:
                label = s.get("label", "")
                name = s.get("likely_name")
                evidence_text = s.get("evidence", "")

                if not name or not label:
                    continue

                # Find the cluster ID for this label in the transcript
                diar = result_dict.get("diarization", {})
                for resolved in diar.get("speakers_resolved", []):
                    if resolved.get("local_label") == label or resolved.get("speaker_cluster_id") == label:
                        cluster_id = resolved["speaker_cluster_id"]

                        # Add as candidate evidence (don't auto-confirm from LLM alone)
                        from audioscript.speakers.models import SpeakerEvidence, generate_id, now_iso
                        identity_db.add_evidence(SpeakerEvidence(
                            evidence_id=generate_id("ev_"),
                            speaker_cluster_id=cluster_id,
                            type="llm_inference",
                            score=0.6,
                            summary=f"LLM identified as '{name}': {evidence_text}",
                            details={"name": name, "evidence": evidence_text, "role": s.get("role", "")},
                            created_at=now_iso(),
                        ))

                        # Upgrade to candidate if currently unknown
                        identity = identity_db.data["identities"].get(cluster_id, {})
                        if identity.get("status") == "unknown":
                            identity["status"] = "candidate"
                            identity["updated_at"] = now_iso()

                        identity_db.save()
                        break
        except Exception as e:
            logger.debug("Failed to apply LLM speaker hints: %s", e)

    def _export_minotes(
        self,
        result_dict: dict[str, Any],
        file_path: Path,
        summary: str | None,
    ) -> None:
        """Export transcription to MiNotes."""
        try:
            from audioscript.exporters.minotes_exporter import MiNotesExporter
            from audioscript.formatters.markdown_formatter import render_markdown

            exporter = MiNotesExporter(sync_dir=self.settings.minotes_sync_dir)
            if not exporter.is_already_exported(file_path) or self.settings.force:
                exporter.ensure_registered()
                exporter.ensure_transcript_class()
                md_content = render_markdown(
                    result_dict, file_path,
                    metadata=result_dict.get("metadata"),
                    summary=summary,
                )
                export_path = exporter.export(md_content, file_path, result_dict)
                self.console.print(f"Exported to MiNotes: {export_path}")

                if summary:
                    exporter.journal_entry(file_path, summary)
        except ImportError:
            logger.warning("MiNotes exporter not available")
        except Exception as e:
            self.console.print(f"[yellow]Warning:[/] MiNotes export failed: {e}")
            logger.warning("MiNotes export failed: %s", e, exc_info=True)

    def _run_diarization(
        self,
        result: dict[str, Any],
        audio_path: Path,
        file_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Run diarization pipeline and merge with transcription result."""
        self.console.print(f"Diarizing: {file_path.name}")
        diarizer = self._get_diarizer()

        def diar_progress(step: str, completed: int | None, total: int | None) -> None:
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

        # Speaker identity resolution (new system)
        try:
            from audioscript.speakers.resolution import SpeakerResolutionEngine

            identity_db = self._get_identity_db()
            call_id = get_file_hash(file_path) if file_path.exists() else file_path.stem
            engine = SpeakerResolutionEngine(
                identity_db,
                match_threshold=self.settings.speaker_similarity_threshold,
            )
            # Calendar joiner (Stage E) — optional, requires ms365-cli
            calendar_joiner = None
            try:
                from audioscript.speakers.calendar import CalendarJoiner
                cj = CalendarJoiner()
                if cj.is_available():
                    calendar_joiner = cj
            except Exception:
                pass

            resolutions = engine.resolve_call(
                diar_result, call_id, file_path,
                call_metadata=result.get("metadata") if isinstance(result, dict) else None,
                calendar_joiner=calendar_joiner,
                transcript_segments=result.get("segments") if isinstance(result, dict) else None,
            )
            result = engine.apply_to_transcript(result, resolutions)

            # Report resolution results
            new_count = sum(1 for r in resolutions if r.is_new_cluster)
            known_count = len(resolutions) - new_count
            self.console.print(
                f"Found [bold]{diar_result['num_speakers']}[/] speakers "
                f"({known_count} linked, {new_count} new)"
            )
            for r in resolutions:
                name = r.display_name or r.speaker_cluster_id
                self.console.print(
                    f"  {r.local_label} → {name} "
                    f"({r.status}, {r.confidence:.0%})"
                )
        except Exception as res_err:
            logger.warning("Speaker resolution failed, falling back to basic assignment: %s", res_err)
            # Fallback to legacy speaker DB matching
            speaker_db = self._get_speaker_db()
            result = diarizer.assign_speakers(
                result, diar_result,
                speaker_db=speaker_db,
                similarity_threshold=self.settings.speaker_similarity_threshold,
                allow_overlap=self.settings.allow_overlap,
            )
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

        # Evaluate with error handling
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
