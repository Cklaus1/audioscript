"""Speaker resolution engine — resolves diarized voices to stable cluster identities.

Pipeline stages (Phase 1):
  A: Diarization (already done by pyannote)
  B: Embedding extraction (already done by pyannote)
  C: DB match — cosine similarity against existing clusters
  D: Cross-call stitch — link unknown voices across calls

Phase 2 (future): Calendar join, transcript inference
Phase 3 (future): LLM-based identity reasoning (DeepScript)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audioscript.speakers.identity_db import SpeakerIdentityDB, _cosine_similarity
from audioscript.speakers.models import (
    SpeakerEvidence,
    SpeakerOccurrence,
    generate_id,
    now_iso,
)

logger = logging.getLogger(__name__)


# Confidence bands (PRD §6)
CONFIDENCE_AUTO_CONFIRM = 0.92
CONFIDENCE_PROBABLE = 0.80
CONFIDENCE_CANDIDATE = 0.60


@dataclass
class ResolutionResult:
    """Result of resolving one diarized speaker to a cluster identity."""

    speaker_cluster_id: str
    local_label: str  # SPEAKER_00
    display_name: str | None = None
    status: str = "unknown"
    confidence: float = 0.0
    source: str = "auto_cluster"
    is_new_cluster: bool = False
    candidate_names: list[dict[str, Any]] = field(default_factory=list)


class SpeakerResolutionEngine:
    """Resolves diarized speakers to stable cluster identities.

    Guiding principle: Link first. Name second. Confirm last.

    Resolution order (trust ladder):
    1. Match known confirmed identity (highest trust)
    2. Match known unknown cluster (preserves cross-call linkage)
    3. Create new unknown cluster (for truly new voices)
    """

    def __init__(
        self,
        identity_db: SpeakerIdentityDB,
        match_threshold: float = 0.70,
        auto_confirm_threshold: float = CONFIDENCE_AUTO_CONFIRM,
    ) -> None:
        self.db = identity_db
        self.match_threshold = match_threshold
        self.auto_confirm_threshold = auto_confirm_threshold

    def resolve_call(
        self,
        diarization_result: dict[str, Any],
        call_id: str,
        file_path: Path | None = None,
        call_metadata: dict[str, Any] | None = None,
        calendar_joiner: Any | None = None,
        transcript_segments: list[dict[str, Any]] | None = None,
    ) -> list[ResolutionResult]:
        """Run full resolution pipeline for all speakers in one call.

        Args:
            diarization_result: Output from SpeakerDiarizer.diarize() containing
                'speakers', 'segments', and 'speaker_embeddings'.
            call_id: Unique identifier for this call (usually file hash).
            file_path: Source audio file path (for logging).
            call_metadata: Audio metadata dict (for calendar matching by timestamp).
            calendar_joiner: CalendarJoiner instance (for Stage E).
            transcript_segments: Transcript segments (for Stage F name hints).

        Returns:
            List of ResolutionResult, one per diarized speaker.
        """
        embeddings = diarization_result.get("speaker_embeddings", {})
        segments = diarization_result.get("segments", [])
        now = now_iso()

        # Compute per-speaker speaking time
        speaking_times: dict[str, float] = {}
        for seg in segments:
            spk = seg.get("speaker", "")
            duration = seg.get("end", 0) - seg.get("start", 0)
            speaking_times[spk] = speaking_times.get(spk, 0) + duration

        results: list[ResolutionResult] = []
        used_clusters: set[str] = set()  # Prevent duplicate assignments

        # Sort speakers by speaking time (most speaking first gets best match)
        sorted_speakers = sorted(
            embeddings.keys(),
            key=lambda s: speaking_times.get(s, 0),
            reverse=True,
        )

        for local_label in sorted_speakers:
            embedding = embeddings[local_label]

            # Convert numpy to list if needed
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            speaking_secs = speaking_times.get(local_label, 0)

            # Stage C: DB match
            result = self._stage_c_db_match(
                embedding, local_label, call_id, speaking_secs, used_clusters,
            )

            if result is None:
                # Stage D: Create new cluster (cross-call stitching happens
                # naturally — next call will match this cluster)
                result = self._stage_d_new_cluster(
                    embedding, local_label, call_id, speaking_secs,
                )

            used_clusters.add(result.speaker_cluster_id)

            # Record occurrence
            self.db.add_occurrence(SpeakerOccurrence(
                occurrence_id=generate_id("occ_"),
                call_id=call_id,
                speaker_cluster_id=result.speaker_cluster_id,
                local_label=local_label,
                display_name=result.display_name,
                identity_status=result.status,
                resolution_source=result.source,
                resolution_confidence=result.confidence,
                total_speaking_seconds=speaking_secs,
            ))

            results.append(result)

        # Stage E: Calendar join — generate candidates from attendees
        if calendar_joiner and call_metadata:
            self._stage_e_calendar(results, call_metadata, calendar_joiner)

        # Stage F: Transcript name hints — regex-based name extraction
        if transcript_segments:
            self._stage_f_transcript_hints(results, transcript_segments)

        # Stage G: Aggregate confidence (re-score with all evidence)
        self._stage_g_aggregate_confidence(results)

        # Update co-speaker lists
        self._update_co_speakers(results)

        # Save all changes
        self.db.save()

        logger.info(
            "Resolved %d speakers in %s: %d existing, %d new",
            len(results), call_id,
            sum(1 for r in results if not r.is_new_cluster),
            sum(1 for r in results if r.is_new_cluster),
        )

        return results

    def _stage_c_db_match(
        self,
        embedding: list[float],
        local_label: str,
        call_id: str,
        speaking_secs: float,
        used_clusters: set[str],
    ) -> ResolutionResult | None:
        """Stage C: Match against existing clusters in the DB.

        Returns ResolutionResult if a match is found, None otherwise.
        """
        best_id = None
        best_score = 0.0

        for cluster_id, identity_data in self.db.data["identities"].items():
            if cluster_id in used_clusters:
                continue  # Already assigned in this call

            centroid = identity_data.get("embedding_centroid", [])
            if not centroid:
                continue

            score = _cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_id = cluster_id

        if best_id is None or best_score < self.match_threshold:
            return None

        # Match found — determine confidence band
        status = self._apply_confidence_bands(best_score)
        identity_data = self.db.data["identities"][best_id]
        existing_status = identity_data.get("status", "unknown")
        existing_name = identity_data.get("canonical_name")

        # If already confirmed and strong match, keep confirmed
        if existing_status == "confirmed" and best_score >= self.auto_confirm_threshold:
            status = "confirmed"
        elif existing_status == "confirmed":
            # Strong existing identity, moderate new match — keep as probable
            status = max(status, "probable", key=lambda s: ["unknown", "candidate", "probable", "confirmed"].index(s))

        # Update cluster with new data + status upgrade
        self.db.update_cluster(best_id, embedding, call_id, speaking_secs, status=status)

        # Record evidence
        self.db.add_evidence(SpeakerEvidence(
            evidence_id=generate_id("ev_"),
            speaker_cluster_id=best_id,
            call_id=call_id,
            type="db_match",
            score=best_score,
            summary=f"Matched {local_label} to {best_id} (cosine={best_score:.3f})",
            details={"cosine_similarity": round(best_score, 4)},
            created_at=now_iso(),
        ))

        return ResolutionResult(
            speaker_cluster_id=best_id,
            local_label=local_label,
            display_name=existing_name,
            status=status,
            confidence=best_score,
            source="db_match",
            is_new_cluster=False,
        )

    def _stage_d_new_cluster(
        self,
        embedding: list[float],
        local_label: str,
        call_id: str,
        speaking_secs: float,
    ) -> ResolutionResult:
        """Stage D: Create a new unknown cluster for an unmatched voice.

        Cross-call stitching happens automatically — when this voice appears
        in future calls, Stage C will match it to this cluster.
        """
        cluster_id = self.db.create_cluster(
            embedding=embedding,
            call_id=call_id,
            local_label=local_label,
            speaking_seconds=speaking_secs,
            status="unknown",
            created_from="auto_cluster",
        )

        return ResolutionResult(
            speaker_cluster_id=cluster_id,
            local_label=local_label,
            display_name=None,
            status="unknown",
            confidence=1.0,  # Confidence in the cluster itself (self-match)
            source="auto_cluster",
            is_new_cluster=True,
        )

    def _stage_e_calendar(
        self,
        results: list[ResolutionResult],
        call_metadata: dict[str, Any],
        calendar_joiner: Any,
    ) -> None:
        """Stage E: Calendar join — match call to event, generate candidates."""
        try:
            audio_meta = call_metadata.get("audio", {})
            format_tags = audio_meta.get("format_tags", {})

            # Get call timestamp from metadata
            creation_time = format_tags.get("creation_time")
            if not creation_time:
                return

            duration = audio_meta.get("duration_seconds", 0)
            event = calendar_joiner.match_call(creation_time, duration)
            if not event:
                return

            # Get already-confirmed names to exclude
            confirmed_names = {r.display_name for r in results if r.display_name}
            resolved_ids = {r.speaker_cluster_id for r in results}

            candidates = calendar_joiner.generate_candidates(event, resolved_ids, confirmed_names)

            if not candidates:
                return

            # Distribute candidates to unresolved speakers
            unresolved = [r for r in results if r.status in ("unknown", "candidate")]
            for r in unresolved:
                r.candidate_names.extend(candidates)

                # Record evidence
                self.db.add_evidence(SpeakerEvidence(
                    evidence_id=generate_id("ev_"),
                    speaker_cluster_id=r.speaker_cluster_id,
                    type="calendar_overlap",
                    score=0.5,
                    summary=f"Calendar event '{event.title}' has unmatched attendees: {[c['name'] for c in candidates]}",
                    details={"event_id": event.event_id, "event_title": event.title},
                    created_at=now_iso(),
                ))

            logger.info(
                "Stage E: Calendar matched '%s' with %d candidates for %d unresolved speakers",
                event.title, len(candidates), len(unresolved),
            )
        except Exception as e:
            logger.debug("Stage E calendar join failed: %s", e)

    def _stage_f_transcript_hints(
        self,
        results: list[ResolutionResult],
        transcript_segments: list[dict[str, Any]],
    ) -> None:
        """Stage F: Extract name hints from transcript via regex."""
        try:
            from audioscript.speakers.transcript_hints import extract_name_hints, match_hints_to_speakers

            hints = extract_name_hints(transcript_segments)
            if not hints:
                return

            grouped = match_hints_to_speakers(hints, transcript_segments)

            for result in results:
                label = result.local_label
                # Self-introductions: name refers to this speaker
                speaker_hints = grouped.get(label, [])
                for hint in speaker_hints:
                    result.candidate_names.append({
                        "name": hint.name,
                        "score": hint.confidence,
                        "source": f"transcript_{hint.pattern}",
                    })

                    self.db.add_evidence(SpeakerEvidence(
                        evidence_id=generate_id("ev_"),
                        speaker_cluster_id=result.speaker_cluster_id,
                        type="transcript_hint",
                        score=hint.confidence,
                        summary=f"Transcript hint: '{hint.name}' via {hint.pattern} at {hint.timestamp:.1f}s",
                        details={"pattern": hint.pattern, "timestamp": hint.timestamp},
                        created_at=now_iso(),
                    ))

            logger.info("Stage F: Found %d name hints in transcript", len(hints))
        except Exception as e:
            logger.debug("Stage F transcript hints failed: %s", e)

    def _stage_g_aggregate_confidence(self, results: list[ResolutionResult]) -> None:
        """Stage G: Re-score confidence using all evidence sources.

        Aggregate: 0.5*embedding + 0.3*calendar + 0.2*transcript
        Only upgrades status, never downgrades confirmed speakers.
        """
        for result in results:
            if result.status == "confirmed":
                continue  # Don't re-score confirmed speakers

            embedding_score = result.confidence if result.source == "db_match" else 0.0

            # Best calendar candidate score
            cal_scores = [c["score"] for c in result.candidate_names if c.get("source") == "calendar_attendee"]
            calendar_score = max(cal_scores) if cal_scores else 0.0

            # Best transcript candidate score
            tx_scores = [c["score"] for c in result.candidate_names if "transcript" in c.get("source", "")]
            transcript_score = max(tx_scores) if tx_scores else 0.0

            # Weighted aggregate
            aggregate = (
                0.5 * embedding_score
                + 0.3 * calendar_score
                + 0.2 * transcript_score
            )

            if aggregate > result.confidence:
                result.confidence = aggregate
                new_status = self._apply_confidence_bands(aggregate)
                # Only upgrade status
                status_order = ["unknown", "candidate", "probable", "confirmed"]
                if status_order.index(new_status) > status_order.index(result.status):
                    result.status = new_status
                    result.source = "aggregate"

    def _apply_confidence_bands(self, score: float) -> str:
        """Map a confidence score to a status (PRD §6)."""
        if score >= self.auto_confirm_threshold:
            return "confirmed"
        if score >= CONFIDENCE_PROBABLE:
            return "probable"
        if score >= CONFIDENCE_CANDIDATE:
            return "candidate"
        return "unknown"

    def _update_co_speakers(self, results: list[ResolutionResult]) -> None:
        """After resolving all speakers in a call, update co-speaker lists."""
        cluster_ids = [r.speaker_cluster_id for r in results]

        for result in results:
            identity = self.db.data["identities"].get(result.speaker_cluster_id)
            if not identity:
                continue

            co_speakers = set(identity.get("typical_co_speakers", []))
            for other_id in cluster_ids:
                if other_id != result.speaker_cluster_id:
                    co_speakers.add(other_id)

            identity["typical_co_speakers"] = list(co_speakers)

    def apply_to_transcript(
        self,
        result_dict: dict[str, Any],
        resolutions: list[ResolutionResult],
    ) -> dict[str, Any]:
        """Apply resolution results to a transcript dict.

        Updates segment speaker labels and adds enriched speaker info.
        """
        # Build mapping: local_label → resolution
        label_map: dict[str, ResolutionResult] = {}
        for res in resolutions:
            label_map[res.local_label] = res

        # Update segment speaker labels
        for seg in result_dict.get("segments", []):
            speaker = seg.get("speaker")
            if speaker and speaker in label_map:
                res = label_map[speaker]
                seg["speaker"] = res.display_name or res.speaker_cluster_id
                seg["speaker_cluster_id"] = res.speaker_cluster_id
                seg["speaker_confidence"] = res.confidence

        # Update diarization metadata
        diar = result_dict.get("diarization", {})
        diar["speakers_resolved"] = [
            {
                "local_label": r.local_label,
                "speaker_cluster_id": r.speaker_cluster_id,
                "display_name": r.display_name,
                "status": r.status,
                "confidence": round(r.confidence, 3),
                "source": r.source,
                "is_new": r.is_new_cluster,
            }
            for r in resolutions
        ]
        result_dict["diarization"] = diar

        return result_dict
