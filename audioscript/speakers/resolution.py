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
    ) -> list[ResolutionResult]:
        """Run full resolution pipeline for all speakers in one call.

        Args:
            diarization_result: Output from SpeakerDiarizer.diarize() containing
                'speakers', 'segments', and 'speaker_embeddings'.
            call_id: Unique identifier for this call (usually file hash).
            file_path: Source audio file path (for logging).

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

        # Update cluster with new data
        self.db.update_cluster(best_id, embedding, call_id, speaking_secs)

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
