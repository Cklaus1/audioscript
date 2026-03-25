"""Speaker identity database — persistent cross-call voice tracking.

Replaces the old SpeakerDatabase (name→embedding only) with a full
cluster-based identity system. Every voice gets a stable spk_xxxx ID,
even if unnamed. Cross-call linking happens naturally.

Storage: JSON file with atomic writes (same pattern as ProcessingManifest).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from audioscript.speakers.models import (
    SpeakerEvidence,
    SpeakerIdentity,
    SpeakerOccurrence,
    generate_id,
    generate_speaker_id,
    now_iso,
)

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SpeakerIdentityDB:
    """Persistent speaker identity store with cross-call linking.

    Guiding principle: Link first. Name second. Confirm last.
    """

    VERSION = "2.0"

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load the identity database or create empty."""
        if not self.db_path.exists():
            return {"version": self.VERSION, "identities": {}, "occurrences": [], "evidence": []}

        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)

            # Auto-migrate from v1 format
            if data.get("version") == "1.0" and "speakers" in data:
                return self._migrate_v1(data)

            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load speaker identity DB: %s", e)
            return {"version": self.VERSION, "identities": {}, "occurrences": [], "evidence": []}

    def save(self) -> None:
        """Persist to disk atomically (write to temp, then rename)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=self.db_path.parent,
            prefix=".speaker_id_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
            os.replace(tmp_path, self.db_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # --- Core Matching ---

    def match_embedding(
        self,
        embedding: list[float],
        threshold: float = 0.70,
    ) -> tuple[str | None, float]:
        """Find best matching cluster by cosine similarity.

        Returns (speaker_cluster_id, score) or (None, 0.0).
        """
        best_id = None
        best_score = 0.0

        for cluster_id, identity_data in self.data["identities"].items():
            centroid = identity_data.get("embedding_centroid", [])
            if not centroid:
                continue

            score = _cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_id = cluster_id

        if best_id and best_score >= threshold:
            return best_id, best_score
        return None, best_score

    # --- Cluster Management ---

    def create_cluster(
        self,
        embedding: list[float],
        call_id: str,
        local_label: str,
        speaking_seconds: float = 0.0,
        status: str = "unknown",
        name: str | None = None,
        created_from: str = "auto_cluster",
    ) -> str:
        """Create a new speaker cluster. Returns the new spk_xxxx ID."""
        cluster_id = generate_speaker_id()
        now = now_iso()

        identity = SpeakerIdentity(
            speaker_cluster_id=cluster_id,
            canonical_name=name,
            status=status,
            embedding_centroid=list(embedding),
            sample_count=1,
            first_seen=now,
            last_seen=now,
            total_calls=1,
            total_speaking_seconds=speaking_seconds,
            created_from=created_from,
            updated_at=now,
        )

        self.data["identities"][cluster_id] = identity.to_dict()

        # Record evidence
        self.add_evidence(SpeakerEvidence(
            evidence_id=generate_id("ev_"),
            speaker_cluster_id=cluster_id,
            call_id=call_id,
            type="auto_cluster",
            score=1.0,
            summary=f"New cluster created from {local_label} in {call_id}",
            created_at=now,
        ))

        self.save()
        logger.info("Created speaker cluster %s from %s", cluster_id, local_label)
        return cluster_id

    def update_cluster(
        self,
        cluster_id: str,
        embedding: list[float],
        call_id: str,
        speaking_seconds: float = 0.0,
    ) -> None:
        """Update an existing cluster with a new occurrence.

        Incremental centroid averaging + metadata update.
        """
        identity = self.data["identities"].get(cluster_id)
        if not identity:
            return

        now = now_iso()

        # Incremental centroid averaging
        old_centroid = identity["embedding_centroid"]
        n = identity["sample_count"]
        if old_centroid and len(old_centroid) == len(embedding):
            new_centroid = [
                (old * n + new) / (n + 1)
                for old, new in zip(old_centroid, embedding)
            ]
            identity["embedding_centroid"] = new_centroid
        else:
            identity["embedding_centroid"] = list(embedding)

        identity["sample_count"] = n + 1
        identity["last_seen"] = now
        identity["total_calls"] = identity.get("total_calls", 0) + 1
        identity["total_speaking_seconds"] = (
            identity.get("total_speaking_seconds", 0) + speaking_seconds
        )
        identity["updated_at"] = now

        self.save()

    # --- Occurrence Tracking ---

    def add_occurrence(self, occurrence: SpeakerOccurrence) -> None:
        """Record a speaker appearance in a call."""
        self.data["occurrences"].append(occurrence.to_dict())
        self.save()

    # --- Evidence Trail ---

    def add_evidence(self, evidence: SpeakerEvidence) -> None:
        """Record an identity decision with evidence."""
        self.data["evidence"].append(evidence.to_dict())
        # Don't save here — caller manages save batching

    # --- Identity Management ---

    def get_identity(self, cluster_id: str) -> SpeakerIdentity | None:
        """Get a speaker identity by cluster ID."""
        data = self.data["identities"].get(cluster_id)
        if not data:
            return None
        return SpeakerIdentity.from_dict(data)

    def list_identities(self, status: str | None = None) -> list[SpeakerIdentity]:
        """List all speaker identities, optionally filtered by status."""
        identities = []
        for data in self.data["identities"].values():
            if status and data.get("status") != status:
                continue
            identities.append(SpeakerIdentity.from_dict(data))
        return identities

    def confirm_identity(
        self,
        cluster_id: str,
        name: str,
        source: str = "user_confirmation",
    ) -> bool:
        """Assign a confirmed name to a speaker cluster."""
        identity = self.data["identities"].get(cluster_id)
        if not identity:
            return False

        now = now_iso()
        identity["canonical_name"] = name
        identity["status"] = "confirmed"
        identity["updated_at"] = now

        self.add_evidence(SpeakerEvidence(
            evidence_id=generate_id("ev_"),
            speaker_cluster_id=cluster_id,
            type=source,
            score=1.0,
            summary=f"Identity confirmed as '{name}' via {source}",
            created_at=now,
        ))

        self.save()
        logger.info("Confirmed %s as '%s'", cluster_id, name)
        return True

    def set_status(self, cluster_id: str, status: str) -> bool:
        """Update the resolution status of a cluster."""
        identity = self.data["identities"].get(cluster_id)
        if not identity:
            return False
        identity["status"] = status
        identity["updated_at"] = now_iso()
        self.save()
        return True

    def add_alias(self, cluster_id: str, alias: str) -> bool:
        """Add an alternative name to a cluster."""
        identity = self.data["identities"].get(cluster_id)
        if not identity:
            return False
        aliases = identity.get("aliases", [])
        if alias not in aliases:
            aliases.append(alias)
            identity["aliases"] = aliases
            identity["updated_at"] = now_iso()
            self.save()
        return True

    # --- Query ---

    def get_occurrences(
        self,
        cluster_id: str | None = None,
        call_id: str | None = None,
    ) -> list[SpeakerOccurrence]:
        """Get occurrences, optionally filtered."""
        results = []
        for data in self.data["occurrences"]:
            if cluster_id and data.get("speaker_cluster_id") != cluster_id:
                continue
            if call_id and data.get("call_id") != call_id:
                continue
            results.append(SpeakerOccurrence.from_dict(data))
        return results

    def get_unknown_speakers(self, min_calls: int = 1) -> list[SpeakerIdentity]:
        """Get unknown speakers with at least min_calls appearances."""
        return [
            SpeakerIdentity.from_dict(data)
            for data in self.data["identities"].values()
            if data.get("status") in ("unknown", "candidate")
            and data.get("total_calls", 0) >= min_calls
        ]

    def get_co_speakers(self, cluster_id: str) -> list[str]:
        """Get cluster IDs that frequently appear in the same calls."""
        identity = self.data["identities"].get(cluster_id)
        if not identity:
            return []
        return identity.get("typical_co_speakers", [])

    @property
    def cluster_count(self) -> int:
        return len(self.data["identities"])

    @property
    def confirmed_count(self) -> int:
        return sum(
            1 for d in self.data["identities"].values()
            if d.get("status") == "confirmed"
        )

    @property
    def unknown_count(self) -> int:
        return sum(
            1 for d in self.data["identities"].values()
            if d.get("status") in ("unknown", "candidate")
        )

    # --- Migration ---

    def _migrate_v1(self, old_data: dict[str, Any]) -> dict[str, Any]:
        """Migrate from v1 speakers.json (name→embedding) to v2 format."""
        logger.info("Migrating speaker DB from v1.0 to v2.0")
        now = now_iso()
        new_data: dict[str, Any] = {
            "version": self.VERSION,
            "identities": {},
            "occurrences": [],
            "evidence": [],
        }

        for name, speaker_data in old_data.get("speakers", {}).items():
            cluster_id = generate_speaker_id()
            embedding = speaker_data.get("embedding", [])
            num_samples = speaker_data.get("num_samples", 1)

            identity = SpeakerIdentity(
                speaker_cluster_id=cluster_id,
                canonical_name=name,
                status="confirmed",
                embedding_centroid=embedding,
                sample_count=num_samples,
                first_seen=now,
                last_seen=now,
                total_calls=num_samples,
                total_speaking_seconds=0.0,
                created_from="migrated_v1",
                updated_at=now,
            )
            new_data["identities"][cluster_id] = identity.to_dict()

        logger.info("Migrated %d speakers to v2 format", len(new_data["identities"]))
        return new_data

    def migrate_from_v1(self, old_db_path: Path) -> int:
        """Explicitly migrate an old v1 speakers.json file."""
        try:
            with open(old_db_path) as f:
                old_data = json.load(f)
            if old_data.get("version") == "1.0":
                self.data = self._migrate_v1(old_data)
                self.save()
                return len(self.data["identities"])
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to migrate v1 DB: %s", e)
        return 0
