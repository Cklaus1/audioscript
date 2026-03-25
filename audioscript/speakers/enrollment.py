"""Speaker enrollment — register known speakers from voice samples or clusters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audioscript.speakers.identity_db import SpeakerIdentityDB
from audioscript.speakers.models import SpeakerEvidence, generate_id, now_iso

logger = logging.getLogger(__name__)


class SpeakerEnrollment:
    """Enroll speakers into the identity database."""

    def __init__(self, identity_db: SpeakerIdentityDB) -> None:
        self.db = identity_db

    def enroll_from_audio(
        self,
        name: str,
        audio_path: Path,
        hf_token: str | None = None,
    ) -> str:
        """Enroll a speaker from a voice sample.

        Runs diarization on the sample to extract speaker embeddings,
        then creates a confirmed cluster with the given name.

        Returns the new speaker_cluster_id.
        """
        from audioscript.processors.diarizer import SpeakerDiarizer

        diarizer = SpeakerDiarizer(hf_token=hf_token)
        diar_result = diarizer.diarize(audio_path)

        embeddings = diar_result.get("speaker_embeddings", {})
        if not embeddings:
            raise ValueError(f"No speakers detected in {audio_path}")

        # Use the speaker with the most speaking time
        segments = diar_result.get("segments", [])
        speaking_times: dict[str, float] = {}
        for seg in segments:
            spk = seg.get("speaker", "")
            duration = seg.get("end", 0) - seg.get("start", 0)
            speaking_times[spk] = speaking_times.get(spk, 0) + duration

        primary_speaker = max(speaking_times, key=speaking_times.get)
        embedding = embeddings[primary_speaker]

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        # Check if this voice already exists
        match_id, match_score = self.db.match_embedding(embedding, threshold=0.80)
        if match_id:
            # Update existing cluster with the name
            self.db.confirm_identity(match_id, name, source="enrollment")
            logger.info("Enrolled '%s' → existing cluster %s (score=%.3f)", name, match_id, match_score)
            return match_id

        # Create new confirmed cluster
        cluster_id = self.db.create_cluster(
            embedding=embedding,
            call_id=f"enrollment_{audio_path.stem}",
            local_label="ENROLLED",
            speaking_seconds=speaking_times.get(primary_speaker, 0),
            status="confirmed",
            name=name,
            created_from="enrollment",
        )

        logger.info("Enrolled '%s' → new cluster %s", name, cluster_id)
        return cluster_id

    def enroll_from_cluster(
        self,
        name: str,
        cluster_id: str,
    ) -> bool:
        """Promote an existing cluster to confirmed with a name.

        This is the same as `speakers label` but via the enrollment API.
        """
        return self.db.confirm_identity(cluster_id, name, source="enrollment")
