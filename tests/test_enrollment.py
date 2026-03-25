"""Tests for audioscript.speakers.enrollment — SpeakerEnrollment."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioscript.speakers.enrollment import SpeakerEnrollment
from audioscript.speakers.identity_db import SpeakerIdentityDB


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SpeakerIdentityDB."""
    db_path = tmp_path / "speakers_identity.json"
    return SpeakerIdentityDB(db_path)


def _mock_diarize_result(speaker: str = "SPEAKER_00", duration: float = 30.0):
    """Create a mock diarization result."""
    return {
        "segments": [
            {"speaker": speaker, "start": 0.0, "end": duration, "text": "hello"},
        ],
        "speaker_embeddings": {
            speaker: [0.1, 0.2, 0.3, 0.4, 0.5],
        },
    }


class TestSpeakerEnrollment:

    def test_enroll_from_cluster_confirms_existing(self, tmp_db):
        """enroll_from_cluster confirms an existing cluster with a name."""
        cluster_id = tmp_db.create_cluster(
            embedding=[0.1, 0.2, 0.3],
            call_id="call_001",
            local_label="SPEAKER_00",
            status="unknown",
        )
        enrollment = SpeakerEnrollment(tmp_db)
        result = enrollment.enroll_from_cluster("Alice", cluster_id)
        assert result is True

        identity = tmp_db.get_identity(cluster_id)
        assert identity.canonical_name == "Alice"
        assert identity.status == "confirmed"

    @patch("audioscript.processors.diarizer.SpeakerDiarizer", create=True)
    def test_enroll_from_audio_creates_confirmed_cluster(self, mock_diarizer_cls, tmp_db):
        """enroll_from_audio with mock diarizer creates a confirmed cluster."""
        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = _mock_diarize_result()
        mock_diarizer_cls.return_value = mock_diarizer

        enrollment = SpeakerEnrollment(tmp_db)
        cluster_id = enrollment.enroll_from_audio("Bob", Path("/fake/sample.wav"))

        assert cluster_id.startswith("spk_")
        identity = tmp_db.get_identity(cluster_id)
        assert identity.canonical_name == "Bob"
        assert identity.status == "confirmed"

    @patch("audioscript.processors.diarizer.SpeakerDiarizer", create=True)
    def test_enroll_from_audio_matches_existing_cluster(self, mock_diarizer_cls, tmp_db):
        """enroll_from_audio matches existing cluster when voice is already known."""
        # Pre-create a cluster with the same embedding
        existing_id = tmp_db.create_cluster(
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            call_id="call_old",
            local_label="SPEAKER_00",
            status="unknown",
        )

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = _mock_diarize_result()
        mock_diarizer_cls.return_value = mock_diarizer

        enrollment = SpeakerEnrollment(tmp_db)
        # The embedding is identical so cosine sim = 1.0, above 0.80 threshold
        cluster_id = enrollment.enroll_from_audio("Carol", Path("/fake/sample.wav"))

        assert cluster_id == existing_id
        identity = tmp_db.get_identity(existing_id)
        assert identity.canonical_name == "Carol"
        assert identity.status == "confirmed"

    @patch("audioscript.processors.diarizer.SpeakerDiarizer", create=True)
    def test_enrollment_creates_evidence_trail(self, mock_diarizer_cls, tmp_db):
        """Enrollment creates evidence records in the DB."""
        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = _mock_diarize_result()
        mock_diarizer_cls.return_value = mock_diarizer

        enrollment = SpeakerEnrollment(tmp_db)
        cluster_id = enrollment.enroll_from_audio("Dave", Path("/fake/sample.wav"))

        evidence = tmp_db.data["evidence"]
        # Should have at least one evidence record for this cluster
        cluster_evidence = [e for e in evidence if e["speaker_cluster_id"] == cluster_id]
        assert len(cluster_evidence) >= 1
        # Check that evidence has proper fields
        ev = cluster_evidence[0]
        assert "evidence_id" in ev
        assert "summary" in ev
        assert ev["speaker_cluster_id"] == cluster_id
