"""Tests for audioscript.speakers.models."""

import re

from audioscript.speakers.models import (
    SpeakerEvidence,
    SpeakerIdentity,
    SpeakerOccurrence,
    SpeakerReviewItem,
    generate_id,
    generate_speaker_id,
)


class TestGenerateIds:
    def test_generate_speaker_id_format(self):
        sid = generate_speaker_id()
        assert re.fullmatch(r"spk_[0-9a-f]{8}", sid), f"Bad speaker id: {sid}"

    def test_generate_id_with_prefix(self):
        eid = generate_id("ev_")
        assert eid.startswith("ev_")
        assert len(eid) == len("ev_") + 12


class TestSpeakerIdentity:
    def test_display_name_returns_canonical_name(self):
        si = SpeakerIdentity(speaker_cluster_id="spk_aabbccdd", canonical_name="Alice")
        assert si.display_name == "Alice"

    def test_display_name_returns_cluster_id_when_no_name(self):
        si = SpeakerIdentity(speaker_cluster_id="spk_aabbccdd")
        assert si.display_name == "spk_aabbccdd"

    def test_to_dict_from_dict_roundtrip(self):
        si = SpeakerIdentity(
            speaker_cluster_id="spk_aabbccdd",
            canonical_name="Bob",
            aliases=["Robert"],
            status="confirmed",
            embedding_centroid=[1.0, 2.0, 3.0],
            sample_count=5,
            first_seen="2025-01-01T00:00:00",
            last_seen="2025-06-01T00:00:00",
            total_calls=5,
            total_speaking_seconds=120.5,
            typical_co_speakers=["spk_11223344"],
            created_from="auto_cluster",
            updated_at="2025-06-01T00:00:00",
        )
        d = si.to_dict()
        restored = SpeakerIdentity.from_dict(d)
        assert restored.speaker_cluster_id == si.speaker_cluster_id
        assert restored.canonical_name == si.canonical_name
        assert restored.aliases == si.aliases
        assert restored.embedding_centroid == si.embedding_centroid
        assert restored.total_speaking_seconds == si.total_speaking_seconds


class TestSpeakerOccurrence:
    def test_to_dict_from_dict_roundtrip(self):
        occ = SpeakerOccurrence(
            occurrence_id="occ_abc123",
            call_id="call_001",
            speaker_cluster_id="spk_aabbccdd",
            local_label="SPEAKER_00",
            display_name="Alice",
            identity_status="confirmed",
            resolution_source="db_match",
            resolution_confidence=0.95,
            total_speaking_seconds=30.0,
        )
        d = occ.to_dict()
        restored = SpeakerOccurrence.from_dict(d)
        assert restored.occurrence_id == occ.occurrence_id
        assert restored.call_id == occ.call_id
        assert restored.resolution_confidence == occ.resolution_confidence


class TestSpeakerEvidence:
    def test_to_dict_from_dict_roundtrip(self):
        ev = SpeakerEvidence(
            evidence_id="ev_abc123",
            speaker_cluster_id="spk_aabbccdd",
            call_id="call_001",
            type="db_match",
            score=0.92,
            summary="Matched speaker",
            details={"cosine_similarity": 0.92},
            created_at="2025-01-01T00:00:00",
        )
        d = ev.to_dict()
        restored = SpeakerEvidence.from_dict(d)
        assert restored.evidence_id == ev.evidence_id
        assert restored.score == ev.score
        assert restored.details == ev.details


class TestSpeakerReviewItem:
    def test_to_dict_from_dict_roundtrip(self):
        ri = SpeakerReviewItem(
            review_item_id="ri_abc123",
            speaker_cluster_id="spk_aabbccdd",
            priority_score=0.8,
            reason="frequent unknown",
            summary="Appears in 12 calls",
            candidate_names=["Alice", "Bob"],
            status="open",
            created_at="2025-01-01T00:00:00",
        )
        d = ri.to_dict()
        restored = SpeakerReviewItem.from_dict(d)
        assert restored.review_item_id == ri.review_item_id
        assert restored.candidate_names == ri.candidate_names
        assert restored.status == ri.status
