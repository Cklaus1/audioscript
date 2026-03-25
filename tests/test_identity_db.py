"""Tests for audioscript.speakers.identity_db."""

import tempfile
from pathlib import Path

from audioscript.speakers.identity_db import SpeakerIdentityDB
from audioscript.speakers.models import SpeakerOccurrence


class TestSpeakerIdentityDBInit:
    def test_creates_empty_db_when_file_missing(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            assert db.data["version"] == "2.0"
            assert db.data["identities"] == {}
            assert db.data["occurrences"] == []
            assert db.data["evidence"] == []


class TestCreateCluster:
    def test_creates_cluster_with_spk_id_and_saves(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "speakers.json"
            db = SpeakerIdentityDB(db_path)
            cid = db.create_cluster(
                embedding=[1.0, 0.0, 0.0],
                call_id="call_001",
                local_label="SPEAKER_00",
                speaking_seconds=10.0,
            )
            assert cid.startswith("spk_")
            assert len(db.data["identities"]) == 1
            # Caller must save explicitly (batch pattern)
            db.save()
            db2 = SpeakerIdentityDB(db_path)
            assert cid in db2.data["identities"]


class TestMatchEmbedding:
    def test_finds_best_match_above_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            cid = db.create_cluster(
                embedding=[1.0, 0.0, 0.0],
                call_id="call_001",
                local_label="SPEAKER_00",
            )
            match_id, score = db.match_embedding([0.9, 0.1, 0.0], threshold=0.70)
            assert match_id == cid
            assert score > 0.70

    def test_returns_none_below_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            db.create_cluster(
                embedding=[1.0, 0.0, 0.0],
                call_id="call_001",
                local_label="SPEAKER_00",
            )
            # Orthogonal vector => cosine ~ 0
            match_id, score = db.match_embedding([0.0, 0.0, 1.0], threshold=0.70)
            assert match_id is None

    def test_returns_none_for_empty_db(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            match_id, score = db.match_embedding([1.0, 0.0, 0.0])
            assert match_id is None
            assert score == 0.0


class TestUpdateCluster:
    def test_updates_centroid_via_incremental_averaging(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            cid = db.create_cluster(
                embedding=[1.0, 0.0, 0.0],
                call_id="call_001",
                local_label="SPEAKER_00",
            )
            db.update_cluster(cid, [0.0, 1.0, 0.0], "call_002")
            centroid = db.data["identities"][cid]["embedding_centroid"]
            # After averaging (1,0,0) and (0,1,0): should be (0.5, 0.5, 0)
            assert abs(centroid[0] - 0.5) < 1e-9
            assert abs(centroid[1] - 0.5) < 1e-9
            assert abs(centroid[2] - 0.0) < 1e-9

    def test_increments_total_calls_and_speaking_seconds(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            cid = db.create_cluster(
                embedding=[1.0, 0.0, 0.0],
                call_id="call_001",
                local_label="SPEAKER_00",
                speaking_seconds=10.0,
            )
            db.update_cluster(cid, [0.9, 0.1, 0.0], "call_002", speaking_seconds=20.0)
            identity = db.data["identities"][cid]
            assert identity["total_calls"] == 2
            assert identity["total_speaking_seconds"] == 30.0


class TestOccurrences:
    def _make_db(self, td):
        db = SpeakerIdentityDB(Path(td) / "speakers.json")
        occ1 = SpeakerOccurrence(
            occurrence_id="occ_001",
            call_id="call_001",
            speaker_cluster_id="spk_aabbccdd",
            local_label="SPEAKER_00",
        )
        occ2 = SpeakerOccurrence(
            occurrence_id="occ_002",
            call_id="call_002",
            speaker_cluster_id="spk_aabbccdd",
            local_label="SPEAKER_00",
        )
        occ3 = SpeakerOccurrence(
            occurrence_id="occ_003",
            call_id="call_001",
            speaker_cluster_id="spk_11223344",
            local_label="SPEAKER_01",
        )
        db.add_occurrence(occ1)
        db.add_occurrence(occ2)
        db.add_occurrence(occ3)
        return db

    def test_add_occurrence_records(self):
        with tempfile.TemporaryDirectory() as td:
            db = self._make_db(td)
            assert len(db.data["occurrences"]) == 3

    def test_get_occurrences_filters_by_cluster_id(self):
        with tempfile.TemporaryDirectory() as td:
            db = self._make_db(td)
            occs = db.get_occurrences(cluster_id="spk_aabbccdd")
            assert len(occs) == 2

    def test_get_occurrences_filters_by_call_id(self):
        with tempfile.TemporaryDirectory() as td:
            db = self._make_db(td)
            occs = db.get_occurrences(call_id="call_001")
            assert len(occs) == 2


class TestConfirmIdentity:
    def test_sets_name_and_status_to_confirmed(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            cid = db.create_cluster(
                embedding=[1.0, 0.0, 0.0],
                call_id="call_001",
                local_label="SPEAKER_00",
            )
            result = db.confirm_identity(cid, "Alice")
            assert result is True
            identity = db.data["identities"][cid]
            assert identity["canonical_name"] == "Alice"
            assert identity["status"] == "confirmed"


class TestListIdentities:
    def test_returns_all_identities(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            db.create_cluster([1.0, 0.0, 0.0], "c1", "S0")
            db.create_cluster([0.0, 1.0, 0.0], "c2", "S1")
            assert len(db.list_identities()) == 2

    def test_filters_by_status(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            cid1 = db.create_cluster([1.0, 0.0, 0.0], "c1", "S0")
            db.create_cluster([0.0, 1.0, 0.0], "c2", "S1")
            db.confirm_identity(cid1, "Alice")
            unknowns = db.list_identities(status="unknown")
            assert len(unknowns) == 1


class TestGetUnknownSpeakers:
    def test_returns_only_unknown_and_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            cid1 = db.create_cluster([1.0, 0.0, 0.0], "c1", "S0")
            db.create_cluster([0.0, 1.0, 0.0], "c2", "S1")
            db.confirm_identity(cid1, "Alice")
            unknowns = db.get_unknown_speakers()
            assert len(unknowns) == 1
            assert unknowns[0].status == "unknown"


class TestMigrateV1:
    def test_converts_old_format_to_v2(self):
        with tempfile.TemporaryDirectory() as td:
            db = SpeakerIdentityDB(Path(td) / "speakers.json")
            old_data = {
                "version": "1.0",
                "speakers": {
                    "Alice": {"embedding": [1.0, 0.0, 0.0], "num_samples": 5},
                    "Bob": {"embedding": [0.0, 1.0, 0.0], "num_samples": 3},
                },
            }
            new_data = db._migrate_v1(old_data)
            assert new_data["version"] == "2.0"
            assert len(new_data["identities"]) == 2
            # Check that names are preserved as canonical_name
            names = [
                v["canonical_name"] for v in new_data["identities"].values()
            ]
            assert "Alice" in names
            assert "Bob" in names
            # Check status is confirmed
            for v in new_data["identities"].values():
                assert v["status"] == "confirmed"
