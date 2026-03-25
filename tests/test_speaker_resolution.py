"""Tests for audioscript.speakers.resolution."""

import tempfile
from pathlib import Path

from audioscript.speakers.identity_db import SpeakerIdentityDB
from audioscript.speakers.resolution import (
    CONFIDENCE_AUTO_CONFIRM,
    CONFIDENCE_CANDIDATE,
    CONFIDENCE_PROBABLE,
    SpeakerResolutionEngine,
)


def _make_engine(td):
    """Create a fresh engine with a temporary DB."""
    db_path = Path(td) / "speakers.json"
    db = SpeakerIdentityDB(db_path)
    engine = SpeakerResolutionEngine(db, match_threshold=0.70)
    return engine, db


def _make_diarization(speaker_embeddings, segments=None):
    """Build a minimal diarization result dict."""
    if segments is None:
        segments = []
        for label in speaker_embeddings:
            segments.append({"speaker": label, "start": 0.0, "end": 10.0, "text": "hello"})
    return {
        "speaker_embeddings": speaker_embeddings,
        "segments": segments,
    }


class TestResolveCallEmpty:
    def test_empty_embeddings_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            diar = _make_diarization({})
            results = engine.resolve_call(diar, "call_001")
            assert results == []


class TestResolveCallNewClusters:
    def test_creates_new_clusters_for_unknown_speakers(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            diar = _make_diarization({
                "SPEAKER_00": [1.0, 0.0, 0.0],
                "SPEAKER_01": [0.0, 1.0, 0.0],
            })
            results = engine.resolve_call(diar, "call_001")
            assert len(results) == 2
            assert all(r.is_new_cluster for r in results)
            assert db.cluster_count == 2


class TestResolveCallMatchesExisting:
    def test_matches_existing_clusters(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            # First call: create clusters
            cid = db.create_cluster([1.0, 0.0, 0.0], "call_000", "SPEAKER_00")
            # Second call: similar embedding should match
            diar = _make_diarization({"SPEAKER_00": [0.98, 0.1, 0.0]})
            results = engine.resolve_call(diar, "call_001")
            assert len(results) == 1
            assert results[0].speaker_cluster_id == cid
            assert results[0].is_new_cluster is False


class TestResolveCallNoDuplicateAssignment:
    def test_no_same_cluster_for_two_speakers(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            # Create one cluster
            db.create_cluster([1.0, 0.0, 0.0], "call_000", "S0")
            # Two speakers both similar to that cluster
            diar = _make_diarization({
                "SPEAKER_00": [0.99, 0.05, 0.0],
                "SPEAKER_01": [0.95, 0.1, 0.0],
            })
            results = engine.resolve_call(diar, "call_001")
            cluster_ids = [r.speaker_cluster_id for r in results]
            # All cluster IDs must be unique
            assert len(set(cluster_ids)) == len(cluster_ids)


class TestApplyConfidenceBands:
    def test_returns_correct_status_for_each_band(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            assert engine._apply_confidence_bands(0.95) == "confirmed"
            assert engine._apply_confidence_bands(CONFIDENCE_AUTO_CONFIRM) == "confirmed"
            assert engine._apply_confidence_bands(0.85) == "probable"
            assert engine._apply_confidence_bands(CONFIDENCE_PROBABLE) == "probable"
            assert engine._apply_confidence_bands(0.65) == "candidate"
            assert engine._apply_confidence_bands(CONFIDENCE_CANDIDATE) == "candidate"
            assert engine._apply_confidence_bands(0.50) == "unknown"


class TestApplyToTranscript:
    def test_updates_segment_speaker_labels(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            diar = _make_diarization({"SPEAKER_00": [1.0, 0.0, 0.0]})
            results = engine.resolve_call(diar, "call_001")
            transcript = {
                "segments": [
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "hi"},
                ],
                "diarization": {},
            }
            updated = engine.apply_to_transcript(transcript, results)
            seg = updated["segments"][0]
            assert seg["speaker_cluster_id"] == results[0].speaker_cluster_id
            assert "speaker_confidence" in seg

    def test_adds_speakers_resolved_to_diarization(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            diar = _make_diarization({"SPEAKER_00": [1.0, 0.0, 0.0]})
            results = engine.resolve_call(diar, "call_001")
            transcript = {"segments": [], "diarization": {}}
            updated = engine.apply_to_transcript(transcript, results)
            assert "speakers_resolved" in updated["diarization"]
            assert len(updated["diarization"]["speakers_resolved"]) == 1


class TestCrossCallLinking:
    def test_second_call_matches_first_call_clusters(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            # First call
            diar1 = _make_diarization({"SPEAKER_00": [1.0, 0.0, 0.0]})
            results1 = engine.resolve_call(diar1, "call_001")
            cluster_id_1 = results1[0].speaker_cluster_id
            # Second call with similar embedding
            diar2 = _make_diarization({"SPEAKER_00": [0.98, 0.1, 0.0]})
            results2 = engine.resolve_call(diar2, "call_002")
            # Should match the same cluster
            assert results2[0].speaker_cluster_id == cluster_id_1
            assert results2[0].is_new_cluster is False


class TestHighConfidenceConfirmed:
    def test_confirmed_speaker_stays_confirmed(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            # Create and confirm a cluster
            cid = db.create_cluster([1.0, 0.0, 0.0], "call_000", "S0")
            db.confirm_identity(cid, "Alice")
            # High confidence match
            diar = _make_diarization({"SPEAKER_00": [0.99, 0.01, 0.0]})
            results = engine.resolve_call(diar, "call_001")
            assert results[0].status == "confirmed"
            assert results[0].display_name == "Alice"


class TestNewClusterStatus:
    def test_new_clusters_get_unknown_and_is_new(self):
        with tempfile.TemporaryDirectory() as td:
            engine, db = _make_engine(td)
            diar = _make_diarization({"SPEAKER_00": [1.0, 0.0, 0.0]})
            results = engine.resolve_call(diar, "call_001")
            assert results[0].status == "unknown"
            assert results[0].is_new_cluster is True
