"""Tests for the speaker diarizer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audioscript.processors.diarizer import (
    SpeakerDatabase,
    SpeakerDiarizer,
    _cosine_similarity,
    _load_rttm,
)


# --- SpeakerDiarizer init tests ---

def test_diarizer_requires_hf_token():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="HuggingFace access token"):
            SpeakerDiarizer(hf_token=None)


def test_diarizer_accepts_hf_token():
    diarizer = SpeakerDiarizer(hf_token="test-token")
    assert diarizer.hf_token == "test-token"
    assert diarizer.model_name == "pyannote/speaker-diarization-3.1"


def test_diarizer_reads_env_token():
    with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
        diarizer = SpeakerDiarizer()
        assert diarizer.hf_token == "env-token"


def test_diarizer_batch_sizes():
    diarizer = SpeakerDiarizer(
        hf_token="test", segmentation_batch_size=64, embedding_batch_size=128,
    )
    assert diarizer.segmentation_batch_size == 64
    assert diarizer.embedding_batch_size == 128


# --- Majority speaker tests ---

def test_majority_speaker_single():
    diar_segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    assert SpeakerDiarizer._majority_speaker(1.0, 4.0, diar_segments) == "SPEAKER_00"
    assert SpeakerDiarizer._majority_speaker(6.0, 9.0, diar_segments) == "SPEAKER_01"


def test_majority_speaker_overlap():
    diar_segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    result = SpeakerDiarizer._majority_speaker(2.0, 7.0, diar_segments)
    assert result == "SPEAKER_00"


def test_majority_speaker_no_overlap():
    diar_segments = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
    assert SpeakerDiarizer._majority_speaker(10.0, 15.0, diar_segments) == "UNKNOWN"


def test_majority_speaker_empty():
    assert SpeakerDiarizer._majority_speaker(0.0, 5.0, []) == "UNKNOWN"


# --- Speakers in range (overlap mode) ---

def test_speakers_in_range_single():
    diar_segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 8.0, "speaker": "SPEAKER_01"},
    ]
    result = SpeakerDiarizer._speakers_in_range(2.0, 6.0, diar_segments)
    assert "SPEAKER_00" in result
    assert "SPEAKER_01" in result


def test_speakers_in_range_empty():
    assert SpeakerDiarizer._speakers_in_range(0.0, 5.0, []) == []


def test_speakers_in_range_min_overlap():
    diar_segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 4.95, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    # SPEAKER_01 has only 0.05s overlap, below 0.1 threshold
    result = SpeakerDiarizer._speakers_in_range(0.0, 5.0, diar_segments, min_overlap=0.1)
    assert result == ["SPEAKER_00"]


# --- Assign speakers tests ---

def test_assign_speakers_segments():
    diarizer = SpeakerDiarizer(hf_token="test-token")
    whisper_result = {
        "text": "Hello. Goodbye.",
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "Hello."},
            {"start": 5.0, "end": 8.0, "text": "Goodbye."},
        ],
    }
    diarization = {
        "segments": [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
            {"start": 4.5, "end": 9.0, "speaker": "SPEAKER_01"},
        ],
        "num_speakers": 2,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "speaker_embeddings": {},
        "overlap": {},
    }
    result = diarizer.assign_speakers(whisper_result, diarization)
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][1]["speaker"] == "SPEAKER_01"
    assert result["diarization"]["num_speakers"] == 2


def test_assign_speakers_with_words():
    diarizer = SpeakerDiarizer(hf_token="test-token")
    whisper_result = {
        "text": "Hello there. Goodbye now.",
        "segments": [
            {
                "start": 0.0, "end": 5.0, "text": "Hello there.",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "Hello"},
                    {"start": 1.5, "end": 2.5, "word": "there."},
                ],
            },
        ],
    }
    diarization = {
        "segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
        "num_speakers": 1,
        "speakers": ["SPEAKER_00"],
        "speaker_embeddings": {},
        "overlap": {},
    }
    result = diarizer.assign_speakers(whisper_result, diarization)
    assert result["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][0]["words"][1]["speaker"] == "SPEAKER_00"


def test_assign_speakers_with_overlap():
    diarizer = SpeakerDiarizer(hf_token="test-token")
    whisper_result = {
        "text": "Hello there.",
        "segments": [{"start": 0.0, "end": 5.0, "text": "Hello there."}],
    }
    diarization = {
        "segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
        ],
        "num_speakers": 2,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "speaker_embeddings": {},
        "overlap": {},
    }
    result = diarizer.assign_speakers(whisper_result, diarization, allow_overlap=True)
    seg = result["segments"][0]
    assert isinstance(seg["speaker"], list)
    assert "SPEAKER_00" in seg["speaker"]
    assert "SPEAKER_01" in seg["speaker"]
    assert seg["is_overlap"] is True


def test_assign_speakers_with_speaker_db():
    diarizer = SpeakerDiarizer(hf_token="test-token")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "speakers.json"
        db = SpeakerDatabase(db_path)
        db.add_speaker("Alice", np.array([1.0, 0.0, 0.0]))
        db.add_speaker("Bob", np.array([0.0, 1.0, 0.0]))

        whisper_result = {
            "text": "Hello. Goodbye.",
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "Hello."},
                {"start": 5.0, "end": 8.0, "text": "Goodbye."},
            ],
        }
        diarization = {
            "segments": [
                {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
                {"start": 4.5, "end": 9.0, "speaker": "SPEAKER_01"},
            ],
            "num_speakers": 2,
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_embeddings": {
                "SPEAKER_00": np.array([0.99, 0.01, 0.0]),  # close to Alice
                "SPEAKER_01": np.array([0.01, 0.98, 0.02]),  # close to Bob
            },
            "overlap": {},
        }

        result = diarizer.assign_speakers(
            whisper_result, diarization, speaker_db=db, similarity_threshold=0.9,
        )
        assert result["segments"][0]["speaker"] == "Alice"
        assert result["segments"][1]["speaker"] == "Bob"
        assert result["diarization"]["identified"]["SPEAKER_00"] == "Alice"


# --- SpeakerDatabase tests ---

def test_speaker_database_create():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "speakers.json"
        db = SpeakerDatabase(db_path)
        assert db.speaker_names == []

        db.add_speaker("Alice", np.array([1.0, 0.0, 0.0]))
        assert "Alice" in db.speaker_names
        assert db_path.exists()


def test_speaker_database_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "speakers.json"
        db1 = SpeakerDatabase(db_path)
        db1.add_speaker("Alice", np.array([1.0, 0.0, 0.0]))

        db2 = SpeakerDatabase(db_path)
        assert "Alice" in db2.speaker_names


def test_speaker_database_running_mean():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "speakers.json"
        db = SpeakerDatabase(db_path)
        db.add_speaker("Alice", np.array([1.0, 0.0]))
        db.add_speaker("Alice", np.array([0.0, 1.0]))

        emb = np.array(db.data["speakers"]["Alice"]["embedding"])
        assert db.data["speakers"]["Alice"]["num_samples"] == 2
        np.testing.assert_allclose(emb, [0.5, 0.5])


def test_speaker_database_remove():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "speakers.json"
        db = SpeakerDatabase(db_path)
        db.add_speaker("Alice", np.array([1.0]))
        assert db.remove_speaker("Alice") is True
        assert "Alice" not in db.speaker_names
        assert db.remove_speaker("Alice") is False


def test_speaker_database_identify():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "speakers.json"
        db = SpeakerDatabase(db_path)
        db.add_speaker("Alice", np.array([1.0, 0.0, 0.0]))
        db.add_speaker("Bob", np.array([0.0, 1.0, 0.0]))

        embeddings = {
            "SPEAKER_00": np.array([0.98, 0.02, 0.0]),
            "SPEAKER_01": np.array([0.01, 0.99, 0.0]),
        }
        mapping = db.identify(embeddings, threshold=0.9)
        assert mapping["SPEAKER_00"] == "Alice"
        assert mapping["SPEAKER_01"] == "Bob"


def test_speaker_database_identify_below_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "speakers.json"
        db = SpeakerDatabase(db_path)
        db.add_speaker("Alice", np.array([1.0, 0.0]))

        embeddings = {"SPEAKER_00": np.array([0.0, 1.0])}
        mapping = db.identify(embeddings, threshold=0.5)
        assert mapping == {}


# --- Cosine similarity ---

def test_cosine_similarity():
    assert _cosine_similarity(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1.0)
    assert _cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)
    assert _cosine_similarity(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(-1.0)
    assert _cosine_similarity(np.array([0, 0]), np.array([1, 0])) == 0.0


# --- RTTM ---

def test_save_rttm():
    diarizer = SpeakerDiarizer(hf_token="test-token")
    segments = [
        {"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 7.5, "speaker": "SPEAKER_01"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rttm_path = Path(tmp) / "output.rttm"
        diarizer.save_rttm(segments, rttm_path, file_id="test")

        content = rttm_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert "SPEAKER_00" in lines[0]
        assert "SPEAKER_01" in lines[1]
        parts = lines[0].split()
        assert parts[0] == "SPEAKER"
        assert float(parts[3]) == 0.0
        assert float(parts[4]) == 3.5


def test_load_rttm():
    with tempfile.TemporaryDirectory() as tmp:
        rttm_path = Path(tmp) / "ref.rttm"
        rttm_path.write_text(
            "SPEAKER test 1 0.000 3.500 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            "SPEAKER test 1 4.000 3.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
        )
        annotation = _load_rttm(rttm_path, file_id="test")
        tracks = list(annotation.itertracks(yield_label=True))
        assert len(tracks) == 2
        assert tracks[0][2] == "SPEAKER_00"
        assert tracks[1][2] == "SPEAKER_01"


# --- Save embeddings ---

def test_save_embeddings():
    diarizer = SpeakerDiarizer(hf_token="test-token")
    embeddings = {
        "SPEAKER_00": np.array([0.1, 0.2, 0.3]),
        "SPEAKER_01": np.array([0.4, 0.5, 0.6]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        emb_path = Path(tmp) / "embeddings.json"
        diarizer.save_embeddings(embeddings, emb_path)

        with open(emb_path) as f:
            data = json.load(f)
        assert "SPEAKER_00" in data
        assert data["SPEAKER_00"]["dimension"] == 3
        assert len(data["SPEAKER_00"]["embedding"]) == 3
