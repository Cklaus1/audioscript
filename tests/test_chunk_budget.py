"""The summary chunk cap fits the 16k-token local model, and a partial merge says so."""
from audioscript.config.settings import AudioScriptConfig
from audioscript.processors.chunker import chunk_transcript


def _segments(n, words=40):
    return [{"start": i * 6.0, "end": i * 6.0 + 5.5, "speaker": f"SPEAKER_0{i % 3}",
             "text": " ".join(["word"] * words)} for i in range(n)]


def test_default_chunk_cap_is_12k():
    assert AudioScriptConfig().llm_max_chunk_chars == 12_000


def test_every_chunk_respects_the_cap():
    chunks = chunk_transcript(_segments(1500), target_minutes=30, max_chars=12_000)
    assert len(chunks) > 1
    assert all(c.char_count <= 12_000 for c in chunks)
