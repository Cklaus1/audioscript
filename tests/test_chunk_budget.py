"""Summary chunks fit a 16k-token local model (English and Korean); invalid JSON is retried."""
import json
from types import SimpleNamespace

from audioscript.config.settings import AudioScriptConfig
from audioscript.llm import analyzer
from audioscript.processors.chunker import chunk_transcript, estimate_segment_tokens


def _segments(n, text):
    return [{"start": i * 6.0, "end": i * 6.0 + 5.5, "speaker": f"SPEAKER_0{i % 3}", "text": text}
            for i in range(n)]


def test_default_budget_is_7500_estimated_tokens():
    assert AudioScriptConfig().llm_max_chunk_tokens == 7_500


def test_korean_counts_about_one_token_per_character():
    en = estimate_segment_tokens({"text": "a" * 300})
    ko = estimate_segment_tokens({"text": "가" * 300})
    assert ko >= 300 and en < 120


def test_korean_chunks_are_smaller_than_english_ones_in_characters():
    en = chunk_transcript(_segments(1500, " ".join(["word"] * 40)), max_chars=9_000, size_fn=estimate_segment_tokens)
    ko = chunk_transcript(_segments(1500, "오늘 회의에서 이야기한 내용을 정리해 볼게요 " * 4), max_chars=9_000, size_fn=estimate_segment_tokens)
    assert len(en) > 1 and len(ko) > 1
    assert all(sum(estimate_segment_tokens(s) for s in c.segments) <= 9_000 for c in en + ko)
    assert max(c.char_count for c in ko) < max(c.char_count for c in en)


def test_without_size_fn_the_cap_is_characters():
    chunks = chunk_transcript(_segments(1500, " ".join(["word"] * 40)), max_chars=12_000)
    assert all(c.char_count <= 12_000 for c in chunks)


def test_invalid_json_is_asked_again_then_parsed(monkeypatch):
    answers = iter(['{"title": "x", "summary": ', '{"title": "ok", "summary": "s"}'])

    def fake_call(client, model, user_message, max_retries):
        return next(answers), 10, 5

    monkeypatch.setattr(analyzer, "_call_openai_compatible", fake_call)
    result = analyzer.analyze_transcript("hello", provider="openai", base_url="http://localhost:11435/v1",
                                         api_key="local", client=SimpleNamespace())
    assert result == {"title": "ok", "summary": "s"}


def test_invalid_json_every_time_gives_none(monkeypatch):
    calls = []

    def fake_call(client, model, user_message, max_retries):
        calls.append(1)
        return "not json", 10, 5

    monkeypatch.setattr(analyzer, "_call_openai_compatible", fake_call)
    assert analyzer.analyze_transcript("hello", provider="openai", base_url="http://localhost:11435/v1",
                                       api_key="local", client=SimpleNamespace()) is None
    assert len(calls) == analyzer.JSON_ATTEMPTS
