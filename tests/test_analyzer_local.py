"""Local-AI behaviour of the OpenAI-compatible analyzer path.

Owner decision 2026-10-12: transcripts are analysed by on-box models only. Local
reasoning models (Qwen on vLLM/ollama) must have "thinking" disabled or they burn the
whole max_tokens budget and return an empty answer; hosted APIs must NOT receive the
extra field (they reject unknown body keys).
"""
from types import SimpleNamespace

from audioscript.llm.analyzer import (
    _call_openai_compatible,
    _local_extra_body,
    _strip_thinking,
)


class _FakeClient:
    def __init__(self, base_url: str, content):
        self.base_url = base_url
        self.kwargs = None
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def test_loopback_endpoints_disable_thinking():
    for url in ("http://localhost:11435/v1/", "http://127.0.0.1:11434/v1", "http://[::1]:8000/v1"):
        assert _local_extra_body(url) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_hosted_endpoints_get_no_extra_body():
    for url in ("https://api.openai.com/v1", "https://integrate.api.nvidia.com/v1", "", "not a url"):
        assert _local_extra_body(url) is None


def test_call_sends_extra_body_only_to_local_server():
    local = _FakeClient("http://localhost:11435/v1/", '{"ok": true}')
    _call_openai_compatible(local, "Qwen/Qwen3.5-4B", "hi")
    assert local.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    hosted = _FakeClient("https://api.openai.com/v1", '{"ok": true}')
    _call_openai_compatible(hosted, "gpt-x", "hi")
    assert "extra_body" not in hosted.kwargs


def test_think_block_is_stripped_and_none_content_is_safe():
    c = _FakeClient("http://localhost:11435/v1/", "<think>\nreasoning...\n</think>\n{\"a\": 1}")
    text, tin, tout = _call_openai_compatible(c, "m", "hi")
    assert text == '{"a": 1}' and (tin, tout) == (10, 5)
    assert _strip_thinking('{"a": 1}') == '{"a": 1}'

    empty = _FakeClient("http://localhost:11435/v1/", None)  # reasoning-only reply
    assert _call_openai_compatible(empty, "m", "hi")[0] == ""
