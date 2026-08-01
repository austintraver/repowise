"""Unit tests for OllamaProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

pytest.importorskip("openai", reason="openai SDK not installed")

from repowise.core.providers.llm.base import ProviderError, SamplingParameters
from repowise.core.providers.llm.ollama import OllamaProvider


def test_supported_reasoning_modes_are_auto_and_off():
    provider = OllamaProvider(model="test")

    assert provider.supported_reasoning_modes() == ("auto", "off")


def test_available_model_options_reads_local_tags(monkeypatch):
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "models": [
                    {
                        "name": "llama3.2:latest",
                        "details": {
                            "family": "llama",
                            "parameter_size": "3B",
                        },
                    },
                    {"model": "qwen2.5-coder:7b"},
                ]
            }

    captured: dict[str, object] = {}

    def fake_get(url, *, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("httpx.get", fake_get)

    options = OllamaProvider(base_url="http://localhost:11434").available_model_options()

    assert captured["url"] == "http://localhost:11434/api/tags"
    model_names = [option.model for option in options]
    assert model_names == ["llama3.2:latest", "qwen2.5-coder:7b"]
    llama = options[0]
    assert llama.source == "local"
    assert llama.notes == "llama, 3B"
    assert llama.reasoning_modes == ("auto", "off")


def _make_mock_chat_response(text: str = "# Doc\nContent.") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 120
    usage.completion_tokens = 60

    choice = MagicMock()
    choice.message.content = text

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@respx.mock
async def test_generate_auto_uses_native_chat_and_forwards_sampling():
    provider = OllamaProvider(model="test")
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"content": "# Doc\nContent."},
                "prompt_eval_count": 120,
                "eval_count": 60,
                "done_reason": "stop",
            },
        )
    )

    result = await provider.generate(
        "system",
        "user",
        reasoning="auto",
        sampling=SamplingParameters(temperature=1.0, top_p=0.95, top_k=64),
    )

    payload = route.calls[0].request.read()
    assert b'"options":{"num_predict":4096,"temperature":1.0,"top_p":0.95,"top_k":64}' in payload
    assert b'"think"' not in payload
    assert route.calls[0].request.extensions["timeout"] == {
        "connect": 5.0,
        "read": 600.0,
        "write": 600.0,
        "pool": 600.0,
    }
    assert result.usage["outbound_sampling"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }


@respx.mock
async def test_generate_off_disables_reasoning():
    provider = OllamaProvider(model="test")
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"content": "ok"}, "done_reason": "stop"},
        )
    )

    await provider.generate("system", "user", reasoning="off")

    assert b'"think":false' in route.calls[0].request.read()


@respx.mock
async def test_generate_accepts_legacy_openai_base_url():
    provider = OllamaProvider(model="test", base_url="http://localhost:11434/v1")
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"content": "ok"}, "done_reason": "stop"},
        )
    )

    await provider.generate("system", "user")

    assert route.called


@pytest.mark.parametrize("reasoning", ["none", "minimal", "low"])
async def test_generate_rejects_unsupported_reasoning_modes(reasoning):
    provider = OllamaProvider(model="test")
    provider._client.chat.completions.create = AsyncMock()

    with pytest.raises(ProviderError, match=f"reasoning='{reasoning}' is not supported"):
        await provider.generate("system", "user", reasoning=reasoning)

    provider._client.chat.completions.create.assert_not_called()
