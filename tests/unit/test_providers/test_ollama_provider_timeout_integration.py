"""Retry coverage for Ollama's native ``/api/chat`` transport."""

import httpx
import pytest
import respx
from tenacity import wait_none

from repowise.core.providers.llm.base import ProviderError
from repowise.core.providers.llm.ollama import OllamaProvider


@pytest.mark.asyncio
@respx.mock
async def test_ollama_retries_native_transport_errors() -> None:
    provider = OllamaProvider(model="test-model", base_url="http://localhost:9999")
    provider._generate_with_retry.retry.wait = wait_none()
    route = respx.post("http://localhost:9999/api/chat").mock(
        side_effect=httpx.ConnectTimeout("timed out")
    )

    with pytest.raises(ProviderError, match="timed out"):
        await provider.generate("", "test", max_tokens=10)

    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_ollama_succeeds_after_native_transport_retry() -> None:
    provider = OllamaProvider(model="test-model", base_url="http://localhost:9999")
    provider._generate_with_retry.retry.wait = wait_none()
    route = respx.post("http://localhost:9999/api/chat").mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.Response(
                200,
                json={
                    "message": {"content": "Success"},
                    "prompt_eval_count": 5,
                    "eval_count": 1,
                    "done_reason": "stop",
                },
            ),
        ]
    )

    result = await provider.generate("", "test", max_tokens=10)

    assert route.call_count == 2
    assert result.content == "Success"
    assert result.input_tokens == 5
    assert result.output_tokens == 1


@pytest.mark.asyncio
@respx.mock
async def test_ollama_does_not_retry_invalid_request() -> None:
    provider = OllamaProvider(model="test-model", base_url="http://localhost:9999")
    provider._generate_with_retry.retry.wait = wait_none()
    route = respx.post("http://localhost:9999/api/chat").mock(
        return_value=httpx.Response(400, json={"error": "invalid option"})
    )

    with pytest.raises(ProviderError):
        await provider.generate("", "test", max_tokens=10)

    assert route.call_count == 1
