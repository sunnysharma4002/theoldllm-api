"""The OpenAI-compatible facade, exercised without a browser."""

import asyncio
import json

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from theoldllm.browser import AsyncTheOldLLM  # noqa: E402
from theoldllm.exceptions import AccessDeniedError  # noqa: E402
from theoldllm.proxy import OpenAICompatProxy  # noqa: E402
from tests.bodies import OK_BODY  # noqa: E402


class Transport:
    def __init__(self, body=OK_BODY, error=None):
        self.body = body
        self.error = error
        self.calls = []

    async def request(self, payload):
        self.calls.append(dict(payload))
        if self.error:
            raise self.error
        return {"status": 200, "text": self.body}


def with_proxy(transport, coro_factory):
    async def go():
        client = AsyncTheOldLLM(transport=transport)
        proxy = OpenAICompatProxy(client=client)
        async with TestClient(TestServer(proxy.app())) as http:
            return await coro_factory(http, client)

    return asyncio.run(go())


def test_models_endpoint_lists_the_picker_catalogue():
    async def scenario(http, _client):
        async with http.get("/v1/models") as response:
            assert response.status == 200
            return await response.json()

    body = with_proxy(Transport(), scenario)
    ids = [m["id"] for m in body["data"]]
    assert "gpt-6-luna" in ids
    assert "gpt-5-mini-aichat" not in ids
    assert body["object"] == "list"


def test_non_streaming_completion_has_openai_shape():
    transport = Transport()

    async def scenario(http, _client):
        async with http.post(
            "/v1/chat/completions",
            json={"model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            assert response.status == 200
            return await response.json()

    body = with_proxy(transport, scenario)
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-6-luna"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert body["choices"][0]["finish_reason"] == "stop"

    # The caller's own system prompt is not rewritten with the site's default.
    payload = transport.calls[0]
    assert not any(m["role"] == "system" for m in payload["messages"])


def test_streaming_completion_emits_deltas_and_done():
    async def scenario(http, _client):
        async with http.post(
            "/v1/chat/completions",
            json={"model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/event-stream")
            return await response.text()

    text = with_proxy(Transport(), scenario)
    events = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    deltas = [json.loads(e) for e in events[:-1]]
    contents = [d["choices"][0]["delta"]["content"] for d in deltas if "content" in d["choices"][0]["delta"]]
    assert contents == ["he", "llo"]
    assert deltas[-1]["choices"][0]["finish_reason"] == "stop"


def test_access_denied_maps_to_403_with_a_useful_code():
    transport = Transport(error=AccessDeniedError("Access denied", status_code=403, code="ACCESS_DENIED"))

    async def scenario(http, _client):
        async with http.post(
            "/v1/chat/completions",
            json={"model": "gpt-6-sol", "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            return response.status, await response.json()

    status, body = with_proxy(transport, scenario)
    assert status == 403
    assert body["error"]["code"] == "access_denied"


def test_unknown_model_maps_to_400():
    async def scenario(http, _client):
        async with http.post(
            "/v1/chat/completions",
            json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            return response.status, await response.json()

    status, body = with_proxy(Transport(), scenario)
    assert status == 400
    assert body["error"]["code"] == "model_not_found"


def test_health_reports_browser_state():
    async def scenario(http, _client):
        async with http.get("/health") as response:
            return await response.json()

    body = with_proxy(Transport(), scenario)
    assert body["status"] == "starting"  # transport-only client has no page
    assert body["upstream"].startswith("https://theoldllm")
