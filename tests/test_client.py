"""Client behaviour with the page replaced by a fake.

No Playwright, no network: the browser is the thing under test's *seam*, and
everything around it (payload, token header, retries, error mapping) is checked
here. ``test_live.py`` covers the real browser when one is available.
"""

import asyncio

import pytest

from theoldllm.browser import CHAT_PATH, AsyncTheOldLLM
from theoldllm.client import TheOldLLM
from theoldllm.exceptions import APIError, AccessDeniedError, TurnstileError
from tests.bodies import ACCESS_DENIED_BODY, BOT_FAILED_BODY, OK_BODY


class FakePage:
    def __init__(self, bodies, *, mint_error=None, first_evaluate_raises=False):
        self.bodies = list(bodies)
        self.mint_error = mint_error
        self.first_evaluate_raises = first_evaluate_raises
        self.tokens_minted = 0
        self.sent = []
        self.settles = 0
        self.reloads = 0

    async def wait_for_load_state(self, *args, **kwargs):
        self.settles += 1

    async def wait_for_function(self, *args, **kwargs):
        return True

    async def reload(self, *args, **kwargs):
        self.reloads += 1

    async def evaluate(self, script, arg=None):
        if "turnstile.render" in script:
            if self.mint_error:
                raise RuntimeError(self.mint_error)
            self.tokens_minted += 1
            return f"token-{self.tokens_minted}"
        if "x-turnstile-token" in script:
            if self.first_evaluate_raises:
                self.first_evaluate_raises = False
                raise RuntimeError("Execution context was destroyed, most likely because of a navigation.")
            self.sent.append(arg)
            body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
            return {"status": 200, "text": body}
        return {"readyState": "complete", "turnstile": "function", "title": "TheOldLLM"}


def make_client(page, **options):
    client = AsyncTheOldLLM(settle_delay=0, **options)
    client._page = page
    client._started = True
    return client


def run(coro):
    return asyncio.run(coro)


def test_each_request_carries_its_own_turnstile_token():
    page = FakePage([OK_BODY])
    client = make_client(page)

    async def go():
        assert await client.chat("gpt-6-luna", [{"role": "user", "content": "one"}]) == "hello"
        assert await client.chat("gpt-6-luna", [{"role": "user", "content": "two"}]) == "hello"

    run(go())
    assert page.tokens_minted == 2
    assert [call["token"] for call in page.sent] == ["token-1", "token-2"]
    assert {call["path"] for call in page.sent} == {CHAT_PATH}


def test_payload_reaches_the_page_with_the_resolved_model():
    page = FakePage([OK_BODY])
    client = make_client(page)
    run(client.chat("claude-opus-4-7", [{"role": "user", "content": "hi"}], system=None, max_tokens=64))

    payload = page.sent[0]["payload"]
    assert payload["model"] == "claude-opus-4-7"
    assert payload["provider"] == "anthropic"
    assert payload["max_tokens"] == 64
    assert payload["stream"] is True


def test_bot_check_failure_is_retried_with_a_fresh_token():
    page = FakePage([BOT_FAILED_BODY, OK_BODY])
    client = make_client(page)

    assert run(client.chat("gpt-6-luna", [{"role": "user", "content": "hi"}])) == "hello"
    assert page.tokens_minted == 2
    assert [call["token"] for call in page.sent] == ["token-1", "token-2"]


def test_repeated_bot_check_failures_reload_the_page():
    page = FakePage([BOT_FAILED_BODY, BOT_FAILED_BODY, OK_BODY])
    client = make_client(page)

    assert run(client.chat("gpt-6-luna", [{"role": "user", "content": "hi"}])) == "hello"
    assert page.reloads == 1  # the second rejection rebuilds the human-check runtime
    assert page.tokens_minted == 3


def test_navigation_during_the_call_is_retried():
    page = FakePage([OK_BODY], first_evaluate_raises=True)
    client = make_client(page)

    assert run(client.chat("gpt-6-luna", [{"role": "user", "content": "hi"}])) == "hello"
    assert page.settles >= 2  # the page was re-settled before the retry


def test_access_denied_for_a_verified_model_is_explained():
    page = FakePage([ACCESS_DENIED_BODY])
    client = make_client(page)

    with pytest.raises(AccessDeniedError) as info:
        run(client.chat("gpt-6-sol", [{"role": "user", "content": "hi"}]))
    assert info.value.model == "gpt-6-sol"
    assert "requiresVerifiedAccount" in str(info.value)
    assert "gpt-6-luna" in str(info.value)


def test_access_denied_for_an_anonymous_model_is_left_alone():
    page = FakePage([ACCESS_DENIED_BODY])
    client = make_client(page)

    with pytest.raises(AccessDeniedError) as info:
        run(client.chat("gpt-6-luna", [{"role": "user", "content": "hi"}]))
    assert "requiresVerifiedAccount" not in str(info.value)


def test_non_200_response_raises_api_error():
    class FailingPage(FakePage):
        async def evaluate(self, script, arg=None):
            if "x-turnstile-token" in script:
                return {"status": 429, "text": "<html>Vercel Security Checkpoint</html>"}
            return await super().evaluate(script, arg)

    client = make_client(FailingPage([""]))
    with pytest.raises(APIError) as info:
        run(client.chat("gpt-6-luna", [{"role": "user", "content": "hi"}]))
    assert info.value.status_code == 429


def test_turnstile_timeout_surfaces_as_a_turnstile_error():
    page = FakePage([OK_BODY], mint_error="turnstile timeout after 30000ms")
    client = make_client(page)

    with pytest.raises(TurnstileError) as info:
        run(client.chat("gpt-6-luna", [{"role": "user", "content": "hi"}]))
    assert info.value.timed_out
    assert "headless=False" in str(info.value)
    assert page.sent == []


class FakeTransport:
    """Stands in for the browser in ``AsyncTheOldLLM(transport=...)``."""

    def __init__(self, body=OK_BODY):
        self.body = body
        self.calls = []

    async def request(self, payload):
        self.calls.append(dict(payload))
        return {"status": 200, "text": self.body}


def test_transport_hook_skips_the_browser_entirely():
    transport = FakeTransport()

    async def go():
        async with AsyncTheOldLLM(transport=transport) as client:
            assert client.page is None
            return await client.chat("gpt-6-luna", [{"role": "user", "content": "hi"}])

    assert run(go()) == "hello"
    assert transport.calls[0]["model"] == "gpt-6-luna"


def test_streaming_yields_each_delta():
    transport = FakeTransport()

    async def go():
        async with AsyncTheOldLLM(transport=transport) as client:
            chunks = [chunk async for chunk in client.chat_stream("gpt-6-luna", [{"role": "user", "content": "hi"}])]
            return [(c.content, c.is_done) for c in chunks]

    assert run(go()) == [("he", False), ("llo", False), ("", True)]


def test_sync_client_bridges_the_background_loop():
    transport = FakeTransport()
    with TheOldLLM(transport=transport) as client:
        assert client.chat("gpt-6-luna", [{"role": "user", "content": "hi"}]) == "hello"
        streamed = "".join(c.content for c in client.chat_stream("gpt-6-luna", [{"role": "user", "content": "hi"}]))
    assert streamed == "hello"
    assert len(transport.calls) == 2


def test_concurrent_calls_are_serialised_on_one_page():
    page = FakePage([OK_BODY, OK_BODY])
    client = make_client(page)

    async def go():
        return await asyncio.gather(
            client.chat("gpt-6-luna", [{"role": "user", "content": "a"}]),
            client.chat("gpt-6-luna", [{"role": "user", "content": "b"}]),
        )

    assert run(go()) == ["hello", "hello"]
    # Two calls, two tokens, and the page was never asked twice at once.
    assert page.tokens_minted == 2
    assert len(page.sent) == 2
