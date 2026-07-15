from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncGenerator, Callable, Optional

from .client import BASE_URL
from .exceptions import APIError, RateLimitError, StreamError
from .models import Model, Models, Provider
from .streaming import ChatCompletionChunk, parse_sse_line


class PlaywrightTheOldLLM:
    """Browser-based client that solves Vercel's Turnstile challenge automatically.

    Uses Playwright to launch a real Chromium browser, solve the Vercel
    security checkpoint (Turnstile), and then intercept API calls.

    Requires playwright and a chromium browser:
        pip install playwright
        playwright install chromium
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        headless: bool = False,
        storage_path: Optional[str] = None,
        proxy: Optional[dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.headless = headless
        self.storage_path = storage_path
        self.proxy = proxy
        self._browser = None
        self._context = None
        self._page = None
        self._session_ready = asyncio.Event()
        self._default_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    async def _ensure_session(self):
        if self._page and self._session_ready.is_set():
            return

        from playwright.async_api import async_playwright

        if self.storage_path:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        p = await async_playwright().start()
        launch_args = {
            "headless": self.headless,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--single-process",
            ],
        }
        if self.proxy:
            launch_args["proxy"] = self.proxy

        self._browser = await p.chromium.launch(**launch_args)

        ctx_args = {
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "locale": "en-US",
            "viewport": {"width": 1280, "height": 720},
        }
        if self.storage_path:
            ctx_args["storage_state"] = self.storage_path

        self._context = await self._browser.new_context(**ctx_args)
        self._page = await self._context.new_page()

        # Listen for all API responses
        self._api_responses: dict[str, asyncio.Future] = {}

        async def handle_response(response):
            url = response.url
            if self.base_url in url and ("/api/chatgpt" in url or "/api/aichat" in url):
                request = response.request
                req_id = id(request)
                if req_id in self._api_responses:
                    fut = self._api_responses.pop(req_id)
                    if not fut.done():
                        fut.set_result(response)

        self._page.on("response", handle_response)

        # Navigate to the app
        await self._page.goto(self.base_url, wait_until="networkidle")

        # Wait for Turnstile to pass and page to fully load
        try:
            await self._page.wait_for_selector("#root", timeout=30000)
        except Exception:
            pass

        # Save cookies for next time
        if self.storage_path:
            await self._context.storage_state(path=self.storage_path)

        self._session_ready.set()

    async def _make_api_request(
        self,
        endpoint: str,
        payload: dict,
        timeout: int = 120,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        await self._ensure_session()

        script = f"""
        (async () => {{
            const resp = await fetch('{endpoint}', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: {json.dumps(json.dumps(payload))},
                credentials: 'include',
            }});
            if (!resp.ok) {{
                const text = await resp.text();
                return {{ error: true, status: resp.status, body: text }};
            }}
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            const chunks = [];
            while (true) {{
                const {{ done, value }} = await reader.read();
                if (done) break;
                chunks.push(decoder.decode(value, {{ stream: true }}));
            }}
            return {{ error: false, data: chunks.join('') }};
        }})()
        """

        result = await self._page.evaluate(script)

        if result.get("error"):
            status = result["status"]
            body = result.get("body", "")
            if status == 429:
                raise RateLimitError(body, status_code=status, body=body)
            raise APIError(body, status_code=status, body=body)

        raw = result.get("data", "")
        buffer = ""
        for char in raw:
            buffer += char
            if char == "\n":
                line = buffer.strip()
                buffer = ""
                if not line:
                    continue
                chunk = parse_sse_line(line)
                if chunk is not None:
                    yield chunk
                    if chunk.is_done:
                        return

    def _build_payload(
        self,
        model_id: str,
        messages: list[dict],
        provider: Provider,
        stream: bool = True,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        web_search: bool = False,
        upstream_provider: Optional[str] = None,
        **kwargs,
    ) -> dict:
        p = self._default_params.copy()
        if temperature is not None:
            p["temperature"] = temperature
        if top_p is not None:
            p["top_p"] = top_p
        if frequency_penalty is not None:
            p["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            p["presence_penalty"] = presence_penalty

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
            **p,
        }

        m = Models.by_id(model_id)
        default_max = m.max_tokens if m else 8192
        payload["max_tokens"] = max_tokens or default_max

        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if web_search:
            payload["webSearch"] = True

        if provider == Provider.AICHAT:
            up = upstream_provider or (m and m.upstream_provider)
            if up:
                payload["provider"] = up

        payload.update(kwargs)
        return payload

    async def chat_stream(
        self,
        model: str | Model,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        web_search: bool = False,
        **kwargs,
    ):
        model_id, provider = (
            (model.id, model.provider) if isinstance(model, Model)
            else (model, Provider.CHATGPT)
        )
        m = Models.by_id(model_id)
        if m:
            provider = m.provider

        from .client import API_ENDPOINTS
        endpoint_path = API_ENDPOINTS.get(provider, "/api/chatgpt")
        endpoint = f"{self.base_url}{endpoint_path}"

        payload = self._build_payload(
            model_id=model_id,
            messages=messages,
            provider=provider,
            stream=True,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            web_search=web_search,
            upstream_provider=kwargs.pop("upstream_provider", None),
            **kwargs,
        )

        async for chunk in self._make_api_request(endpoint, payload):
            yield chunk

    async def chat(
        self,
        model: str | Model,
        messages: list[dict],
        **kwargs,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.chat_stream(model=model, messages=messages, **kwargs):
            if chunk.content:
                parts.append(chunk.content)
        return "".join(parts)

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._context = None
            self._page = None
            self._session_ready.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    @staticmethod
    def user_message(content: str) -> dict:
        return {"role": "user", "content": content}

    @staticmethod
    def assistant_message(content: str) -> dict:
        return {"role": "assistant", "content": content}

    @staticmethod
    def system_message(content: str) -> dict:
        return {"role": "system", "content": content}
