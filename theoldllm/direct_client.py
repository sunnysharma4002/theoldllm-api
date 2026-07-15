from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Optional

from curl_cffi import requests as curl_requests
from curl_cffi.requests import AsyncSession

from .client import BASE_URL, API_ENDPOINTS
from .exceptions import APIError, RateLimitError, StreamError
from .models import Model, Models, Provider
from .streaming import ChatCompletionChunk, parse_sse_line


class DirectTheOldLLM:
    """Lightweight client that uses curl_cffi TLS impersonation to bypass Vercel WAF.

    No browser needed — curl_cffi mimics Chrome's TLS fingerprint, which is
    enough to pass Vercel's security checkpoint.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = 120,
        impersonate: str = "chrome120",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.impersonate = impersonate
        self._session = curl_requests.Session()
        self._default_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def _get_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/event-stream",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        }

    def _resolve_provider(self, model: str | Model) -> tuple[str, Provider]:
        if isinstance(model, Model):
            return model.id, model.provider
        m = Models.by_id(model)
        return (m.id, m.provider) if m else (model, Provider.CHATGPT)

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

    def chat_stream(
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
        model_id, provider = self._resolve_provider(model)
        # Both providers work through /api/chatgpt
        endpoint = f"{self.base_url}/api/chatgpt"

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

        resp = self._session.post(
            endpoint,
            headers=self._get_headers(),
            json=payload,
            impersonate=self.impersonate,
            stream=True,
            timeout=self.timeout,
        )

        if not resp.ok:
            body = resp.text
            if resp.status_code == 429:
                raise RateLimitError(body, status_code=resp.status_code, body=body)
            raise APIError(body, status_code=resp.status_code, body=body)

        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            chunk = parse_sse_line(line)
            if chunk is not None:
                yield chunk
                if chunk.is_done:
                    return

    def chat(
        self,
        model: str | Model,
        messages: list[dict],
        **kwargs,
    ) -> str:
        parts: list[str] = []
        for chunk in self.chat_stream(model=model, messages=messages, **kwargs):
            if chunk.content:
                parts.append(chunk.content)
        return "".join(parts)


class AsyncDirectTheOldLLM:
    """Async version using curl_cffi's AsyncSession."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = 120,
        impersonate: str = "chrome120",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.impersonate = impersonate
        self._default_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def _resolve_provider(self, model: str | Model) -> tuple[str, Provider]:
        if isinstance(model, Model):
            return model.id, model.provider
        m = Models.by_id(model)
        return (m.id, m.provider) if m else (model, Provider.CHATGPT)

    def _build_payload(self, **kwargs) -> dict:
        p = self._default_params.copy()
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            if kwargs.get(k) is not None:
                p[k] = kwargs[k]

        payload: dict[str, Any] = {
            "model": kwargs["model_id"],
            "messages": kwargs["messages"],
            "stream": kwargs.get("stream", True),
            **p,
        }

        m = Models.by_id(kwargs["model_id"])
        default_max = m.max_tokens if m else 8192
        payload["max_tokens"] = kwargs.get("max_tokens") or default_max

        if kwargs.get("reasoning_effort"):
            payload["reasoning_effort"] = kwargs["reasoning_effort"]
        if kwargs.get("web_search"):
            payload["webSearch"] = True

        provider = kwargs.get("provider")
        if provider == Provider.AICHAT:
            up = kwargs.get("upstream_provider") or (m and m.upstream_provider)
            if up:
                payload["provider"] = up

        return payload

    async def chat_stream(
        self,
        model: str | Model,
        messages: list[dict],
        **kwargs,
    ):
        model_id, provider = self._resolve_provider(model)
        endpoint_path = API_ENDPOINTS.get(provider, "/api/chatgpt")
        endpoint = f"{self.base_url}{endpoint_path}"

        payload = self._build_payload(
            model_id=model_id,
            messages=messages,
            provider=provider,
            stream=True,
            **kwargs,
        )

        headers = {
            "Content-Type": "application/json",
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/event-stream",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        }

        async with AsyncSession() as session:
            async with session.stream(
                "POST",
                endpoint,
                headers=headers,
                json=payload,
                impersonate=self.impersonate,
                timeout=self.timeout,
            ) as resp:
                if not resp.ok:
                    body = await resp.atext()
                    if resp.status_code == 429:
                        raise RateLimitError(body, status_code=resp.status_code, body=body)
                    raise APIError(body, status_code=resp.status_code, body=body)

                buffer = ""
                async for raw_chunk in resp.aiter_lines():
                    buffer += raw_chunk + "\n"
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        chunk = parse_sse_line(line)
                        if chunk is not None:
                            yield chunk
                            if chunk.is_done:
                                return

    async def chat(self, model, messages, **kwargs) -> str:
        parts = []
        async for chunk in self.chat_stream(model=model, messages=messages, **kwargs):
            if chunk.content:
                parts.append(chunk.content)
        return "".join(parts)
