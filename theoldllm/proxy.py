"""OpenAI-compatible HTTP facade over the browser client.

``POST /v1/chat/completions`` and ``GET /v1/models`` — enough for
``@ai-sdk/openai-compatible``, opencode, LangChain, or the ``openai`` Python
package pointed at ``http://127.0.0.1:11434/v1``.

The facade is deliberately dumb: the browser client already serialises calls
(Turnstile tokens are page-scoped), so requests queue rather than run in
parallel. Latency is the upstream model's, plus a browser round trip.

Note: this is a *local* convenience. The upstream still sees one page and one
Turnstile solve per request, so the same "keep volume low" caveat applies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Mapping, Optional, Sequence

from .browser import BASE_URL, AsyncTheOldLLM
from .exceptions import (
    AccessDeniedError,
    APIError,
    BotCheckError,
    BrowserError,
    TurnstileError,
    UnknownModelError,
)
from .models import Models
from .payload import build_payload

LOG = logging.getLogger("theoldllm.proxy")

DEFAULT_MODEL = "gpt-6-luna"


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _sse(payload: Mapping[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _error_body(code: str, message: str, kind: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": code}}


def _status_for(error: Exception) -> tuple[int, str, str]:
    if isinstance(error, UnknownModelError):
        return 400, "model_not_found", "invalid_request_error"
    if isinstance(error, AccessDeniedError):
        return 403, "access_denied", "permission_error"
    if isinstance(error, BotCheckError):
        return 502, "bot_check_failed", "upstream_error"
    if isinstance(error, TurnstileError):
        return 503, "turnstile_unavailable", "upstream_error"
    if isinstance(error, BrowserError):
        return 503, "browser_unavailable", "upstream_error"
    if isinstance(error, APIError):
        return 502, "upstream_error", "upstream_error"
    return 500, "internal_error", "server_error"


class OpenAICompatProxy:
    """aiohttp app exposing the browser client as an OpenAI-compatible API."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        *,
        client: Optional[AsyncTheOldLLM] = None,
        client_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self._client = client
        self._owns_client = client is None
        self._client_options = client_options or {}
        self._runner: Any = None
        self._site: Any = None
        self._client_lock: Optional[asyncio.Lock] = None
        self._browser_error: Optional[str] = None

    # ----------------------------------------------------------------- client

    async def client(self) -> AsyncTheOldLLM:
        if self._client is None:
            if self._client_lock is None:
                self._client_lock = asyncio.Lock()
            async with self._client_lock:
                # Re-check under the lock: a burst of requests must not each
                # launch their own browser.
                if self._client is None:
                    client = AsyncTheOldLLM(**self._client_options)
                    try:
                        await client.start()
                    except Exception as error:
                        self._browser_error = f"{type(error).__name__}: {error}"
                        raise
                    self._browser_error = None
                    self._client = client
        return self._client

    # --------------------------------------------------------------- handlers

    async def _chat(self, request: Any) -> Any:
        from aiohttp import web

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(_error_body("invalid_json", "body is not JSON", "invalid_request_error"), status=400)

        model = body.get("model") or DEFAULT_MODEL
        messages: Sequence[Mapping[str, Any]] = body.get("messages") or []
        stream = bool(body.get("stream"))
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        reasoning_effort = body.get("reasoning_effort")

        try:
            payload = build_payload(
                model,
                messages,
                # The caller supplies its own system prompt; injecting the
                # site's default here would silently rewrite agent prompts.
                system=None,
                temperature=float(body.get("temperature", 0.7)),
                top_p=float(body.get("top_p", 0.9)),
                frequency_penalty=float(body.get("frequency_penalty", 0.0)),
                presence_penalty=float(body.get("presence_penalty", 0.0)),
                max_tokens=int(max_tokens) if max_tokens else None,
                reasoning_effort=reasoning_effort,
            )
            client = await self.client()
        except Exception as error:  # noqa: BLE001 - mapped below
            status, code, kind = _status_for(error)
            LOG.warning("rejected request: %s", error)
            return web.json_response(_error_body(code, str(error), kind), status=status)

        model_id = str(payload["model"])
        created = int(time.time())
        completion_id = _completion_id()

        if stream:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
            await response.prepare(request)
            try:
                first = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                await response.write(_sse(first))
                async for chunk in client.chat_stream(model_id, messages, system=None, max_tokens=max_tokens, reasoning_effort=reasoning_effort):
                    delta: dict[str, Any] = {}
                    if chunk.content:
                        delta["content"] = chunk.content
                    if chunk.reasoning_content:
                        delta["reasoning_content"] = chunk.reasoning_content
                    if delta or chunk.finish_reason:
                        await response.write(
                            _sse(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": chunk.finish_reason,
                                        }
                                    ],
                                }
                            )
                        )
                await response.write(
                    _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                    )
                )
                await response.write(b"data: [DONE]\n\n")
            except Exception as error:  # noqa: BLE001 - headers already sent
                LOG.exception("stream failed")
                code, kind = _status_for(error)[1:]
                await response.write(_sse(_error_body(code, str(error), kind)))
            finally:
                await response.write_eof()
            return response

        try:
            parts: list[str] = []
            async for chunk in client.chat_stream(
                model_id,
                messages,
                system=None,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            ):
                parts.append(chunk.content)
            text = "".join(parts)
        except Exception as error:  # noqa: BLE001 - mapped below
            status, code, kind = _status_for(error)
            LOG.warning("chat failed: %s", error)
            return web.json_response(_error_body(code, str(error), kind), status=status)

        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    async def _models(self, request: Any) -> Any:
        from aiohttp import web

        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model.id,
                        "object": "model",
                        "created": 0,
                        "owned_by": model.provider.value,
                        "root": model.id,
                    }
                    for model in Models.picker()
                ],
            }
        )

    async def _health(self, request: Any) -> Any:
        from aiohttp import web

        page = self._client.page if self._client else None
        body: dict[str, Any] = {
            "status": "ok" if page is not None else ("error" if self._browser_error else "starting"),
            "browser": bool(page),
            "models": len(Models.picker()),
            "upstream": BASE_URL,
        }
        if self._browser_error:
            body["browser_error"] = self._browser_error

        # /health?deep=1 runs a real completion, which is the only reliable way
        # to tell "container is up" from "Cloudflare will not let us talk".
        if request.query.get("deep") in {"1", "true", "yes"}:
            if self._client is None:
                try:
                    await self.client()
                except Exception as error:  # noqa: BLE001 - reported in the body
                    body["deep"] = {"ok": False, "error": str(error), "code": "browser_unavailable"}
                    return web.json_response(body, status=503)
            try:
                reply = await asyncio.wait_for(
                    self._client.chat(
                        DEFAULT_MODEL,
                        [{"role": "user", "content": "Reply with exactly: ok"}],
                        max_tokens=16,
                    ),
                    timeout=float(request.query.get("timeout", "90")),
                )
                body["deep"] = {"ok": True, "model": DEFAULT_MODEL, "reply": reply}
            except asyncio.TimeoutError:
                body["status"] = "error"
                body["deep"] = {"ok": False, "code": "timeout", "error": "completion timed out"}
            except Exception as error:  # noqa: BLE001 - reported in the body
                body["status"] = "error"
                body["deep"] = {"ok": False, "code": _status_for(error)[1], "error": str(error)}
        return web.json_response(body)

    # -------------------------------------------------------------- lifecycle

    def app(self) -> Any:
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/health", self._health)
        app.router.add_route("OPTIONS", "/v1/chat/completions", self._cors)
        app.router.add_route("OPTIONS", "/v1/models", self._cors)
        return app

    async def _cors(self, request: Any) -> Any:
        from aiohttp import web

        return web.Response(
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )

    async def start(self) -> None:
        from aiohttp import web

        runner = web.AppRunner(self.app())
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self._runner, self._site = runner, site
        LOG.info("listening on http://%s:%d/v1", self.host, self.port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = self._site = None
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def serve_forever(self) -> None:
        await self.start()
        try:
            try:
                await self.client()  # warm the browser up
            except Exception as error:  # noqa: BLE001 - keep serving so /health can explain
                # Crashing here would put the container in a restart loop, which
                # hides the real reason. Stay up and report it instead.
                LOG.error("browser failed to start: %s", error)
                LOG.error("the facade is still listening; check /health?deep=1")
            await asyncio.Event().wait()
        finally:
            await self.stop()


async def serve(host: str = "127.0.0.1", port: int = 11434, **client_options: Any) -> None:
    """Run the facade until interrupted."""
    await OpenAICompatProxy(host=host, port=port, client_options=client_options).serve_forever()


__all__ = ["DEFAULT_MODEL", "OpenAICompatProxy", "serve"]
