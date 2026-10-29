"""SSE decoding for ``POST /api/aichat``.

The route always answers ``text/event-stream`` — even when the request body has
``"stream": false`` — and reports failures as an HTTP 200 with a
``stream_error`` event, so HTTP status codes alone are useless. A complete body
looks like this::

    : connected

    data: {"choices":[{"delta":{"content":"he"}}]}

    data: {"choices":[{"delta":{"content":"llo"}}]}

    data: [DONE]

Failure bodies look like::

    data: {"stream_error":{"error":"Access denied","code":"BOT_CHECK_FAILED","status":403}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Optional

from .exceptions import (
    AccessDeniedError,
    APIError,
    BotCheckError,
    RateLimitError,
    StreamError,
)

_DONE = "[DONE]"


@dataclass
class ChatCompletionChunk:
    """One decoded delta.

    ``content`` is the visible text; ``reasoning_content`` carries the thinking
    stream when a reasoning model emits one.
    """

    content: str = ""
    reasoning_content: Optional[str] = None
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[dict] = None
    raw: Optional[dict] = None

    @property
    def is_done(self) -> bool:
        return self.finish_reason is not None


def _error_from_event(event: dict) -> APIError:
    err = event.get("stream_error") or {}
    code = str(err.get("code") or "STREAM_ERROR")
    message = str(err.get("error") or "stream failed")
    status = err.get("status")
    status_code: Optional[int] = None
    if isinstance(status, int):
        status_code = status
    elif isinstance(status, str) and status.isdigit():
        status_code = int(status)

    if code == "BOT_CHECK_FAILED":
        return BotCheckError(
            f"{message}: the Turnstile token or human-check header was rejected "
            f"(tokens are single use, so one request = one token)",
            status_code=status_code,
            code=code,
            body=json.dumps(event),
        )
    if code == "ACCESS_DENIED":
        return AccessDeniedError(message, status_code=status_code, code=code, body=json.dumps(event))
    if status_code == 429:
        return RateLimitError(message, status_code=status_code, code=code, body=json.dumps(event))
    return APIError(message, status_code=status_code, code=code, body=json.dumps(event))


def parse_sse_line(line: str) -> Optional[ChatCompletionChunk]:
    """Decode a single line of the stream.

    Returns ``None`` for keep-alives (``: connected``), empty lines and unknown
    fields, a ``finish_reason="stop"`` chunk for ``data: [DONE]``, and raises a
    typed error for a ``stream_error`` event.
    """
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None

    data = line[len("data:") :].strip()
    if not data:
        return None
    if data == _DONE:
        return ChatCompletionChunk(finish_reason="stop")

    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        # A truncated tail is normal when the connection dies mid-line.
        return None

    if not isinstance(event, dict):
        return None
    if "stream_error" in event:
        raise _error_from_event(event)

    chunk = ChatCompletionChunk(
        model=event.get("model"),
        usage=event.get("usage"),
        raw=event,
    )

    choices = event.get("choices") or []
    if not choices:
        return chunk

    choice = choices[0] or {}
    delta = choice.get("delta") or choice.get("message") or {}
    if isinstance(delta.get("content"), str):
        chunk.content = delta["content"]
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
    if isinstance(reasoning, str) and reasoning:
        chunk.reasoning_content = reasoning
    chunk.finish_reason = choice.get("finish_reason")
    return chunk


def iter_sse(body: str) -> Iterator[ChatCompletionChunk]:
    """Yield every delta in a complete SSE body, in order."""
    for line in body.splitlines():
        chunk = parse_sse_line(line)
        if chunk is not None:
            yield chunk
            if chunk.is_done:
                return


class SSEDecoder:
    """Incremental decoder for a live socket (used by the proxy)."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[ChatCompletionChunk]:
        self._buffer += text
        out: list[ChatCompletionChunk] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            chunk = parse_sse_line(line)
            if chunk is not None:
                out.append(chunk)
                if chunk.is_done:
                    self._buffer = ""
        return out

    def flush(self) -> list[ChatCompletionChunk]:
        rest, self._buffer = self._buffer, ""
        chunk = parse_sse_line(rest)
        return [chunk] if chunk is not None else []


def text_of(body: str) -> str:
    """Concatenate the visible text of a complete SSE body."""
    return "".join(c.content for c in iter_sse(body))


def ensure_stream(body: str) -> str:
    """Raise if ``body`` is not an SSE stream at all (e.g. an HTML error page)."""
    if "data:" in body or body.lstrip().startswith(":"):
        return body
    raise StreamError(
        f"expected an SSE body, got {len(body)} bytes: {body[:200]!r}"
    )


__all__ = [
    "ChatCompletionChunk",
    "SSEDecoder",
    "ensure_stream",
    "iter_sse",
    "parse_sse_line",
    "text_of",
]
