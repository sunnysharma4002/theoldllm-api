"""Request bodies for ``POST /api/aichat``.

The shape below is copied from what the site's own bundle sends; fields the
bundle never sends are not emitted. ``stream`` is still sent as ``true`` even
when you want a single answer, because the server ignores ``false`` and always
replies with SSE.
"""

from __future__ import annotations

import uuid
from typing import Any, Mapping, Optional, Sequence

from .models import DEFAULT_MAX_TOKENS, Model, Models

#: The system prompt the site's UI prepends when the user has not set one.
DEFAULT_SYSTEM = "You are a helpful AI assistant."

_ROLES = frozenset({"system", "user", "assistant"})


def normalise_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Validate and copy a chat history.

    Raises:
        ValueError: if a message is not ``{"role": ..., "content": ...}`` with a
            known role and a string body.
    """
    out: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} is not a mapping: {message!r}")
        role = message.get("role")
        content = message.get("content")
        if role not in _ROLES:
            raise ValueError(f"message {index} has unknown role {role!r}; use one of {sorted(_ROLES)}")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content must be a string, got {type(content).__name__}")
        out.append({"role": role, "content": content})
    if not out:
        raise ValueError("messages is empty")
    return out


def build_payload(
    model: str | Model,
    messages: Sequence[Mapping[str, Any]],
    *,
    system: Optional[str] = DEFAULT_SYSTEM,
    temperature: float = 0.7,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = "medium",
    thread_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a request body.

    Args:
        model: catalogue id (or an already-resolved :class:`Model`).
        messages: history without a system prompt.
        system: system prompt prepended when ``messages`` has none. ``None``
            or ``""`` disables it.
        max_tokens: output budget; defaults to the site's own 16000.
        reasoning_effort: ``"low"``/``"medium"``/``"high"``/``"max"``, or
            ``None`` to omit the field.
        thread_id: conversation id; a fresh uuid4 per call by default.
        extra: additional top-level fields, merged last.
    """
    entry = model if isinstance(model, Model) else Models.by_id(model)
    assert entry is not None  # Models.by_id raises unless strict=False

    history = normalise_messages(messages)
    if system and not any(m["role"] == "system" for m in history):
        history.insert(0, {"role": "system", "content": system})

    payload: dict[str, Any] = {
        "model": entry.id,
        "messages": history,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "provider": entry.provider.value,
        "threadId": thread_id or str(uuid.uuid4()),
        "max_tokens": max_tokens or entry.max_tokens or DEFAULT_MAX_TOKENS,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if extra:
        payload.update(dict(extra))
    return payload


__all__ = ["DEFAULT_SYSTEM", "build_payload", "normalise_messages"]
