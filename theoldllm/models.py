"""Model catalogue.

The ids below are the ones the site's own bundle ships (``assets/index-*.js``);
the upstream ``provider`` string is echoed back in the request body and the
``upstream_model`` column is what the server forwards to the LLM vendor.

Nothing here is guessed. ``gpt-6-luna``, ``gpt-5.5-thinking`` and
``gemini-3.8-flash`` were verified to answer anonymously; the ids marked
``requires_account`` returned ``403 ACCESS_DENIED`` for the same anonymous
request, which is the flag the bundle calls ``requiresVerifiedAccount``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from .exceptions import UnknownModelError

#: Matches the site's own default output budget.
DEFAULT_MAX_TOKENS = 16000


class Provider(str, Enum):
    """Value sent as the ``provider`` field of the request."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    provider: Provider
    upstream_model: str = ""
    requires_account: bool = False
    unavailable: bool = False
    hidden: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    supports_reasoning: bool = True
    description: str = ""

    @property
    def upstream_provider(self) -> str:
        """Backwards-compatible alias for :attr:`provider`."""
        return self.provider.value

    @property
    def anonymous(self) -> bool:
        """True when the model answers without a signed-in account."""
        return not self.requires_account and not self.unavailable


MODELS: tuple[Model, ...] = (
    Model("gpt-6-sol", "GPT-6 Sol", Provider.OPENAI, "gpt-6-sol", requires_account=True),
    Model("gpt-6-luna", "GPT-6 Luna", Provider.OPENAI, "gpt-6-luna"),
    Model(
        "gpt-6-astra",
        "GPT-6 Astra",
        Provider.OPENAI,
        "gpt-6-astra",
        requires_account=True,
        unavailable=True,
        description="Force-disabled in the bundle; listing it is informational.",
    ),
    Model("gpt-5.6-sol", "GPT-5.6 Sol", Provider.OPENAI, "gpt-5.6-sol", requires_account=True),
    Model("gpt-5.6-terra", "GPT-5.6 Terra", Provider.OPENAI, "gpt-5.6-terra", requires_account=True),
    Model("gpt-5.6-luna", "GPT-5.6 Luna", Provider.OPENAI, "gpt-5.6-luna", requires_account=True),
    Model("gpt-5.5-thinking", "GPT-5.5 Thinking", Provider.OPENAI, "gpt-5.5"),
    Model(
        "gpt-5-mini-aichat",
        "GPT-5 Mini",
        Provider.OPENAI,
        "gpt-5-mini",
        hidden=True,
        description="Kept as a fallback in the bundle; not shown in the model picker.",
    ),
    Model(
        "claude-opus-4-7",
        "Claude Opus 5.5",
        Provider.ANTHROPIC,
        "claude-opus-4-7",
        requires_account=True,
    ),
    Model(
        "gemini-3.1-pro",
        "Gemini 3.1 Pro",
        Provider.GOOGLE,
        "gemini-3.1-pro-preview",
        requires_account=True,
    ),
    Model("gemini-3.8-flash", "Gemini 3.8 Flash", Provider.GOOGLE, "gemini-3.8-flash"),
    Model("grok-4.7-latest", "Grok 4.7", Provider.XAI, "grok-4.7", requires_account=True),
    Model("grok-4.5-latest", "Grok 4.5", Provider.XAI, "grok-4.5-latest"),
    Model("grok-4.3-latest", "Grok 4.3", Provider.XAI, "grok-4.3-latest"),
)

#: Ids that existed in earlier versions of this client and no longer exist on
#: the site. They are resolved to the closest model that does, with a
#: ``DeprecationWarning``, so old scripts keep running instead of dying.
ALIASES: dict[str, str] = {
    "gpt-5.6-sol-preview": "gpt-5.6-sol",
    "gpt-5.5": "gpt-5.5-thinking",
    "gpt-5.5-sol": "gpt-5.5-thinking",
    "gpt-5.4": "gpt-5.5-thinking",
    "gpt-5.3": "gpt-5.5-thinking",
    "gpt-5.2": "gpt-5.5-thinking",
    "gpt-5.1": "gpt-5.5-thinking",
    "gpt-5-mini": "gpt-5-mini-aichat",
    "claude-sonnet-5": "claude-opus-4-7",
    "claude-opus-4-8": "claude-opus-4-7",
    "claude-opus-4.5": "claude-opus-4-7",
    "claude-fable-5": "claude-opus-4-7",
    "gemini-3.5-flash": "gemini-3.8-flash",
    "grok-4.2-latest": "grok-4.3-latest",
    "grok-4.1-latest": "grok-4.3-latest",
    "grok-4": "grok-4.3-latest",
}


@dataclass
class _Registry:
    """Thin, importable registry; ``Models.ALL`` is a plain list."""

    ALL: list[Model] = field(default_factory=lambda: list(MODELS))

    def by_id(self, model_id: str, *, strict: bool = True) -> Optional[Model]:
        """Return the model for ``model_id``, following deprecation aliases.

        Args:
            model_id: catalogue id or a historical alias.
            strict: raise :class:`UnknownModelError` instead of returning None.
        """
        for m in self.ALL:
            if m.id == model_id:
                return m
        alias = ALIASES.get(model_id)
        if alias is not None:
            warnings.warn(
                f"model {model_id!r} no longer exists on TheOldLLM; "
                f"using {alias!r} instead",
                DeprecationWarning,
                stacklevel=3,
            )
            return self.by_id(alias, strict=strict)
        if strict:
            raise UnknownModelError(model_id, self.list_ids())
        return None

    def by_provider(self, provider: Provider | str) -> list[Model]:
        want = Provider(provider) if isinstance(provider, str) else provider
        return [m for m in self.ALL if m.provider is want]

    def list_ids(self, *, include_hidden: bool = True, include_unavailable: bool = True) -> list[str]:
        return [
            m.id
            for m in self.ALL
            if (include_hidden or not m.hidden)
            and (include_unavailable or not m.unavailable)
        ]

    def anonymous(self) -> list[Model]:
        return [m for m in self.ALL if m.anonymous and not m.hidden]

    def picker(self) -> list[Model]:
        """What the site's own model dropdown shows."""
        return [m for m in self.ALL if not m.hidden and not m.unavailable]

    def __iter__(self) -> Iterable[Model]:  # pragma: no cover - convenience
        return iter(self.ALL)


Models = _Registry()

#: Models that need a signed-in account; anonymous calls get ``ACCESS_DENIED``.
VERIFIED_ONLY: frozenset[str] = frozenset(m.id for m in MODELS if m.requires_account)

__all__ = [
    "ALIASES",
    "DEFAULT_MAX_TOKENS",
    "Model",
    "Models",
    "MODELS",
    "Provider",
    "VERIFIED_ONLY",
]
