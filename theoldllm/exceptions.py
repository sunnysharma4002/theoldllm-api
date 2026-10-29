from __future__ import annotations

from typing import Any, Optional


class TheOldLLMError(Exception):
    """Base exception for every error this package raises."""


class APIError(TheOldLLMError):
    """The endpoint answered with an error (HTTP status and/or ``stream_error``)."""

    def __init__(
        self,
        message: str = "API error",
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        body: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.body = body
        super().__init__(message)


class AuthenticationError(APIError):
    """A 401/403 from the edge, before the app route ran."""


class RateLimitError(APIError):
    """A 429 (Vercel checkpoint, per-IP throttling) or the site's quota error."""


class StreamError(TheOldLLMError):
    """The SSE body ended unexpectedly or was not parseable."""


class BotCheckError(APIError):
    """``BOT_CHECK_FAILED``: the Turnstile token or the ``x-is-human`` blob was stale.

    Always caused by transport behaviour, never by the model. A request issued
    through a fresh page context with a freshly minted token does not fail this
    way; reusing a token (they are single use) or replaying a captured
    ``x-is-human`` value does.
    """


class AccessDeniedError(APIError):
    """``ACCESS_DENIED``: the model is flagged ``requiresVerifiedAccount``.

    Not a client bug and not fixable by retrying — the account must be signed in
    on the page that issues the request.
    """

    def __init__(self, message: str = "Access denied", **kwargs: Any) -> None:
        self.model: Optional[str] = None
        super().__init__(message, **kwargs)


class UnknownModelError(TheOldLLMError):
    """The model id is not in the catalogue the site ships."""

    def __init__(self, model: str, known: list[str]) -> None:
        self.model = model
        self.known = known
        super().__init__(
            f"unknown model {model!r}; available: {', '.join(known)}"
        )


class BrowserError(TheOldLLMError):
    """The browser could not be driven (launch failure, missing driver, ...)."""


class BrowserNotReadyError(BrowserError):
    """The page loaded but the chat runtime / Turnstile widget never appeared."""


class TurnstileError(BrowserError):
    """No Turnstile token could be minted inside the page.

    In practice this means a headless context: the widget's callback never fires
    there. Run with ``headless=False``.
    """

    def __init__(self, message: str = "could not mint a Turnstile token", *, timed_out: bool = False) -> None:
        self.timed_out = timed_out
        super().__init__(message)
