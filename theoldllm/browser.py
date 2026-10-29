"""Playwright-backed client — the only way to reach the chat route.

Why a browser is mandatory
--------------------------
``POST /api/aichat`` is protected by three layers, all of which have to be
satisfied from inside a real page:

1. **Vercel Security Checkpoint** answers ``429`` to non-browser sockets before
   the route runs. Replaying the browser's own headers, cookies and TLS
   fingerprint does not help (that is what the deleted ``curl_cffi`` "bypass"
   claimed, and it never worked).
2. **Human check.** The bundle patches ``window.fetch`` for the chat route and
   adds ``x-is-human`` (a signed, request-bound blob), ``x-path`` and
   ``x-method``. Replaying a captured ``x-is-human`` yields ``BOT_CHECK_FAILED``.
3. **Cloudflare Turnstile.** The page must mint a token for site key
   ``0x4AAAAAADvYUN7pfr-aCtEA`` and send it as ``x-turnstile-token``. Tokens are
   single use.

Consequence: the request has to be issued *by the page*, and every call needs a
fresh token — which is exactly what :meth:`AsyncTheOldLLM.request` does.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from typing import Any, AsyncIterator, Mapping, Optional, Protocol, Sequence

from .exceptions import (
    AccessDeniedError,
    APIError,
    BrowserError,
    BrowserNotReadyError,
    TurnstileError,
)
from .models import Model, Models, VERIFIED_ONLY
from .payload import build_payload
from .streaming import ChatCompletionChunk, iter_sse

BASE_URL = "https://theoldllm.vercel.app"
CHAT_PATH = "/api/aichat"
LEGACY_CHAT_PATH = "/api/chatgpt"
TURNSTILE_SITEKEY = "0x4AAAAAADvYUN7pfr-aCtEA"

LOG = logging.getLogger("theoldllm")

_MINT_TOKEN_JS = r"""
async ({ sitekey, timeoutMs }) => {
  if (typeof window.turnstile === "undefined" || !window.turnstile.render) {
    throw new Error("window.turnstile is not available on this page");
  }
  const holder = document.createElement("div");
  holder.setAttribute("aria-hidden", "true");
  holder.style.display = "none";
  document.body.appendChild(holder);

  return await new Promise((resolve, reject) => {
    let widget = null;
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("turnstile timeout after " + timeoutMs + "ms"));
    }, timeoutMs);

    function cleanup() {
      window.clearTimeout(timer);
      try { window.turnstile.remove(widget); } catch (error) { /* already gone */ }
      holder.remove();
    }

    try {
      // A solution callback only fires in a real (headed) browser context.
      widget = window.turnstile.render(holder, {
        sitekey: sitekey,
        callback: (token) => { cleanup(); resolve(token); },
        "error-callback": (code) => {
          cleanup();
          reject(new Error("turnstile error-callback: " + String(code)));
        },
        "timeout-callback": () => {
          cleanup();
          reject(new Error("turnstile timeout-callback"));
        },
      });
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}
"""

_REQUEST_JS = r"""
async ({ path, payload, token }) => {
  // Bare `fetch` resolves to window.fetch, i.e. the bundle's patched wrapper,
  // which is what appends the signed x-is-human header.
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-turnstile-token": token,
    },
    body: JSON.stringify(payload),
    credentials: "include",
  });
  return { status: response.status, text: await response.text() };
}
"""

_PAGE_STATE_JS = r"""
() => ({
  readyState: document.readyState,
  turnstile: typeof window.turnstile,
  title: document.title,
})
"""

#: Errors that mean "the document changed under us", so the call can be redone.
_RETRYABLE = (
    "Execution context was destroyed",
    "Target closed",
    "Target page, context or browser has been closed",
    "Navigating frame was detached",
    "frame was detached",
    "Page closed",
)


class Transport(Protocol):
    """Minimal interface :class:`AsyncTheOldLLM` needs from a page.

    Implemented by :class:`PageTransport` for real runs and by fakes in the
    tests, which is why the parsing/error logic is testable without Playwright.
    """

    async def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        ...


class PageTransport:
    """Issues one authenticated POST from inside a page, given a live page object."""

    def __init__(self, page: Any, *, path: str = CHAT_PATH) -> None:
        self._page = page
        self._path = path

    async def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST ``payload`` with a freshly minted Turnstile token."""
        token = await self._page.evaluate(
            _MINT_TOKEN_JS,
            {"sitekey": TURNSTILE_SITEKEY, "timeoutMs": 30_000},
        )
        result = await self._page.evaluate(
            _REQUEST_JS,
            {"path": self._path, "payload": dict(payload), "token": token},
        )
        return {"status": int(result.get("status") or 0), "text": str(result.get("text") or "")}


class AsyncTheOldLLM:
    """Async client. Use as an async context manager.

    Args:
        base_url: site root.
        headless: keep ``False``. In headless Chromium the Turnstile callback
            never fires, so token minting times out; a headed window solves it in
            about a second. Set it only if you have verified headless works for
            you.
        storage_path: JSON file for cookies/localStorage between runs.
        proxy: Playwright ``proxy=`` dict, e.g. ``{"server": "http://...:8080"}``.
        channel: Playwright browser channel, typically ``"chrome"``. Real Chrome
            is far less likely to be challenged than the bundled Chromium build;
            it falls back to the bundled one if the channel is unavailable.
        user_agent: overrides the browser's own UA. Leave it ``None`` — a forced
            UA that disagrees with the actual browser is a fingerprinting signal.
        transport: testing hook — anything with an async ``request(payload)``
            returning ``{"status": int, "text": str}``. When set, no browser is
            launched.
        request_timeout / token_timeout / load_timeout: seconds.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        headless: bool = False,
        storage_path: Optional[str] = None,
        proxy: Optional[dict] = None,
        channel: Optional[str] = None,
        user_agent: Optional[str] = None,
        path: str = CHAT_PATH,
        settle_delay: float = 1.0,
        request_timeout: float = 240.0,
        token_timeout: float = 30.0,
        load_timeout: float = 60.0,
        transport: Optional[Transport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headless = headless
        self.storage_path = storage_path
        self.proxy = proxy
        self.channel = channel
        self.user_agent = user_agent
        self.path = path
        self.settle_delay = settle_delay
        self.request_timeout = request_timeout
        self.token_timeout = token_timeout
        self.load_timeout = load_timeout

        self._transport = transport
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._started = False
        self._closed = False
        # Locks are created on first use so the object can be built outside a
        # running loop (Python 3.9 binds a Lock to the loop it is made in).
        self._request_lock: Optional[asyncio.Lock] = None
        self._start_guard: Optional[asyncio.Lock] = None

    def _lock(self, name: str) -> asyncio.Lock:
        lock = getattr(self, name)
        if lock is None:
            lock = asyncio.Lock()
            setattr(self, name, lock)
        return lock

    # ------------------------------------------------------------------ setup

    @property
    def page(self) -> Any:
        return self._page

    async def start(self) -> "AsyncTheOldLLM":
        async with self._lock("_start_guard"):
            if self._started:
                return self
            if self._transport is None:
                if self.headless:
                    warnings.warn(
                        "headless=True: Cloudflare Turnstile generally never fires in "
                        "headless Chromium, so token minting will time out",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                await self._launch()
            self._started = True
            return self

    async def _launch(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise BrowserError(
                "Playwright is not installed: pip install playwright && "
                "python -m playwright install chromium"
            ) from exc

        self._playwright = await async_playwright().start()
        launch: dict[str, Any] = {
            "headless": self.headless,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                # Keeps navigator.webdriver falsy. Not sufficient on its own.
                "--disable-blink-features=AutomationControlled",
            ],
            # Playwright adds --enable-automation by default; it is one of the
            # cheapest bot signals to read, so drop it.
            "ignore_default_args": ["--enable-automation"],
        }
        if self.proxy:
            launch["proxy"] = self.proxy

        try:
            self._browser = await self._launch_browser(launch)
            context_args: dict[str, Any] = {
                "locale": "en-US",
                "viewport": {"width": 1280, "height": 800},
            }
            if self.user_agent:
                context_args["user_agent"] = self.user_agent
            if self.storage_path and os.path.exists(self.storage_path):
                context_args["storage_state"] = self.storage_path
            self._context = await self._browser.new_context(**context_args)
            self._page = await self._context.new_page()
            await self._page.goto(
                self.base_url,
                wait_until="domcontentloaded",
                timeout=self.load_timeout * 1000,
            )
            await self._settle()
            await self._save_state()
        except Exception as exc:
            await self.aclose()
            raise BrowserError(f"could not prepare the browser: {exc}") from exc

    async def _launch_browser(self, launch: dict[str, Any]) -> Any:
        """Launch the configured channel, falling back to bundled Chromium."""
        if self.channel:
            try:
                browser = await self._playwright.chromium.launch(channel=self.channel, **launch)
                LOG.info("launched channel=%s", self.channel)
                return browser
            except Exception as exc:
                LOG.warning(
                    "channel=%s unavailable (%s); falling back to bundled chromium",
                    self.channel,
                    exc,
                )
        return await self._playwright.chromium.launch(**launch)

    async def aclose(self) -> None:
        self._closed = True
        self._started = False
        for closer in ("_context", "_browser"):
            obj = getattr(self, closer)
            if obj is not None:
                try:
                    await obj.close()
                except Exception:  # pragma: no cover - best effort
                    pass
                setattr(self, closer, None)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # pragma: no cover - best effort
                pass
            self._playwright = None
        self._page = None

    close = aclose

    async def __aenter__(self) -> "AsyncTheOldLLM":
        return await self.start()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def _save_state(self) -> None:
        if not self.storage_path or self._context is None:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            await self._context.storage_state(path=self.storage_path)
        except Exception as exc:  # pragma: no cover - best effort
            LOG.debug("could not save storage state: %s", exc)

    async def _settle(self) -> None:
        """Wait until the app (not Vercel's checkpoint) is interactive.

        A navigation between this call and the next ``evaluate`` destroys the JS
        context, so this runs before every request rather than once at startup.
        """
        page = self._page
        if page is None:
            raise BrowserNotReadyError("start() has not completed")
        try:
            await page.wait_for_load_state("load", timeout=self.load_timeout * 1000)
        except Exception:
            pass
        try:
            await page.wait_for_function(
                "() => document.readyState === 'complete'",
                timeout=min(self.load_timeout, 15) * 1000,
            )
        except Exception:
            pass
        try:
            await page.wait_for_function(
                "() => typeof window.turnstile !== 'undefined'",
                timeout=self.load_timeout * 1000,
            )
        except Exception as exc:
            state = {}
            try:
                state = await page.evaluate(_PAGE_STATE_JS)
            except Exception:
                pass
            raise BrowserNotReadyError(
                "window.turnstile never appeared — the page is probably still on "
                f"Vercel's security checkpoint (state: {state})"
            ) from exc
        # Give the patched fetch / human-check runtime a moment to install.
        await asyncio.sleep(self.settle_delay)

    async def _mint_token(self) -> str:
        try:
            token = await self._page.evaluate(
                _MINT_TOKEN_JS,
                {"sitekey": TURNSTILE_SITEKEY, "timeoutMs": int(self.token_timeout * 1000)},
            )
        except Exception as exc:
            text = str(exc)
            raise TurnstileError(
                f"could not mint a Turnstile token: {text}. "
                "Run with headless=False; a headed window solves it immediately.",
                timed_out="timeout" in text.lower(),
            ) from exc
        if not token or not isinstance(token, str):
            raise TurnstileError("Turnstile returned an empty token")
        return token

    # --------------------------------------------------------------- requests

    async def request(
        self,
        payload: Mapping[str, Any],
        *,
        attempts: int = 3,
    ) -> dict[str, Any]:
        """POST ``payload`` to the chat route and return ``{status, text}``.

        Mints a new Turnstile token per attempt (they are single use), retries
        when the page navigates mid-call, and retries once when the bot check
        rejects the request (the first call on a cold page can race the
        human-check runtime).
        """
        if self._transport is not None:
            result = await self._transport.request(payload)
            return {"status": int(result.get("status", 200)), "text": result.get("text", "")}

        await self.start()
        last_error: Optional[Exception] = None
        bot_failures = 0
        async with self._lock("_request_lock"):
            for attempt in range(1, attempts + 1):
                try:
                    await self._settle()
                    token = await self._mint_token()
                    result = await self._page.evaluate(
                        _REQUEST_JS,
                        {"path": self.path, "payload": dict(payload), "token": token},
                    )
                except TurnstileError:
                    raise
                except Exception as exc:
                    if self._is_retryable(exc) and attempt < attempts:
                        last_error = exc
                        LOG.debug("page changed during request (attempt %d): %s", attempt, exc)
                        continue
                    raise BrowserError(
                        f"the in-page request failed on attempt {attempt}: {exc}"
                    ) from exc

                status = int(result.get("status") or 0)
                text = str(result.get("text") or "")
                if attempt < attempts and status == 200 and '"BOT_CHECK_FAILED"' in text:
                    bot_failures += 1
                    LOG.debug("bot check rejected attempt %d", attempt)
                    if bot_failures >= 2:
                        # The page's human-check runtime has gone stale (long
                        # running server); a reload rebuilds it.
                        await self._reload()
                    else:
                        await asyncio.sleep(1.5)
                    continue
                return {"status": status, "text": text, "attempts": attempt}

            raise BrowserError(
                f"the page call kept losing its execution context: {last_error}"
            )

    async def _reload(self) -> None:
        """Reload the page after repeated bot-check rejections."""
        if self._page is None:
            return
        LOG.info("reloading the page to rebuild the human-check runtime")
        try:
            await self._page.reload(wait_until="domcontentloaded", timeout=self.load_timeout * 1000)
        except Exception as exc:  # pragma: no cover - depends on the page
            LOG.debug("reload failed: %s", exc)
            return
        await self._settle()
        await self._save_state()

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        text = str(exc)
        return any(marker.lower() in text.lower() for marker in _RETRYABLE)

    # -------------------------------------------------------------------- chat

    def build_payload(self, model: str | Model, messages: Sequence[Mapping[str, Any]], **options: Any) -> dict[str, Any]:
        """Build a request body; see :func:`theoldllm.payload.build_payload`."""
        return build_payload(model, messages, **options)

    async def chat_stream(
        self,
        model: str | Model,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Yield the reply delta by delta.

        Note: Playwright returns the whole response body from one ``evaluate``,
        so the deltas are buffered by the page and handed back in order after the
        reply finishes. Ordering and content are exact; wall-clock streaming is
        not.
        """
        body = self.build_payload(model, messages, **options)
        result = await self.request(body)
        model_id = str(body["model"])

        if result["status"] not in (200, 201):
            raise APIError(
                f"upstream returned HTTP {result['status']}",
                status_code=result["status"],
                body=result["text"][:2000],
            )

        try:
            for chunk in iter_sse(result["text"]):
                yield chunk
        except AccessDeniedError as exc:
            raise _explain_denied(exc, model_id) from None

    async def chat(
        self,
        model: str | Model,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> str:
        """Return the whole reply as one string."""
        parts: list[str] = []
        async for chunk in self.chat_stream(model, messages, **options):
            parts.append(chunk.content)
        return "".join(parts)

    # ------------------------------------------------------------------- misc

    @staticmethod
    def models(*, picker_only: bool = False) -> list[Model]:
        return Models.picker() if picker_only else list(Models.ALL)

    @staticmethod
    def user_message(content: str) -> dict[str, str]:
        return {"role": "user", "content": content}

    @staticmethod
    def assistant_message(content: str) -> dict[str, str]:
        return {"role": "assistant", "content": content}

    @staticmethod
    def system_message(content: str) -> dict[str, str]:
        return {"role": "system", "content": content}


def _explain_denied(error: AccessDeniedError, model_id: str) -> AccessDeniedError:
    """Turn ``ACCESS_DENIED`` into something actionable."""
    if model_id in VERIFIED_ONLY:
        anonymous = ", ".join(m.id for m in Models.anonymous()[:4])
        denied = AccessDeniedError(
            f"{error}: {model_id} is flagged requiresVerifiedAccount, so it only "
            f"answers for a signed-in account. Try one of: {anonymous}",
            status_code=error.status_code,
            code=error.code,
            body=error.body,
        )
        denied.model = model_id
        return denied
    error.model = model_id
    return error


__all__ = [
    "AsyncTheOldLLM",
    "BASE_URL",
    "CHAT_PATH",
    "LEGACY_CHAT_PATH",
    "PageTransport",
    "TURNSTILE_SITEKEY",
    "Transport",
]
