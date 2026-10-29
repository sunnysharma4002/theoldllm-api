"""Blocking wrapper around :class:`theoldllm.browser.AsyncTheOldLLM`.

Playwright's objects are bound to the event loop that created them, so the
browser lives in a dedicated background thread with its own loop; calls are
submitted to it and results are handed back through a queue. That keeps the
familiar synchronous API without re-implementing the transport.
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
from typing import Any, Iterator, Mapping, Optional, Sequence

from .browser import (
    BASE_URL,
    CHAT_PATH,
    LEGACY_CHAT_PATH,
    AsyncTheOldLLM,
)
from .models import Model, Models
from .payload import build_payload
from .streaming import ChatCompletionChunk

#: Endpoint per historical "provider" name; ``/api/aichat`` superseded both.
API_ENDPOINTS: dict[str, str] = {
    "aichat": CHAT_PATH,
    "chatgpt": LEGACY_CHAT_PATH,
}

_SENTINEL = object()


class TheOldLLM:
    """Synchronous client. Use as a context manager.

    ```python
    with TheOldLLM() as bot:                    # opens a headed Chromium
        print(bot.chat("gpt-6-luna", [{"role": "user", "content": "hi"}]))
    ```

    Accepts the same options as :class:`AsyncTheOldLLM` (``headless``,
    ``storage_path``, ``proxy``, timeouts, ``transport``).
    """

    def __init__(self, **options: Any) -> None:
        self._options = options
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[AsyncTheOldLLM] = None
        self._start_lock = threading.Lock()

    # ------------------------------------------------------------------ setup

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._start_lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop,),
                name="theoldllm-loop",
                daemon=True,
            )
            thread.start()
            self._loop = loop
            self._thread = thread
            return loop

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        if sys.platform == "win32":  # Playwright needs a proactor loop to spawn the driver
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _submit(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def start(self) -> "TheOldLLM":
        """Launch the browser now instead of on first use."""
        if self._client is not None:
            return self

        async def boot() -> AsyncTheOldLLM:
            client = AsyncTheOldLLM(**self._options)
            return await client.start()

        self._client = self._submit(boot())
        return self

    def close(self) -> None:
        """Close the browser and stop the worker loop."""
        if self._client is not None:
            try:
                self._submit(self._client.aclose())
            finally:
                self._client = None
        with self._start_lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=10)

    def __enter__(self) -> "TheOldLLM":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ calls

    def chat_stream(
        self,
        model: str | Model,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Yield the reply delta by delta (see the note in ``AsyncTheOldLLM``)."""
        if self._client is None:
            self.start()

        pending: "queue.Queue[Any]" = queue.Queue()

        async def pump() -> None:
            try:
                async for chunk in self._client.chat_stream(model, messages, **options):
                    pending.put(chunk)
            except BaseException as exc:  # handed to the caller's thread
                pending.put(exc)
            finally:
                pending.put(_SENTINEL)

        future = asyncio.run_coroutine_threadsafe(pump(), self._loop)
        while True:
            item = pending.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
        future.result()

    def chat(
        self,
        model: str | Model,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> str:
        """Return the whole reply as one string."""
        return "".join(chunk.content for chunk in self.chat_stream(model, messages, **options))

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Send a prepared payload; returns ``{"status", "text"}``."""
        if self._client is None:
            self.start()
        return self._submit(self._client.request(payload))

    @staticmethod
    def build_payload(model: str | Model, messages: Sequence[Mapping[str, Any]], **options: Any) -> dict[str, Any]:
        return build_payload(model, messages, **options)

    @staticmethod
    def models(*, picker_only: bool = False) -> list[Model]:
        return Models.picker() if picker_only else list(Models.ALL)

    user_message = staticmethod(AsyncTheOldLLM.user_message)
    assistant_message = staticmethod(AsyncTheOldLLM.assistant_message)
    system_message = staticmethod(AsyncTheOldLLM.system_message)


__all__ = ["API_ENDPOINTS", "AsyncTheOldLLM", "BASE_URL", "CHAT_PATH", "TheOldLLM"]
