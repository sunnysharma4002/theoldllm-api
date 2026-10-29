"""TheOldLLM — unofficial Python client for <https://theoldllm.vercel.app>.

The chat route can only be reached from inside a page on the site, so this
package drives a real Chromium through Playwright, mints a Cloudflare Turnstile
token per request, and parses the SSE reply. Importing the package does not
launch anything — use it as a context manager.

    >>> from theoldllm import TheOldLLM
    >>> with TheOldLLM() as bot:
    ...     bot.chat("gpt-6-luna", [{"role": "user", "content": "hi"}])
"""

from .browser import (
    BASE_URL,
    CHAT_PATH,
    TURNSTILE_SITEKEY,
    AsyncTheOldLLM,
)
from .client import API_ENDPOINTS, TheOldLLM
from .exceptions import (
    AccessDeniedError,
    APIError,
    AuthenticationError,
    BotCheckError,
    BrowserError,
    BrowserNotReadyError,
    RateLimitError,
    StreamError,
    TheOldLLMError,
    TurnstileError,
    UnknownModelError,
)
from .models import DEFAULT_MAX_TOKENS, MODELS, VERIFIED_ONLY, Model, Models, Provider
from .payload import DEFAULT_SYSTEM, build_payload
from .streaming import ChatCompletionChunk, SSEDecoder, iter_sse, parse_sse_line, text_of

#: Historical name; it is an alias, not a separate implementation.
PlaywrightTheOldLLM = AsyncTheOldLLM

__version__ = "0.2.0"

__all__ = [
    "API_ENDPOINTS",
    "AccessDeniedError",
    "APIError",
    "AsyncTheOldLLM",
    "AuthenticationError",
    "BASE_URL",
    "BotCheckError",
    "BrowserError",
    "BrowserNotReadyError",
    "CHAT_PATH",
    "ChatCompletionChunk",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SYSTEM",
    "MODELS",
    "Model",
    "Models",
    "PlaywrightTheOldLLM",
    "Provider",
    "RateLimitError",
    "SSEDecoder",
    "StreamError",
    "TURNSTILE_SITEKEY",
    "TheOldLLM",
    "TheOldLLMError",
    "TurnstileError",
    "UnknownModelError",
    "VERIFIED_ONLY",
    "build_payload",
    "iter_sse",
    "parse_sse_line",
    "text_of",
]
