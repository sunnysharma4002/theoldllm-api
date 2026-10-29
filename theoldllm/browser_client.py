"""Backwards-compatible import path.

Older versions exported ``PlaywrightTheOldLLM`` from here; it is the same class
as :class:`theoldllm.browser.AsyncTheOldLLM`.
"""

from .browser import AsyncTheOldLLM

PlaywrightTheOldLLM = AsyncTheOldLLM

__all__ = ["AsyncTheOldLLM", "PlaywrightTheOldLLM"]
