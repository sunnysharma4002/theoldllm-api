"""Live smoke test — needs Playwright, Chromium, and a display.

    THEOLDLLM_LIVE=1 pytest tests/test_live.py -s

It opens a real (headed) browser, so it is skipped by default.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("THEOLDLLM_LIVE"),
    reason="set THEOLDLLM_LIVE=1 to run against the real site",
)

MODEL = os.environ.get("THEOLDLLM_MODEL", "gpt-6-luna")


def test_anonymous_model_answers():
    from theoldllm import TheOldLLM

    with TheOldLLM(headless=False) as bot:
        chunks = list(
            bot.chat_stream(MODEL, [{"role": "user", "content": "Reply with exactly: ok"}], max_tokens=32)
        )
    text = "".join(c.content for c in chunks)
    assert text.strip(), "empty reply"
    assert chunks[-1].is_done


def test_verified_model_is_reported_as_needing_an_account():
    from theoldllm import AccessDeniedError, TheOldLLM

    with TheOldLLM(headless=False) as bot:
        with pytest.raises(AccessDeniedError):
            bot.chat("gpt-6-sol", [{"role": "user", "content": "hi"}], max_tokens=16)
