# TheOldLLM Python client

Unofficial client for [TheOldLLM](https://theoldllm.vercel.app/) — a free
multi-model chat front end (GPT-6, GPT-5.6, Claude Opus, Gemini, Grok).

**It drives a real browser.** The chat route cannot be called over plain HTTP;
see [Why a browser is required](#why-a-browser-is-required). If you only want to
talk to it from a console, use [`console/theoldllm.js`](console/theoldllm.js)
instead — no Python, no Playwright.

```python
from theoldllm import TheOldLLM

with TheOldLLM() as bot:                       # opens a headed Chromium window
    print(bot.chat("gpt-6-luna", [{"role": "user", "content": "hello"}]))
```

## Install

```bash
pip install -e ".[proxy,dev]"      # proxy + test extras are optional
python -m playwright install chromium
```

Python 3.9+, Playwright 1.40+.

## Quick start

Async (what the internals use):

```python
import asyncio
from theoldllm import AsyncTheOldLLM

async def main():
    async with AsyncTheOldLLM() as bot:
        reply = await bot.chat("gpt-6-luna", [{"role": "user", "content": "hello"}])
        print(reply)

asyncio.run(main())
```

Streaming — deltas in order, but buffered by the page (see the note below):

```python
with TheOldLLM() as bot:
    for chunk in bot.chat_stream(
        "gpt-5.5-thinking",
        [{"role": "user", "content": "write a haiku"}],
        reasoning_effort="high",
    ):
        print(chunk.content, end="", flush=True)
```

Turnstile requires a **headed** browser. `headless=True` is accepted but the
widget's callback never fires there, so token minting times out — the client says
so instead of silently failing.

## Why a browser is required

Three independent gates sit in front of `POST /api/aichat`:

| Layer | Behaviour | What beats it |
|---|---|---|
| Vercel Security Checkpoint | `429` + "Vercel Security Checkpoint" HTML for any non-browser socket | a real browser session |
| Human check | bundle patches `window.fetch`, adds signed `x-is-human`, `x-path`, `x-method` | the request issued *by the page*, so the wrapper runs |
| Cloudflare Turnstile | `x-turnstile-token` from site key `0x4AAAAAADvYUN7pfr-aCtEA`, **single use** | mint a fresh token in the page before every call |

Replaying captured headers/cookies/TLS fingerprints does not work: a captured
`x-is-human` returns `BOT_CHECK_FAILED`, and a reused Turnstile token does too.
That is why `curl_cffi` "WAF bypass" code (present in v0.1.0 and removed here)
could never work, and why this package drives Playwright instead of `requests`.

## Models

`python -m theoldllm models` prints exactly this table. Ids and upstream
providers come from the site's own bundle (`assets/index-*.js`).

| id | name | provider | needs account |
|---|---|---|---|
| `gpt-6-luna` | GPT-6 Luna | openai | no |
| `gpt-5.5-thinking` | GPT-5.5 Thinking | openai | no |
| `gemini-3.8-flash` | Gemini 3.8 Flash | google | no |
| `grok-4.5-latest` | Grok 4.5 | xai | no |
| `grok-4.3-latest` | Grok 4.3 | xai | no |
| `gpt-6-sol` | GPT-6 Sol | openai | **yes** |
| `gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna` | GPT-5.6 family | openai | **yes** |
| `claude-opus-4-7` | Claude Opus 5.5 | anthropic | **yes** |
| `gemini-3.1-pro` | Gemini 3.1 Pro | google | **yes** |
| `grok-4.7-latest` | Grok 4.7 | xai | **yes** |
| `gpt-6-astra` | GPT-6 Astra | openai | disabled upstream |
| `gpt-5-mini-aichat` | GPT-5 Mini | openai | hidden fallback |

"Needs account" is the bundle's `requiresVerifiedAccount` flag: an anonymous
request gets `403 ACCESS_DENIED` (not a bot-check failure, and not fixable by
retrying). `gpt-6-luna` was verified live without an account; the other "no"
rows are unflagged in the bundle but were not each exercised. Ids from v0.1.0
that no longer exist (`claude-sonnet-5`, `claude-opus-4-8`, `gemini-3.5-flash`,
`gpt-5.4`, …) are mapped to the closest surviving model with a
`DeprecationWarning`; see `theoldllm/models.py`.

## Errors

Failures arrive as **HTTP 200 with a `stream_error` event**, which this client
turns into typed exceptions:

```python
from theoldllm import AccessDeniedError, BotCheckError, TurnstileError

try:
    bot.chat("gpt-6-sol", [{"role": "user", "content": "hi"}])
except AccessDeniedError as error:
    print(error.model)   # 'gpt-6-sol' — needs a signed-in account
```

| Exception | Meaning |
|---|---|
| `AccessDeniedError` | `ACCESS_DENIED`: model is `requiresVerifiedAccount` |
| `BotCheckError` | `BOT_CHECK_FAILED`: token/human-check rejected (retried once automatically) |
| `TurnstileError` | no token could be minted — usually `headless=True` |
| `BrowserError` / `BrowserNotReadyError` | Playwright failed, or the page is stuck on Vercel's checkpoint |
| `UnknownModelError` | id is not in the catalogue; the message lists the real ids |
| `APIError` / `RateLimitError` / `AuthenticationError` | anything else, with `status_code` and `code` |

## CLI

```bash
python -m theoldllm models
python -m theoldllm models --json
python -m theoldllm chat "hello" --model gpt-6-luna --stream
python -m theoldllm chat "hello" --system "be terse" --max-tokens 200
python -m theoldllm serve --port 11434            # OpenAI-compatible facade
```

## OpenAI-compatible facade

`theoldllm serve` exposes `POST /v1/chat/completions`, `GET /v1/models` and
`GET /health` on `127.0.0.1:11434`, which is enough for opencode, the `openai`
package, or LangChain:

```json
{
  "provider": {
    "theoldllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TheOldLLM",
      "options": { "baseURL": "http://127.0.0.1:11434/v1" },
      "models": { "gpt-6-luna": { "name": "GPT-6 Luna" } }
    }
  },
  "model": "theoldllm/gpt-6-luna"
}
```

Requests are serialised because the browser is one page with one Turnstile
widget; expect queuing, not parallelism. `usage` is always zero — the upstream
does not report token counts.

## Docker / Railway

The root `Dockerfile` installs Chromium, Playwright and Xvfb, and runs the facade
inside a virtual display (`xvfb-run`), because headless Chromium does not solve
Turnstile:

```bash
docker build -t theoldllm .
docker run -p 8080:8080 theoldllm
```

On Railway, point the service at this repo; `railway.json` selects the
`Dockerfile`. Two settings matter:

* **Memory ≥ 1 GB.** Chromium plus Xvfb does not fit in 512 MB; the container
  will OOM-loop.
* **`THEOLDLLM_STORAGE`** (default `/app/data/session.json`) keeps cookies, so
  `cf_clearance` survives a restart if you attach a volume to `/app/data`.

The server opens the HTTP port *before* launching the browser, and a browser
failure no longer kills the process (Railway would just restart it and hide the
reason). Ask it directly:

```bash
curl https://<app>.up.railway.app/health          # container up?
curl https://<app>.up.railway.app/health?deep=1   # can it actually complete a chat?
```

`deep` runs one real `gpt-6-luna` completion and returns `{"ok":true,"reply":"ok"}`
or the exact upstream reason, e.g. `{"ok":false,"code":"turnstile_unavailable"}`.
It also returns a `page` fingerprint (`userAgent`, `webdriver`, `brands`,
`challengeFrames`) so a failure can be attributed:

| `page.webdriver` | `page.brands` | Meaning | Lever |
|---|---|---|---|
| `true` | empty | the browser is being detected as automation | `THEOLDLLM_CHANNEL=chrome` (default in the image) |
| `false` | populated | clean browser, still challenged | the **IP**: set `THEOLDLLM_PROXY` to a residential/mobile proxy |
| `false` | populated | clean browser, no challenge frame | the page loaded wrong; check `page.title` |

`THEOLDLLM_PROXY` accepts a URL (`http://host:port`) or JSON for authenticated
proxies (`{"server": "http://host:port", "username": "...", "password": "..."}`).

The image already installs Google Chrome and sets `THEOLDLLM_CHANNEL=chrome`,
because Playwright's bundled Chromium is challenged far more often. If a channel
fails to launch, the client logs a warning and falls back to bundled Chromium.

This is best-effort: Cloudflare may serve an interactive challenge to a
datacenter IP, and nothing can solve that unattended. Run it on a machine where
a real window is acceptable if you need reliability.

## Console client

`console/theoldllm.js` is the same protocol with no dependencies: paste it into
DevTools on an open tab of the site.

```js
await oldllm.chat("hello")                                  // gpt-6-luna
await oldllm.chat("hello", { model: "gpt-5.5-thinking" })
await oldllm.raw(oldllm.buildPayload([{ role: "user", content: "hi" }]))
```

## Verification

* Offline: `pytest` — 57 tests over SSE decoding, payload building, model
  resolution, retry/error mapping, and the OpenAI facade. No network, no browser.
* Live, in a page on the site: the exact JS shipped in
  `theoldllm/browser.py` (`_MINT_TOKEN_JS` + `_REQUEST_JS`) minted a token and
  streamed `shipped js works` from `gpt-6-luna`.
* Not executed in the authoring environment: the Playwright glue around that JS
  (`launch → settle → evaluate`), because Playwright was intentionally absent.
  The JS contract it calls is what was proven live above.
* Live, protocol: `gpt-6-sol` with a valid token → `ACCESS_DENIED`; no token →
  `BOT_CHECK_FAILED`; reused token → `BOT_CHECK_FAILED`; `curl`/`requests` →
  `429` Vercel checkpoint.
* `tests/test_live.py` re-checks the anonymous/verified split against the real
  site; it is skipped unless `THEOLDLLM_LIVE=1`.

## Notes and limits

* Playwright buffers `page.evaluate`, so `chat_stream()` yields the deltas *after*
  the reply completes. Content and ordering are exact; wall-clock streaming is not.
* One Turnstile token per request; do not fire requests concurrently at the same
  page, and keep volume low.
* Private endpoint of a free service, with no SLA. Anything here can break the
  day the site changes its bundle.
* `/api/rtkey` (voice/realtime) is not wrapped: it returns
  `500 {"error":"Failed to get key"}` without a verified account.

## Changes in 0.2.0

* Works at all: sends `x-turnstile-token` and mints a token per request.
* Correct catalogue, `VERIFIED_ONLY`, and deprecation aliases for dead ids.
* `stream_error` events become typed exceptions instead of silent empty replies.
* Retries a request when the page navigates mid-call (the old failure mode was
  `Execution context was destroyed`).
* Removed the `curl_cffi` client and the "bypasses Vercel WAF" claim.
* `TheOldLLM` / `AsyncTheOldLLM` now both drive the browser; `PlaywrightTheOldLLM`
  remains as an alias. Removed: `X-Supabase-Auth`, mouse-jitter "stealth",
  `--single-process`, committed `__pycache__`, `setup.py`/`requirements.txt`
  (use `pyproject.toml`).

MIT.
