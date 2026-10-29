/**
 * TheOldLLM console client — https://theoldllm.vercel.app
 *
 * Paste this whole file into the DevTools console of an open tab on the site
 * (or run it through the app's built-in browser), then:
 *
 *   await oldllm.chat("hello")
 *   await oldllm.chat("hello", { model: "gpt-5.5-thinking" })
 *   await oldllm.ask([{ role: "user", content: "hi" }], { model: "gpt-6-luna" })
 *   oldllm.models                       // catalogue
 *
 * Why it must run in the page: every chat call needs the signed `x-is-human`
 * header, which the site's own fetch wrapper adds, plus a single-use Cloudflare
 * Turnstile token. A non-browser HTTP client gets 429 (Vercel checkpoint) and,
 * if it somehow reaches the route, `BOT_CHECK_FAILED`.
 */
(() => {
  const CHAT_PATH = "/api/aichat";
  const TURNSTILE_SITEKEY = "0x4AAAAAADvYUN7pfr-aCtEA";
  const DEFAULT_MODEL = "gpt-6-luna";

  /** id -> { name, provider, upstream } as shipped in the site bundle. */
  const MODELS = {
    "gpt-6-sol": { name: "GPT-6 Sol", provider: "openai", upstream: "gpt-6-sol", verifiedOnly: true },
    "gpt-6-luna": { name: "GPT-6 Luna", provider: "openai", upstream: "gpt-6-luna" },
    "gpt-6-astra": { name: "GPT-6 Astra", provider: "openai", upstream: "gpt-6-astra", verifiedOnly: true, unavailable: true },
    "gpt-5.6-sol": { name: "GPT-5.6 Sol", provider: "openai", upstream: "gpt-5.6-sol", verifiedOnly: true },
    "gpt-5.6-terra": { name: "GPT-5.6 Terra", provider: "openai", upstream: "gpt-5.6-terra", verifiedOnly: true },
    "gpt-5.6-luna": { name: "GPT-5.6 Luna", provider: "openai", upstream: "gpt-5.6-luna", verifiedOnly: true },
    "gpt-5.5-thinking": { name: "GPT-5.5 Thinking", provider: "openai", upstream: "gpt-5.5" },
    "gpt-5-mini-aichat": { name: "GPT-5 Mini", provider: "openai", upstream: "gpt-5-mini", hidden: true },
    "claude-opus-4-7": { name: "Claude Opus 5.5", provider: "anthropic", upstream: "claude-opus-4-7", verifiedOnly: true },
    "gemini-3.1-pro": { name: "Gemini 3.1 Pro", provider: "google", upstream: "gemini-3.1-pro-preview", verifiedOnly: true },
    "gemini-3.8-flash": { name: "Gemini 3.8 Flash", provider: "google", upstream: "gemini-3.8-flash" },
    "grok-4.7-latest": { name: "Grok 4.7", provider: "xai", upstream: "grok-4.7", verifiedOnly: true },
    "grok-4.5-latest": { name: "Grok 4.5", provider: "xai", upstream: "grok-4.5-latest" },
    "grok-4.3-latest": { name: "Grok 4.3", provider: "xai", upstream: "grok-4.3-latest" },
  };

  class OldLLMError extends Error {
    constructor(code, message, status) {
      super(`${code}: ${message}` + (status ? ` (HTTP ${status})` : ""));
      this.code = code;
      this.status = status;
    }
  }

  /** Mint one Turnstile token. Single use — never cache it. */
  function turnstileToken(timeoutMs = 25000) {
    return new Promise((resolve, reject) => {
      if (typeof window.turnstile === "undefined") {
        reject(new OldLLMError("NO_TURNSTILE", "wait for the page to finish loading"));
        return;
      }
      const holder = document.createElement("div");
      holder.style.display = "none";
      document.body.appendChild(holder);
      const timer = setTimeout(() => {
        cleanup();
        reject(new OldLLMError("TURNSTILE_TIMEOUT", "widget never solved (headless / background tab?)"));
      }, timeoutMs);
      let widget;
      const cleanup = () => {
        try { window.turnstile.remove(widget); } catch (_) {}
        holder.remove();
      };
      widget = window.turnstile.render(holder, {
        sitekey: TURNSTILE_SITEKEY,
        callback: (token) => { clearTimeout(timer); cleanup(); resolve(token); },
        "error-callback": (err) => { clearTimeout(timer); cleanup(); reject(new OldLLMError("TURNSTILE_ERROR", String(err))); },
      });
    });
  }

  function buildPayload(messages, options = {}) {
    const model = options.model || DEFAULT_MODEL;
    if (!MODELS[model]) {
      throw new OldLLMError("UNKNOWN_MODEL", `${model}; known: ${Object.keys(MODELS).join(", ")}`);
    }
    const system = options.system === undefined ? "You are a helpful AI assistant." : options.system;
    const conversation = system ? [{ role: "system", content: system }, ...messages] : [...messages];
    const payload = {
      model,
      messages: conversation,
      stream: true,
      temperature: options.temperature ?? 0.7,
      top_p: options.top_p ?? 0.9,
      frequency_penalty: 0,
      presence_penalty: 0,
      provider: options.provider || MODELS[model].provider,
      threadId: options.threadId || crypto.randomUUID(),
      max_tokens: options.maxTokens ?? 16000,
    };
    if (options.reasoningEffort !== null) payload.reasoning_effort = options.reasoningEffort || "medium";
    return { ...payload, ...(options.extra || {}) };
  }

  /** Parse the SSE body into text; throws on a stream_error event. */
  function parseSse(text) {
    let out = "";
    for (const rawLine of text.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line.startsWith("data:")) continue;      // ": connected" keep-alive
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      let event;
      try { event = JSON.parse(data); } catch (_) { continue; }
      if (event.stream_error) {
        const err = event.stream_error;
        throw new OldLLMError(err.code || "ERROR", err.error || "", err.status);
      }
      for (const choice of event.choices || []) {
        const piece = choice?.delta?.content;
        if (piece) out += piece;
      }
    }
    return out;
  }

  /** Raw call: returns { status, text } with the untouched SSE body. */
  async function raw(payload, path = CHAT_PATH) {
    const token = await turnstileToken();
    const response = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json", "x-turnstile-token": token },
      body: JSON.stringify(payload),
    });
    return { status: response.status, text: await response.text() };
  }

  /** Send prepared messages; resolves to the reply text. */
  async function ask(messages, options = {}) {
    const payload = buildPayload(messages, options);
    const { status, text } = await raw(payload);
    try {
      return parseSse(text);
    } catch (error) {
      if (error.code === "ACCESS_DENIED" && MODELS[payload.model]?.verifiedOnly) {
        error.message += ` — ${payload.model} needs a signed-in account; try gpt-6-luna`;
      }
      if (!error.status) error.status = status;
      throw error;
    }
  }

  async function chat(message, options = {}) {
    return ask([{ role: "user", content: message }], options);
  }

  window.oldllm = { chat, ask, raw, parseSse, buildPayload, models: MODELS, OldLLMError };
  console.log("oldllm ready — try: await oldllm.chat('hello')");
})();
