"""Command line entry point: ``python -m theoldllm``.

    python -m theoldllm models
    python -m theoldllm chat "hello" --model gpt-6-luna --stream
    python -m theoldllm serve --port 11434
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional, Sequence

from . import __version__
from .client import TheOldLLM
from .exceptions import TheOldLLMError
from .models import Models, VERIFIED_ONLY


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="theoldllm", description="Unofficial TheOldLLM client")
    parser.add_argument("--version", action="version", version=f"theoldllm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("models", help="print the model catalogue")
    listing.add_argument("--json", action="store_true", help="machine-readable output")
    listing.add_argument("--all", action="store_true", help="include hidden/disabled models")

    chat = sub.add_parser("chat", help="send one prompt and print the reply")
    chat.add_argument("prompt", nargs="+", help="prompt text")
    chat.add_argument("--model", "-m", default="gpt-6-luna")
    chat.add_argument("--system", "-s", default=None, help="override the system prompt")
    chat.add_argument("--max-tokens", type=int, default=None)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high", "max", "none"])
    chat.add_argument("--stream", action="store_true", help="print deltas as they are decoded")
    chat.add_argument("--headless", action="store_true", help="try headless (Turnstile usually times out)")
    chat.add_argument("--storage", default=None, help="cookie/localStorage JSON to reuse")
    chat.add_argument("--verbose", "-v", action="store_true")

    serve = sub.add_parser("serve", help="run the OpenAI-compatible HTTP facade")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=11434)
    serve.add_argument("--headless", action="store_true")
    serve.add_argument(
        "--storage",
        default=os.environ.get("THEOLDLLM_STORAGE") or None,
        help="cookie/localStorage JSON to reuse (env: THEOLDLLM_STORAGE)",
    )
    serve.add_argument(
        "--channel",
        default=os.environ.get("THEOLDLLM_CHANNEL") or None,
        choices=["chrome", "chromium", "msedge"],
        help="use a real installed browser instead of bundled Chromium (env: THEOLDLLM_CHANNEL)",
    )
    serve.add_argument(
        "--token-timeout",
        type=float,
        default=float(os.environ.get("THEOLDLLM_TOKEN_TIMEOUT", "45")),
        help="seconds to wait for a Turnstile token (env: THEOLDLLM_TOKEN_TIMEOUT)",
    )
    serve.add_argument("--verbose", "-v", action="store_true")
    return parser


def _cmd_models(args: argparse.Namespace) -> int:
    models = Models.ALL if args.all else Models.picker()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": m.id,
                        "name": m.name,
                        "provider": m.provider.value,
                        "upstream": m.upstream_model,
                        "requires_account": m.requires_account,
                        "anonymous": m.anonymous,
                        "hidden": m.hidden,
                        "unavailable": m.unavailable,
                    }
                    for m in models
                ],
                indent=2,
            )
        )
        return 0
    width = max(len(m.id) for m in models)
    for model in models:
        flags = []
        if model.requires_account:
            flags.append("account required")
        if model.hidden:
            flags.append("hidden")
        if model.unavailable:
            flags.append("disabled")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{model.id:<{width}}  {model.name} ({model.provider.value}){suffix}")
    print(
        f"\n{len(models)} models · {len(VERIFIED_ONLY)} need a signed-in account · "
        f"anonymous: {', '.join(m.id for m in Models.anonymous())}",
        file=sys.stderr,
    )
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    system = args.system
    if system is None:
        from .payload import DEFAULT_SYSTEM

        system = DEFAULT_SYSTEM
    options = {
        "system": system or None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning_effort": None if args.reasoning_effort == "none" else args.reasoning_effort,
    }
    messages = [{"role": "user", "content": " ".join(args.prompt)}]
    with TheOldLLM(headless=args.headless, storage_path=args.storage) as bot:
        if not args.stream:
            print(bot.chat(args.model, messages, **options))
            return 0
        for chunk in bot.chat_stream(args.model, messages, **options):
            if chunk.reasoning_content:
                print(chunk.reasoning_content, end="", flush=True, file=sys.stderr)
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()
    return 0


def _proxy_from_env() -> dict[str, Any] | None:
    """Parse ``THEOLDLLM_PROXY``: a URL, or JSON for credentials."""
    raw = os.environ.get("THEOLDLLM_PROXY")
    if not raw:
        return None
    if raw.lstrip().startswith("{"):
        return json.loads(raw)
    return {"server": raw}


def _cmd_serve(args: argparse.Namespace) -> int:
    import asyncio

    from .proxy import serve

    options: dict[str, Any] = {
        "headless": args.headless,
        "storage_path": args.storage,
        "token_timeout": args.token_timeout,
    }
    if args.channel:
        options["channel"] = args.channel
    proxy = _proxy_from_env()
    if proxy:
        options["proxy"] = proxy

    try:
        asyncio.run(serve(host=args.host, port=args.port, **options))
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    handlers = {"models": _cmd_models, "chat": _cmd_chat, "serve": _cmd_serve}
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        return 130
    except TheOldLLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
