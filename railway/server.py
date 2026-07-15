#!/usr/bin/env python3
"""Railway server - TheOldLLM OpenAI-compatible proxy using curl_cffi."""

import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiohttp import web
from theoldllm.direct_client import DirectTheOldLLM
from theoldllm.models import Models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("railway-server")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8080))

client = DirectTheOldLLM()
ready = True  # No warmup needed - curl_cffi connects on first request


async def chat_completions(request):
    try:
        body = await request.json()
        model_id = body.get("model", "gpt-5-mini-aichat")
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")
        top_p = body.get("top_p")

        logger.info(f"Chat: model={model_id}, stream={stream}, msgs={len(messages)}")

        if stream:
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
            )
            await resp.prepare(request)
            for chunk in client.chat_stream(
                model=model_id, messages=messages,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            ):
                await resp.write(_sse_chunk(chunk, model_id).encode())
                if chunk.is_done:
                    await resp.write(_sse_done(model_id).encode())
                    break
            await resp.write(b"data: [DONE]\n\n")
            return resp
        else:
            content = ""
            for chunk in client.chat_stream(
                model=model_id, messages=messages,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            ):
                if chunk.content:
                    content += chunk.content
            return web.json_response({
                "id": "chatcmpl-theoldllm",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            }, headers={"Access-Control-Allow-Origin": "*"})

    except Exception as e:
        logger.exception("Chat error")
        return web.json_response({"error": {"message": str(e), "type": type(e).__name__}}, status=500, headers={"Access-Control-Allow-Origin": "*"})


async def list_models(request):
    seen = set()
    data = []
    for m in Models.ALL:
        if m.id not in seen:
            seen.add(m.id)
            data.append({"id": m.id, "object": "model", "created": 0, "owned_by": m.provider.value, "root": m.id})
    return web.json_response({"object": "list", "data": data}, headers={"Access-Control-Allow-Origin": "*"})


async def health(request):
    return web.json_response({"status": "ok", "model_count": len(Models.ALL)}, headers={"Access-Control-Allow-Origin": "*"})


async def cors(request):
    return web.Response(headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    })


def _sse_chunk(chunk, model_id):
    d = {"id": "chatcmpl-theoldllm", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
    delta = {}
    if chunk.content:
        delta["content"] = chunk.content
    if chunk.reasoning_content:
        delta["reasoning_content"] = chunk.reasoning_content
    if chunk.finish_reason:
        d["choices"][0]["finish_reason"] = chunk.finish_reason
    d["choices"][0]["delta"] = delta
    return f"data: {json.dumps(d)}\n\n"


def _sse_done(model_id):
    return f"data: {json.dumps({'id': 'chatcmpl-theoldllm', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"


async def main():
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", list_models)
    app.router.add_get("/health", health)
    app.router.add_route("OPTIONS", "/v1/chat/completions", cors)
    app.router.add_route("OPTIONS", "/v1/models", cors)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()

    logger.info(f"Server ready on {HOST}:{PORT}")
    logger.info(f"OpenAI endpoint: http://{HOST}:{PORT}/v1")
    logger.info(f"Models: {len(Models.ALL)}")

    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown")
