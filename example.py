"""Two end-to-end examples against the real site.

Run:  python example.py
A Chromium window opens for the length of each example; close it, not the
process, if you want to stop early.
"""

import asyncio

from theoldllm import AsyncTheOldLLM, Models, TheOldLLM


def list_models() -> None:
    for model in Models.picker():
        note = "  (needs a signed-in account)" if model.requires_account else ""
        print(f"{model.id:22s} {model.name:20s} [{model.provider.value}]{note}")
    print()


def sync_example() -> None:
    with TheOldLLM() as bot:
        print("streaming gpt-6-luna:")
        for chunk in bot.chat_stream(
            "gpt-6-luna",
            [
                TheOldLLM.system_message("You are concise."),
                TheOldLLM.user_message("Three facts about the Moon, one line each."),
            ],
            max_tokens=300,
        ):
            print(chunk.content, end="", flush=True)
        print("\n")

        answer = bot.chat("gpt-5.5-thinking", [{"role": "user", "content": "What is 2+2?"}])
        print(f"non-streaming gpt-5.5-thinking: {answer}\n")


async def async_example() -> None:
    async with AsyncTheOldLLM() as bot:
        answer = await bot.chat(
            "gemini-3.8-flash",
            [{"role": "user", "content": "Write a haiku about coding."}],
            max_tokens=200,
        )
        print(f"async gemini-3.8-flash:\n{answer}\n")

        # A model that needs a signed-in account fails with a typed error the
        # caller can branch on instead of an empty string.
        from theoldllm import AccessDeniedError

        try:
            await bot.chat("claude-opus-4-7", [{"role": "user", "content": "hi"}])
        except AccessDeniedError as error:
            print(f"as expected: {error}\n")


if __name__ == "__main__":
    list_models()
    sync_example()
    asyncio.run(async_example())
