import uuid

import pytest

from theoldllm.exceptions import UnknownModelError
from theoldllm.models import Models
from theoldllm.payload import DEFAULT_SYSTEM, build_payload, normalise_messages

PROMPT = [{"role": "user", "content": "hi"}]


def test_payload_matches_the_fields_the_site_sends():
    payload = build_payload("gpt-6-luna", PROMPT)
    assert payload == {
        "model": "gpt-6-luna",
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": "hi"},
        ],
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "provider": "openai",
        "threadId": payload["threadId"],
        "max_tokens": 16000,
        "reasoning_effort": "medium",
    }
    uuid.UUID(payload["threadId"])  # a real uuid4, not a placeholder


def test_stream_flag_is_always_true_because_the_server_ignores_false():
    assert build_payload("gpt-6-luna", PROMPT, extra={"stream": False})["stream"] is False
    assert build_payload("gpt-6-luna", PROMPT)["stream"] is True


def test_system_prompt_is_not_duplicated_when_the_caller_supplies_one():
    payload = build_payload(
        "gpt-6-luna",
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
    )
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user"]
    assert payload["messages"][0]["content"] == "be terse"


def test_system_can_be_disabled():
    payload = build_payload("gpt-6-luna", PROMPT, system=None)
    assert payload["messages"] == PROMPT


def test_provider_comes_from_the_catalogue():
    assert build_payload("claude-opus-4-7", PROMPT)["provider"] == "anthropic"
    assert build_payload("gemini-3.8-flash", PROMPT)["provider"] == "google"
    assert build_payload("grok-4.3-latest", PROMPT)["provider"] == "xai"


def test_reasoning_effort_none_omits_the_field():
    payload = build_payload("gpt-6-luna", PROMPT, reasoning_effort=None)
    assert "reasoning_effort" not in payload


def test_thread_id_is_reusable_and_overridable():
    payload = build_payload("gpt-6-luna", PROMPT, thread_id="thread-1")
    assert payload["threadId"] == "thread-1"


def test_max_tokens_can_be_overridden_and_defaults_to_the_site_budget():
    assert build_payload("gpt-6-luna", PROMPT, max_tokens=64)["max_tokens"] == 64
    assert build_payload("gpt-6-luna", PROMPT)["max_tokens"] == 16000


def test_extra_fields_are_merged_last():
    payload = build_payload("gpt-6-luna", PROMPT, extra={"webSearch": True, "temperature": 0.1})
    assert payload["webSearch"] is True
    assert payload["temperature"] == 0.1


def test_model_object_can_be_passed_directly():
    model = Models.by_id("gpt-6-sol")
    assert build_payload(model, PROMPT)["model"] == "gpt-6-sol"


def test_unknown_model_raises_with_the_catalogue_in_the_message():
    with pytest.raises(UnknownModelError) as info:
        build_payload("claude-9-opus", PROMPT)
    assert "gpt-6-luna" in str(info.value)
    assert info.value.model == "claude-9-opus"


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": "user", "content": None}],
        ["not a mapping"],
    ],
)
def test_message_validation_rejects_garbage(messages):
    with pytest.raises(ValueError):
        normalise_messages(messages)


def test_messages_are_copied_not_mutated():
    original = [{"role": "user", "content": "hi"}]
    normalise_messages(original)
    assert original == [{"role": "user", "content": "hi"}]
