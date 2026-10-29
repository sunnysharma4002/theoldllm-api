import pytest

from theoldllm.exceptions import APIError, AccessDeniedError, BotCheckError
from theoldllm.streaming import SSEDecoder, iter_sse, parse_sse_line, text_of
from tests.bodies import ACCESS_DENIED_BODY, BOT_FAILED_BODY, OK_BODY


def test_keepalive_and_blank_lines_are_ignored():
    assert parse_sse_line(": connected") is None
    assert parse_sse_line("") is None
    assert parse_sse_line("event: message") is None


def test_data_prefix_without_space_is_accepted():
    chunk = parse_sse_line('data:{"choices":[{"delta":{"content":"x"}}]}')
    assert chunk is not None
    assert chunk.content == "x"


def test_done_sentinel_finishes_the_stream():
    chunk = parse_sse_line("data: [DONE]")
    assert chunk is not None
    assert chunk.is_done and chunk.finish_reason == "stop"


def test_full_body_yields_deltas_in_order_then_stops():
    chunks = list(iter_sse(OK_BODY))
    assert [c.content for c in chunks] == ["he", "llo", ""]
    assert chunks[-1].is_done
    assert text_of(OK_BODY) == "hello"


def test_crlf_and_trailing_whitespace_are_tolerated():
    body = OK_BODY.replace("\n", "\r\n") + "  \r\n"
    assert text_of(body) == "hello"


def test_bot_check_failure_raises_typed_error():
    with pytest.raises(BotCheckError) as info:
        text_of(BOT_FAILED_BODY)
    error = info.value
    assert error.code == "BOT_CHECK_FAILED"
    assert error.status_code == 403
    assert "single use" in str(error)


def test_access_denied_raises_typed_error():
    with pytest.raises(AccessDeniedError) as info:
        text_of(ACCESS_DENIED_BODY)
    assert info.value.code == "ACCESS_DENIED"
    assert info.value.status_code == 403


def test_unknown_stream_error_code_becomes_api_error():
    body = 'data: {"stream_error":{"error":"boom","code":"WHATEVER","status":500}}'
    with pytest.raises(APIError) as info:
        list(iter_sse(body))
    assert info.value.code == "WHATEVER"
    assert not isinstance(info.value, BotCheckError)


def test_malformed_json_is_skipped_not_fatal():
    body = 'data: {"choices":[{"delta":{"content":"a"}}]}\ndata: {oops\ndata: [DONE]'
    assert text_of(body) == "a"


def test_message_payload_is_accepted_as_well_as_delta():
    body = 'data: {"choices":[{"message":{"role":"assistant","content":"full"}}]}'
    assert text_of(body) == "full"


def test_reasoning_and_usage_are_exposed():
    body = (
        'data: {"model":"gpt-6-luna","usage":{"total_tokens":3},'
        '"choices":[{"delta":{"reasoning_content":"think","content":"ok"}}]}'
    )
    (chunk,) = list(iter_sse(body))
    assert chunk.reasoning_content == "think"
    assert chunk.content == "ok"
    assert chunk.model == "gpt-6-luna"
    assert chunk.usage == {"total_tokens": 3}


def test_incremental_decoder_handles_split_lines():
    decoder = SSEDecoder()
    parts = [
        ": conn",
        "ected\n\ndata: {\"choices\":[{\"delta\"",
        ':{"content":"hi"}}]}\n\ndata: [DONE]\n',
    ]
    collected = []
    for part in parts:
        collected.extend(decoder.feed(part))
    assert "".join(c.content for c in collected) == "hi"
    assert collected[-1].is_done
    assert decoder.flush() == []
