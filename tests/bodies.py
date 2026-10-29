"""Fixture text for the tests, copied from real response bodies."""

CONNECTED = ": connected"

OK_BODY = (
    ": connected\n"
    "\n"
    'data: {"choices":[{"delta":{"content":"he"}}]}\n'
    "\n"
    'data: {"choices":[{"delta":{"content":"llo"}}]}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)

BOT_FAILED_BODY = (
    ": connected\n\n"
    'data: {"stream_error":{"error":"Access denied","code":"BOT_CHECK_FAILED","status":403}}\n\n'
    "data: [DONE]\n\n"
)

ACCESS_DENIED_BODY = (
    ": connected\n\n"
    'data: {"stream_error":{"error":"Access denied","code":"ACCESS_DENIED","status":403}}\n\n'
    "data: [DONE]\n\n"
)
