"""The Gemini provider, against a fake HTTP transport. No real API call is ever made here."""

import json
from typing import Any

import httpx
import pytest

from services.llm import GeminiLLM, LLMUnavailable, Message, MockLLM, ToolCall, ToolSpec, get_llm

TOOL = ToolSpec(
    "find_slots",
    "List open slots.",
    {
        "type": "object",
        "properties": {"date": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["date"],
        "additionalProperties": False,
    },
)


def gemini(handler: Any, key: str = "test-key", model: str = "test-model") -> GeminiLLM:
    return GeminiLLM(key, model, transport=httpx.MockTransport(handler))


def reply(*parts: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"role": "model", "parts": parts}}]}
    )


async def test_request_carries_the_system_prompt_messages_and_tools() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return reply({"text": "Hello"})

    response = await gemini(handler).generate("Be brief.", [Message("user", "hi")], [TOOL])

    assert response.text == "Hello" and response.tool_calls == []
    assert seen["url"].endswith("/models/test-model:generateContent")
    assert "test-key" not in seen["url"] and seen["key"] == "test-key"
    assert seen["body"]["system_instruction"] == {"parts": [{"text": "Be brief."}]}
    assert seen["body"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    declared = seen["body"]["tools"][0]["functionDeclarations"][0]
    assert declared["name"] == "find_slots"
    assert declared["parameters"] == {
        "type": "OBJECT",
        "properties": {"date": {"type": "STRING"}, "limit": {"type": "INTEGER"}},
        "required": ["date"],
    }


async def test_no_tools_key_is_sent_when_there_are_no_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "tools" not in json.loads(request.content)
        return reply({"text": "ok"})

    await gemini(handler).generate("s", [Message("user", "hi")])


async def test_function_calls_are_parsed() -> None:
    response = await gemini(
        lambda request: reply(
            {"functionCall": {"name": "find_slots", "args": {"date": "2026-10-12"}}}
        )
    ).generate("s", [Message("user", "book")], [TOOL])

    assert response.tool_calls == [ToolCall("find_slots", {"date": "2026-10-12"})]
    assert response.text == ""


async def test_thought_parts_are_not_shown_as_the_reply() -> None:
    response = await gemini(
        lambda request: reply({"text": "internal reasoning", "thought": True}, {"text": "Hi."})
    ).generate("s", [Message("user", "hi")])

    assert response.text == "Hi."


async def test_a_tool_round_trip_replays_the_models_own_turn_and_groups_results() -> None:
    raw_turn = {
        "role": "model",
        "parts": [
            {"functionCall": {"name": "a", "args": {}}, "thoughtSignature": "opaque"},
            {"functionCall": {"name": "b", "args": {}}},
        ],
    }
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["contents"])
        return reply({"text": "done"})

    await gemini(handler).generate(
        "s",
        [
            Message("user", "go"),
            Message("assistant", tool_calls=[ToolCall("a", {}), ToolCall("b", {})], raw=raw_turn),
            Message("tool", tool_name="a", tool_result={"x": 1}),
            Message("tool", tool_name="b", tool_result={"y": 2}),
        ],
    )

    assert sent[0] == [
        {"role": "user", "parts": [{"text": "go"}]},
        raw_turn,  # verbatim, signature included
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "a", "response": {"x": 1}}},
                {"functionResponse": {"name": "b", "response": {"y": 2}}},
            ],
        },
    ]


async def test_remembered_assistant_turns_without_a_raw_form_are_rebuilt() -> None:
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["contents"])
        return reply({"text": "ok"})

    await gemini(handler).generate(
        "s", [Message("user", "a"), Message("assistant", "b"), Message("user", "c")]
    )

    assert sent[0][1] == {"role": "model", "parts": [{"text": "b"}]}


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
async def test_http_errors_become_llm_unavailable_without_leaking_the_body(status: int) -> None:
    llm = gemini(lambda request: httpx.Response(status, text="secret-echo test-key"))

    with pytest.raises(LLMUnavailable) as raised:
        await llm.generate("s", [Message("user", "hi")])

    assert str(status) in str(raised.value)
    assert "secret-echo" not in str(raised.value) and "test-key" not in str(raised.value)


async def test_network_failures_become_llm_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(LLMUnavailable):
        await gemini(handler).generate("s", [Message("user", "hi")])


@pytest.mark.parametrize(
    "body", [{}, {"candidates": []}, {"candidates": [{"finishReason": "SAFETY"}]}, "not json"]
)
async def test_a_response_with_no_candidate_becomes_llm_unavailable(body: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body)

    with pytest.raises(LLMUnavailable):
        await gemini(handler).generate("s", [Message("user", "hi")])


@pytest.mark.parametrize(("key", "model"), [("", "m"), ("k", ""), ("", "")])
async def test_a_missing_key_or_model_fails_before_any_request(key: str, model: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    with pytest.raises(LLMUnavailable):
        await gemini(handler, key, model).generate("s", [Message("user", "hi")])


async def test_the_mock_provider_is_always_unavailable() -> None:
    with pytest.raises(LLMUnavailable):
        await MockLLM().generate("s", [Message("user", "hi")])


def test_tests_never_get_a_real_provider() -> None:
    get_llm.cache_clear()

    assert isinstance(get_llm(), MockLLM)
