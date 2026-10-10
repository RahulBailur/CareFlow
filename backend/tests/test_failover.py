"""The LLM fallback chain: the order of tiers, the circuit breaker, and which tier answered."""

import json
from typing import Any

import httpx
import pytest
from httpx import AsyncClient

from agents.intent_classifier import IntentClassifier
from agents.orchestrator import run_text_turn
from services.failover import CircuitBreaker, FailoverLLM
from services.llm import (
    GeminiLLM,
    LLMResponse,
    LLMUnavailable,
    Message,
    OllamaLLM,
    ToolCall,
    ToolSpec,
)
from services.redis_client import ResponseCache
from tests.helpers import auth, make_user
from tests.test_chat import seed_hospital

KEYWORDS_ONLY = IntentClassifier()

# --- circuit breaker --------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def breaker(threshold: int = 3, cooldown: float = 60.0) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    return CircuitBreaker(threshold, cooldown, clock), clock


def test_the_breaker_starts_closed() -> None:
    circuit, _ = breaker()

    assert circuit.state == "closed" and circuit.allows()


def test_it_opens_only_after_the_threshold_of_failures_in_a_row() -> None:
    circuit, _ = breaker(threshold=3)

    circuit.record_failure()
    circuit.record_failure()
    assert circuit.allows()

    circuit.record_failure()
    assert circuit.state == "open" and not circuit.allows()


def test_a_success_in_between_starts_the_count_again() -> None:
    circuit, _ = breaker(threshold=3)

    for _ in range(5):
        circuit.record_failure()
        circuit.record_failure()
        circuit.record_success()

    assert circuit.state == "closed"


def test_an_open_breaker_lets_calls_through_again_after_the_cool_down() -> None:
    circuit, clock = breaker(threshold=1, cooldown=60)
    circuit.record_failure()

    clock.now = 59.9
    assert not circuit.allows()
    clock.now = 60.0
    assert circuit.state == "half_open" and circuit.allows()


def test_a_success_after_the_cool_down_closes_it() -> None:
    circuit, clock = breaker(threshold=3, cooldown=60)
    for _ in range(3):
        circuit.record_failure()
    clock.now = 61

    circuit.record_success()

    assert circuit.state == "closed"
    circuit.record_failure()
    assert circuit.allows(), "the failure count started again from zero"


def test_one_failure_after_the_cool_down_opens_it_for_another_full_cool_down() -> None:
    circuit, clock = breaker(threshold=3, cooldown=60)
    for _ in range(3):
        circuit.record_failure()
    clock.now = 61

    circuit.record_failure()

    assert circuit.state == "open"
    clock.now = 120
    assert not circuit.allows()
    clock.now = 121
    assert circuit.allows()


# --- the chain --------------------------------------------------------------------------


class Tier:
    def __init__(self, name: str, error: str | None = None, text: str = "") -> None:
        self.name, self.error, self.text = name, error, text or f"from {name}"
        self.calls: list[list[Message]] = []

    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self.error:
            raise LLMUnavailable(self.error)
        return LLMResponse(text=self.text)


def chain(*tiers: Tier, threshold: int = 3, cooldown: float = 60.0) -> tuple[FailoverLLM, Clock]:
    clock = Clock()
    return FailoverLLM([(t, CircuitBreaker(threshold, cooldown, clock)) for t in tiers]), clock


USER = [Message("user", "hello")]


async def test_the_first_tier_answers_when_it_can() -> None:
    gemini, ollama = Tier("gemini"), Tier("ollama")
    llm, _ = chain(gemini, ollama)

    response = await llm.generate("s", USER)

    assert response.text == "from gemini" and response.provider == "gemini"
    assert ollama.calls == []


async def test_the_next_tier_answers_when_the_first_fails() -> None:
    gemini, ollama = Tier("gemini", "Gemini returned HTTP 429"), Tier("ollama")
    llm, _ = chain(gemini, ollama)

    response = await llm.generate("s", USER)

    assert response.provider == "ollama"


async def test_when_every_tier_fails_the_reasons_are_reported() -> None:
    llm, _ = chain(Tier("gemini", "HTTP 429"), Tier("ollama", "ConnectError"))

    with pytest.raises(LLMUnavailable) as raised:
        await llm.generate("s", USER)

    assert "gemini: HTTP 429" in str(raised.value) and "ollama: ConnectError" in str(raised.value)


async def test_after_three_failures_gemini_is_skipped_without_being_called() -> None:
    gemini, ollama = Tier("gemini", "HTTP 429"), Tier("ollama")
    llm, _ = chain(gemini, ollama, threshold=3)

    for _ in range(3):
        await llm.generate("s", USER)
    assert len(gemini.calls) == 3 and llm.breaker_states()["gemini"] == "open"

    for _ in range(5):
        assert (await llm.generate("s", USER)).provider == "ollama"
    assert len(gemini.calls) == 3, "an open circuit costs no request and no wait"


async def test_gemini_is_tried_again_after_the_cool_down_and_recovers() -> None:
    gemini, ollama = Tier("gemini", "HTTP 429"), Tier("ollama")
    llm, clock = chain(gemini, ollama, threshold=3, cooldown=60)
    for _ in range(3):
        await llm.generate("s", USER)

    clock.now = 61
    gemini.error = None
    response = await llm.generate("s", USER)

    assert response.provider == "gemini" and llm.breaker_states()["gemini"] == "closed"


async def test_each_tier_has_its_own_breaker() -> None:
    gemini, ollama = Tier("gemini", "HTTP 429"), Tier("ollama", "down")
    llm, _ = chain(gemini, ollama, threshold=2)

    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            await llm.generate("s", USER)
    ollama.error = None
    gemini.error = None

    with pytest.raises(LLMUnavailable, match="circuit open"):
        await llm.generate("s", USER)
    assert llm.breaker_states() == {"gemini": "open", "ollama": "open"}


async def test_an_empty_chain_is_simply_unavailable() -> None:
    with pytest.raises(LLMUnavailable, match="no LLM provider"):
        await FailoverLLM([]).generate("s", USER)


# --- moving between providers mid-turn --------------------------------------------------


async def test_a_gemini_turn_is_not_replayed_verbatim_to_another_provider() -> None:
    """Gemini's own turn format means nothing to Ollama: it gets the plain form."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    ollama = OllamaLLM("http://ollama:11434", "tiny", transport=httpx.MockTransport(handler))
    gemini_turn = Message(
        "assistant",
        tool_calls=[ToolCall("find_slots", {"date": "2026-10-12"})],
        raw={"role": "model", "parts": [{"functionCall": {}, "thoughtSignature": "opaque"}]},
        provider="gemini",
    )

    await ollama.generate(
        "sys",
        [
            Message("user", "book"),
            gemini_turn,
            Message("tool", tool_name="find_slots", tool_result={"n": 1}),
        ],
    )

    assert "thoughtSignature" not in json.dumps(seen["body"])
    assert seen["body"]["messages"][2]["tool_calls"] == [
        {"function": {"name": "find_slots", "arguments": {"date": "2026-10-12"}}}
    ]


async def test_another_providers_turn_is_rebuilt_for_gemini() -> None:
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["contents"])
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    gemini = GeminiLLM("k", "m", transport=httpx.MockTransport(handler))
    ollama_turn = Message("assistant", "one moment", raw={"anything": True}, provider="ollama")

    await gemini.generate("s", [Message("user", "hi"), ollama_turn, Message("user", "go on")])

    assert sent[0][1] == {"role": "model", "parts": [{"text": "one moment"}]}


# --- the Ollama provider ----------------------------------------------------------------

TOOL = ToolSpec(
    "find_slots",
    "List open slots.",
    {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]},
)


def ollama(handler: Any, model: str = "tiny") -> OllamaLLM:
    return OllamaLLM("http://ollama:11434/", model, transport=httpx.MockTransport(handler))


async def test_ollama_request_carries_the_system_prompt_messages_and_tools() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Hello"}})

    response = await ollama(handler).generate("Be brief.", [Message("user", "hi")], [TOOL])

    assert response.text == "Hello" and response.provider == "ollama"
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert seen["body"]["model"] == "tiny" and seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    assert seen["body"]["tools"][0]["function"]["name"] == "find_slots"


@pytest.mark.parametrize("arguments", [{"date": "2026-10-12"}, '{"date": "2026-10-12"}'])
async def test_ollama_tool_calls_are_parsed_whether_arguments_are_an_object_or_a_string(
    arguments: Any,
) -> None:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "find_slots", "arguments": arguments}}],
    }

    response = await ollama(lambda r: httpx.Response(200, json={"message": message})).generate(
        "s", [Message("user", "book")], [TOOL]
    )

    assert response.tool_calls == [ToolCall("find_slots", {"date": "2026-10-12"})]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(404, json={"error": "model not found"}),
        httpx.Response(200, json={}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"message": {"tool_calls": [{"function": {"arguments": "{"}}]}}),
    ],
)
async def test_ollama_failures_become_llm_unavailable(response: httpx.Response) -> None:
    with pytest.raises(LLMUnavailable):
        await ollama(lambda request: response).generate("s", [Message("user", "hi")])


async def test_ollama_not_running_becomes_llm_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailable, match="ConnectError"):
        await ollama(handler).generate("s", [Message("user", "hi")])


async def test_ollama_without_a_model_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    with pytest.raises(LLMUnavailable):
        await ollama(handler, model="").generate("s", [Message("user", "hi")])


async def test_a_slow_gemini_request_times_out_as_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    gemini = GeminiLLM("k", "m", transport=httpx.MockTransport(handler), timeout_s=0.01)

    with pytest.raises(LLMUnavailable, match="ReadTimeout"):
        await gemini.generate("s", [Message("user", "hi")])


# --- which tier answered a turn ---------------------------------------------------------


@pytest.mark.parametrize(
    ("gemini_error", "ollama_error", "answered_by", "used_llm"),
    [
        (None, None, "gemini", True),
        ("HTTP 429", None, "ollama", True),
        ("HTTP 429", "ConnectError", "rules", False),
    ],
)
async def test_a_turn_records_the_tier_that_answered(
    client: AsyncClient,
    gemini_error: str | None,
    ollama_error: str | None,
    answered_by: str,
    used_llm: bool,
) -> None:
    await seed_hospital()
    llm, _ = chain(Tier("gemini", gemini_error), Tier("ollama", ollama_error))

    result = await run_text_turn(
        await make_user(), "What are the OPD timings?", "s-1", llm, KEYWORDS_ONLY
    )

    assert result.answered_by == answered_by and result.used_llm is used_llm
    assert result.reply  # the patient always gets an answer


async def test_a_cache_hit_is_recorded_as_the_cache_tier(client: AsyncClient) -> None:
    import fakeredis

    await seed_hospital()
    cache = ResponseCache(fakeredis.FakeAsyncRedis(decode_responses=True), 600)
    patient, question = await make_user(), "What are the OPD timings?"
    llm, _ = chain(Tier("gemini"))
    await run_text_turn(patient, question, "s-1", llm, KEYWORDS_ONLY, cache=cache)

    again = await run_text_turn(patient, question, "s-2", llm, KEYWORDS_ONLY, cache=cache)

    assert again.answered_by == "cache"


async def test_the_whole_chain_down_still_answers_every_kind_of_question(
    client: AsyncClient,
) -> None:
    await seed_hospital()
    llm, _ = chain(Tier("gemini", "HTTP 429"), Tier("ollama", "ConnectError"))
    patient = await make_user()

    for question in (
        "What are the OPD timings?",
        "I have chest pain",
        "Show me my last prescription",
        "Cancel my appointment",
        "hello",
    ):
        result = await run_text_turn(patient, question, "s-1", llm, KEYWORDS_ONLY)
        assert result.reply and result.answered_by == "rules"


async def test_the_chat_api_reports_the_tier(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat", json={"message": "hello"}, headers=auth(await make_user())
    )

    assert response.json()["answered_by"] == "rules"  # tests run with no model at all


async def test_quota_errors_do_not_cost_every_later_turn_a_wait(client: AsyncClient) -> None:
    """The PRD's case: after three 429s in a row, Gemini is not asked for 60 seconds."""
    await seed_hospital()
    gemini = Tier("gemini", "Gemini returned HTTP 429")
    llm, clock = chain(gemini, threshold=3, cooldown=60)
    patient = await make_user()

    for _ in range(6):
        await run_text_turn(patient, "What are the OPD timings?", "s-1", llm, KEYWORDS_ONLY)
    assert len(gemini.calls) == 3

    clock.now = 61
    gemini.error = None
    result = await run_text_turn(patient, "What are the OPD timings?", "s-1", llm, KEYWORDS_ONLY)
    assert result.answered_by == "gemini"
