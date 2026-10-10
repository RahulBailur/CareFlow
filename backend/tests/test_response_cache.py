"""The Redis response cache: what may be shared between patients, and what never is."""

from typing import Any

import fakeredis
import pytest
from httpx import AsyncClient

from agents.intent_classifier import IntentClassifier
from agents.intent_data import Intent
from agents.orchestrator import run_text_turn
from models.appointment import AppointmentStatus
from models.conversation_turn import ConversationTurn
from models.hospital_config import HospitalConfig
from services.redis_client import KEY_PREFIX, ResponseCache, reply_key
from tests.helpers import make_appointment, make_doctor, make_user, slot
from tests.test_chat import ScriptedLLM, call, say, seed_hospital

KEYWORDS_ONLY = IntentClassifier()


@pytest.fixture
def redis() -> Any:
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def cache(redis: Any) -> ResponseCache:
    return ResponseCache(redis, ttl_seconds=3600)


async def ask(user: Any, text: str, llm: Any, cache: ResponseCache, session: str = "s-1") -> Any:
    return await run_text_turn(user, text, session, llm, KEYWORDS_ONLY, cache=cache)


async def test_a_repeated_hospital_question_is_answered_without_the_llm(
    client: AsyncClient, cache: ResponseCache
) -> None:
    await seed_hospital()
    first_patient, second_patient = await make_user(), await make_user()
    llm = ScriptedLLM(call("get_hospital_info"), say("We are open 9 to 1, Monday to Saturday."))

    first = await ask(first_patient, "What are the OPD timings?", llm, cache)
    again = await ask(second_patient, "What are the OPD timings?", ScriptedLLM(), cache, "s-2")

    assert first.used_llm and not first.cached
    assert again.cached and not again.used_llm
    assert again.reply == "We are open 9 to 1, Monday to Saturday."
    assert again.intent == Intent.SUPPORT and again.reply_language == "en"


async def test_case_spacing_and_punctuation_do_not_miss_the_cache(
    client: AsyncClient, cache: ResponseCache
) -> None:
    await seed_hospital()
    patient = await make_user()
    await ask(patient, "What are the OPD timings?", ScriptedLLM(say("9 to 1.")), cache)

    again = await ask(patient, "  what are the opd TIMINGS  ", ScriptedLLM(), cache)

    assert again.cached and again.reply == "9 to 1."


async def test_a_different_question_or_language_is_a_different_entry(
    client: AsyncClient, cache: ResponseCache
) -> None:
    await seed_hospital()
    patient = await make_user()
    await ask(patient, "What are the OPD timings?", ScriptedLLM(say("9 to 1.")), cache)

    other = await ask(
        patient, "Where is the cardiology department?", ScriptedLLM(say("First floor.")), cache
    )
    hindi = await ask(patient, "ओपीडी का समय क्या है?", ScriptedLLM(say("नौ से एक।")), cache)

    assert not other.cached and other.reply == "First floor."
    assert not hindi.cached and hindi.reply == "नौ से एक।"


async def test_a_cached_reply_still_joins_the_conversation(
    client: AsyncClient, cache: ResponseCache
) -> None:
    await seed_hospital()
    patient = await make_user()
    await ask(patient, "What are the OPD timings?", ScriptedLLM(say("9 to 1.")), cache)

    await ask(patient, "What are the OPD timings?", ScriptedLLM(), cache, "another-session")

    remembered = await ConversationTurn.find(
        ConversationTurn.session_id == "another-session"
    ).to_list()
    assert [turn.text for turn in remembered] == ["What are the OPD timings?", "9 to 1."]


async def test_changing_the_hospital_information_retires_old_replies(
    client: AsyncClient, cache: ResponseCache
) -> None:
    await seed_hospital()
    patient = await make_user()
    await ask(patient, "What are the OPD timings?", ScriptedLLM(say("9 to 1.")), cache)

    config = await HospitalConfig.find_one()
    assert config is not None
    config.opd_timings = "10 am to 4 pm"
    await config.save()
    after = await ask(patient, "What are the OPD timings?", ScriptedLLM(say("10 to 4.")), cache)

    assert not after.cached and after.reply == "10 to 4."


async def test_rule_based_replies_are_not_cached(
    client: AsyncClient, cache: ResponseCache, redis: Any
) -> None:
    await seed_hospital()
    patient = await make_user()

    await ask(patient, "What are the OPD timings?", ScriptedLLM(), cache)  # no LLM answer

    assert await redis.keys("*") == []


async def test_entries_expire(client: AsyncClient, redis: Any) -> None:
    await seed_hospital()
    patient = await make_user()

    await ask(
        patient, "What are the OPD timings?", ScriptedLLM(say("9 to 1.")), ResponseCache(redis, 600)
    )

    (key,) = await redis.keys("*")
    assert key.startswith(KEY_PREFIX) and 0 < await redis.ttl(key) <= 600


# --- what must never be shared ----------------------------------------------------------


@pytest.mark.guardrail
@pytest.mark.parametrize(
    "question",
    [
        "Show me my last prescription",  # records
        "Can you cancel my appointment?",  # booking
        "I have a fever and cough",  # triage
        "Thank you",  # general
    ],
)
async def test_nothing_about_a_patient_is_ever_cached(
    client: AsyncClient, cache: ResponseCache, redis: Any, question: str
) -> None:
    await seed_hospital()
    patient, doctor = await make_user(), await make_doctor()
    await make_appointment(patient, doctor, slot(-5), AppointmentStatus.DONE, "private note")
    await make_appointment(patient, doctor, slot(3))

    for _ in range(2):
        result = await ask(patient, question, ScriptedLLM(say("A reply.")), cache)
        assert not result.cached

    assert await redis.keys("*") == []


@pytest.mark.guardrail
async def test_one_patients_records_reply_never_reaches_another(
    client: AsyncClient, cache: ResponseCache
) -> None:
    alice, bob, doctor = await make_user(), await make_user(), await make_doctor()
    await make_appointment(alice, doctor, slot(-5), AppointmentStatus.DONE, "alice only")
    question = "Show me my last prescription"
    await ask(alice, question, ScriptedLLM(call("get_my_visits"), say("alice only")), cache, "a")

    reply = await ask(
        bob, question, ScriptedLLM(call("get_my_visits"), say("Nothing on file.")), cache, "b"
    )

    assert reply.reply == "Nothing on file." and not reply.cached


@pytest.mark.guardrail
async def test_a_question_that_leans_on_the_conversation_is_not_cached(
    client: AsyncClient, cache: ResponseCache, redis: Any
) -> None:
    """ "Where is it?" means something different in every conversation."""
    await seed_hospital()
    patient = await make_user()

    result = await ask(patient, "Where is it?", ScriptedLLM(say("First floor, Block A.")), cache)

    assert result.intent == Intent.SUPPORT and not result.cached
    assert await redis.keys("*") == []


@pytest.mark.guardrail
async def test_the_patients_wording_is_not_stored_in_redis(
    client: AsyncClient, cache: ResponseCache, redis: Any
) -> None:
    await seed_hospital()
    await ask(await make_user(), "What are the OPD timings?", ScriptedLLM(say("9 to 1.")), cache)

    (key,) = await redis.keys("*")
    stored = key + await redis.get(key)
    assert "opd" not in stored.lower().replace("9 to 1.", "") and "timings" not in stored.lower()


# --- when Redis is missing or broken ----------------------------------------------------


class BrokenRedis:
    async def get(self, key: str) -> None:
        raise ConnectionError("redis is down")

    async def set(self, key: str, value: str, ex: int) -> None:
        raise ConnectionError("redis is down")


async def test_a_broken_redis_does_not_break_the_turn(client: AsyncClient) -> None:
    await seed_hospital()
    broken = ResponseCache(BrokenRedis(), 3600)

    result = await ask(
        await make_user(), "What are the OPD timings?", ScriptedLLM(say("9 to 1.")), broken
    )

    assert result.reply == "9 to 1." and not result.cached


async def test_with_no_redis_configured_the_cache_is_simply_off(client: AsyncClient) -> None:
    await seed_hospital()
    off = ResponseCache(None, 3600)
    patient = await make_user()

    for _ in range(2):
        result = await ask(patient, "What are the OPD timings?", ScriptedLLM(say("9 to 1.")), off)
        assert result.used_llm and not result.cached


def test_keys_differ_by_question_language_and_facts() -> None:
    base = reply_key("what are the opd timings", "en", "facts-1")

    assert base == reply_key("what are the opd timings", "en", "facts-1")
    assert (
        len(
            {
                base,
                reply_key("where is cardiology", "en", "facts-1"),
                reply_key("what are the opd timings", "hi", "facts-1"),
                reply_key("what are the opd timings", "en", "facts-2"),
            }
        )
        == 4
    )
    assert base.startswith(KEY_PREFIX) and "opd" not in base


async def test_the_chat_api_says_when_a_reply_came_from_the_cache(client: AsyncClient) -> None:
    from tests.helpers import auth

    response = await client.post(
        "/api/chat", json={"message": "hello"}, headers=auth(await make_user())
    )

    assert response.json()["cached"] is False
