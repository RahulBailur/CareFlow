"""CareBot text turns: routing, the tool loop, rule-based fallbacks and conversation memory."""

from datetime import timedelta
from typing import Any

import pytest
from httpx import AsyncClient

from agents.base import MAX_TOOL_ROUNDS
from agents.intent_classifier import IntentClassifier
from agents.intent_data import Intent
from agents.orchestrator import run_text_turn
from models.appointment import Appointment, AppointmentStatus
from models.conversation_turn import ConversationTurn
from models.hospital_config import Department, HospitalConfig
from services.llm import LLMResponse, LLMUnavailable, Message, MockLLM, ToolCall, ToolSpec
from tests.helpers import auth, make_appointment, make_doctor, make_user, slot
from time_utils import IST, as_utc

KEYWORDS_ONLY = IntentClassifier()


class ScriptedLLM:
    """Plays back prepared responses and records what it was sent."""

    name = "scripted"

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, list[Message], list[ToolSpec]]] = []

    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        self.calls.append((system, list(messages), list(tools or [])))
        if not self._responses:
            raise LLMUnavailable("script exhausted")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def say(text: str) -> LLMResponse:
    return LLMResponse(text=text)


def call(name: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name, args)])


async def seed_hospital() -> None:
    await HospitalConfig(
        name="Test Hospital",
        address="1 Test Road",
        opd_timings="9 am to 1 pm, Monday to Saturday",
        emergency_contact="1800-000-0000",
        departments=[
            Department(name="Cardiology", location="First floor", opd_timings="9 am to 1 pm"),
            Department(name="ENT", location="Second floor", opd_timings="9 am to 1 pm"),
        ],
    ).insert()


# --- the endpoint -----------------------------------------------------------------------


async def test_chat_requires_login(client: AsyncClient) -> None:
    response = await client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 401


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "x" * 501}, {}])
async def test_chat_rejects_empty_and_oversized_messages(
    client: AsyncClient, body: dict[str, str]
) -> None:
    patient = await make_user()

    response = await client.post("/api/chat", json=body, headers=auth(patient))

    assert response.status_code == 422


async def test_chat_rejects_a_malformed_session_id(client: AsyncClient) -> None:
    patient = await make_user()

    response = await client.post(
        "/api/chat", json={"message": "hello", "session_id": "../x"}, headers=auth(patient)
    )

    assert response.status_code == 422


async def test_chat_without_an_llm_answers_from_rules(client: AsyncClient) -> None:
    await seed_hospital()
    patient = await make_user()

    response = await client.post(
        "/api/chat", json={"message": "What are the OPD timings?"}, headers=auth(patient)
    )

    body = response.json()
    assert response.status_code == 200
    assert body["intent"] == "support"
    assert "9 am to 1 pm" in body["reply"]
    assert body["used_llm"] is False
    assert body["routed_by"] == "keywords"
    assert body["disclaimer"] is None
    assert len(body["session_id"]) >= 8


async def test_chat_keeps_the_session_id_it_is_given(client: AsyncClient) -> None:
    patient = await make_user()

    first = await client.post("/api/chat", json={"message": "hello"}, headers=auth(patient))
    session_id = first.json()["session_id"]
    second = await client.post(
        "/api/chat", json={"message": "thanks", "session_id": session_id}, headers=auth(patient)
    )

    assert second.json()["session_id"] == session_id
    assert await ConversationTurn.find(ConversationTurn.session_id == session_id).count() == 4


# --- rule-based fallbacks ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the emergency contact number?", "1800-000-0000"),
        ("Where is the cardiology department?", "First floor"),
        ("Where is the ENT department?", "Second floor"),
        ("What is the hospital address?", "1 Test Road"),
        ("Is the hospital open on Sunday?", "Monday to Saturday"),
    ],
)
async def test_support_fallback_answers_from_the_hospital_config(
    client: AsyncClient, question: str, expected: str
) -> None:
    await seed_hospital()
    patient = await make_user()

    result = await run_text_turn(patient, question, "session-1", MockLLM(), KEYWORDS_ONLY)

    assert result.intent == Intent.SUPPORT
    assert expected in result.reply


async def test_department_names_match_whole_words_only(client: AsyncClient) -> None:
    """ "ENT" must not be found inside "department"."""
    await seed_hospital()
    patient = await make_user()

    result = await run_text_turn(
        patient, "Where is the cardiology department?", "session-1", MockLLM(), KEYWORDS_ONLY
    )

    assert "Cardiology" in result.reply and "ENT" not in result.reply


async def test_records_fallback_reads_back_the_last_visit(client: AsyncClient) -> None:
    patient, doctor = await make_user(), await make_doctor("Dr. Records")
    await make_appointment(
        patient, doctor, slot(-10), AppointmentStatus.DONE, prescription="Synthetic note 7"
    )

    result = await run_text_turn(
        patient, "Show me my last prescription", "session-1", MockLLM(), KEYWORDS_ONLY
    )

    assert result.intent == Intent.RECORDS
    assert "Dr. Records" in result.reply and "Synthetic note 7" in result.reply


async def test_booking_fallback_lists_upcoming_appointments(client: AsyncClient) -> None:
    patient, doctor = await make_user(), await make_doctor("Dr. Upcoming")
    await make_appointment(patient, doctor, slot(2))

    result = await run_text_turn(
        patient, "Can you cancel my appointment?", "session-1", MockLLM(), KEYWORDS_ONLY
    )

    assert result.intent == Intent.BOOKING
    assert "Dr. Upcoming" in result.reply
    assert await Appointment.find(Appointment.status == AppointmentStatus.BOOKED).count() == 1


async def test_general_fallback_says_what_carebot_can_do(client: AsyncClient) -> None:
    patient = await make_user()

    result = await run_text_turn(patient, "hello", "session-1", MockLLM(), KEYWORDS_ONLY)

    assert result.intent == Intent.GENERAL
    assert "appointment" in result.reply


# --- the tool loop ----------------------------------------------------------------------


async def test_booking_agent_books_through_its_tools(client: AsyncClient) -> None:
    patient, doctor = await make_user(), await make_doctor("Dr. Tool")
    start = slot(1, 4)
    llm = ScriptedLLM(
        call("find_slots", date=start.astimezone(IST).date().isoformat()),
        call("book_slot", doctor_id=str(doctor.id), slot_start=start.isoformat()),
        say("Booked with Dr. Tool."),
    )

    result = await run_text_turn(
        patient, "Book an appointment for tomorrow", "session-1", llm, KEYWORDS_ONLY
    )

    booked = await Appointment.find_one(Appointment.patient_id == patient.id)
    assert booked is not None and as_utc(booked.slot_start) == start
    assert result.reply == "Booked with Dr. Tool."
    assert result.tools_called == ["find_slots", "book_slot"]
    assert result.used_llm
    # The model was shown the real slots before it booked one
    slots_message = llm.calls[1][1][-1]
    assert slots_message.role == "tool" and slots_message.tool_result is not None
    assert slots_message.tool_result["doctors"][0]["doctor_name"] == "Dr. Tool"


async def test_each_agent_only_offers_its_own_tools(client: AsyncClient) -> None:
    patient = await make_user()
    offered: dict[str, set[str]] = {}
    for question in ("book an appointment", "I have a fever", "my prescription", "OPD timings"):
        llm = ScriptedLLM(say("ok"))
        result = await run_text_turn(patient, question, "session-1", llm, KEYWORDS_ONLY)
        offered[result.intent.value] = {tool.name for tool in llm.calls[0][2]}

    assert offered["triage"] == {"symptom_to_department"}
    assert offered["records"] == {"get_my_visits"}
    assert offered["support"] == {"get_hospital_info"}
    assert "book_slot" in offered["booking"] and "get_my_visits" not in offered["booking"]


async def test_a_tool_error_is_returned_to_the_model_not_raised(client: AsyncClient) -> None:
    patient = await make_user()
    llm = ScriptedLLM(call("find_slots", date="not-a-date"), say("Which day would you like?"))

    result = await run_text_turn(patient, "book an appointment", "session-1", llm, KEYWORDS_ONLY)

    tool_message = llm.calls[1][1][-1]
    assert tool_message.tool_result is not None and "error" in tool_message.tool_result
    assert result.reply == "Which day would you like?"


async def test_a_booking_refusal_reaches_the_model_as_an_error(client: AsyncClient) -> None:
    patient, other, doctor = await make_user(), await make_user(), await make_doctor()
    start = slot(1, 2)
    await make_appointment(other, doctor, start)
    llm = ScriptedLLM(
        call("book_slot", doctor_id=str(doctor.id), slot_start=start.isoformat()),
        say("That slot has gone."),
    )

    await run_text_turn(patient, "book an appointment", "session-1", llm, KEYWORDS_ONLY)

    assert llm.calls[1][1][-1].tool_result == {"error": "That slot has just been taken"}
    assert await Appointment.find(Appointment.patient_id == patient.id).count() == 0


async def test_a_model_that_never_stops_calling_tools_is_cut_off(client: AsyncClient) -> None:
    patient = await make_user()
    llm = ScriptedLLM(*[call("list_my_appointments") for _ in range(MAX_TOOL_ROUNDS + 3)])

    result = await run_text_turn(patient, "book an appointment", "session-1", llm, KEYWORDS_ONLY)

    assert len(llm.calls) == MAX_TOOL_ROUNDS
    assert not result.used_llm
    assert "upcoming" in result.reply


async def test_an_llm_failure_mid_turn_falls_back_to_rules(client: AsyncClient) -> None:
    await seed_hospital()
    patient = await make_user()
    llm = ScriptedLLM(LLMUnavailable("quota"))

    result = await run_text_turn(patient, "OPD timings?", "session-1", llm, KEYWORDS_ONLY)

    assert not result.used_llm
    assert "9 am to 1 pm" in result.reply


async def test_the_agent_is_told_the_reply_language(client: AsyncClient) -> None:
    patient = await make_user()
    llm = ScriptedLLM(say("ठीक है"))

    result = await run_text_turn(patient, "ओपीडी का समय क्या है?", "session-1", llm, KEYWORDS_ONLY)

    assert result.language == "hi"
    assert "Reply in Hindi" in llm.calls[0][0]


# --- routing ----------------------------------------------------------------------------


async def test_an_unclear_turn_is_routed_by_the_llm(client: AsyncClient) -> None:
    patient = await make_user()
    llm = ScriptedLLM(say("records"), say("Here is your history."))

    result = await run_text_turn(
        patient, "what did they give me before", "session-1", llm, KEYWORDS_ONLY
    )

    assert result.intent == Intent.RECORDS
    assert result.routed_by == "llm"
    assert llm.calls[0][2] == []  # the routing call carries no tools


async def test_a_clear_turn_never_asks_the_llm_to_route(client: AsyncClient) -> None:
    patient = await make_user()
    llm = ScriptedLLM(say("Sure."))

    result = await run_text_turn(patient, "book an appointment", "session-1", llm, KEYWORDS_ONLY)

    assert result.routed_by == "keywords"
    assert len(llm.calls) == 1


async def test_nonsense_from_the_routing_llm_is_ignored(client: AsyncClient) -> None:
    patient = await make_user()
    llm = ScriptedLLM(say("I think this is probably about billing"), say("Hello!"))

    result = await run_text_turn(patient, "xyzzy", "session-1", llm, KEYWORDS_ONLY)

    assert result.intent == Intent.GENERAL
    assert result.routed_by == "guess"


async def test_a_follow_up_without_an_llm_stays_with_the_previous_agent(
    client: AsyncClient,
) -> None:
    patient, doctor = await make_user(), await make_doctor("Dr. Follow")
    await make_appointment(patient, doctor, slot(3))
    await run_text_turn(patient, "cancel my appointment", "session-1", MockLLM(), KEYWORDS_ONLY)

    result = await run_text_turn(patient, "and then?", "session-1", MockLLM(), KEYWORDS_ONLY)

    assert result.intent == Intent.BOOKING
    assert result.routed_by == "previous"


# --- memory -----------------------------------------------------------------------------


async def test_earlier_turns_of_the_session_are_sent_to_the_model(client: AsyncClient) -> None:
    patient = await make_user()
    await run_text_turn(patient, "OPD timings?", "session-1", ScriptedLLM(say("9 to 1.")))

    llm = ScriptedLLM(say("As I said, 9 to 1."))
    await run_text_turn(patient, "what are the timings again", "session-1", llm, KEYWORDS_ONLY)

    sent = [(m.role, m.text) for m in llm.calls[0][1]]
    assert sent == [
        ("user", "OPD timings?"),
        ("assistant", "9 to 1."),
        ("user", "what are the timings again"),
    ]


@pytest.mark.guardrail
async def test_memory_is_never_shared_between_users_or_sessions(client: AsyncClient) -> None:
    alice, bob = await make_user(), await make_user()
    await run_text_turn(alice, "my prescription", "shared-id", ScriptedLLM(say("Alice's note")))

    for user, session in ((bob, "shared-id"), (alice, "another-id")):
        llm = ScriptedLLM(say("ok"))
        await run_text_turn(user, "my prescription", session, llm, KEYWORDS_ONLY)
        assert [m.text for m in llm.calls[0][1]] == ["my prescription"]


async def test_only_the_latest_turns_are_replayed(client: AsyncClient) -> None:
    patient = await make_user()
    for index in range(6):
        await run_text_turn(patient, f"OPD timings {index}", "s-1", ScriptedLLM(say(f"r{index}")))

    llm = ScriptedLLM(say("ok"))
    await run_text_turn(patient, "OPD timings now", "s-1", llm, KEYWORDS_ONLY)

    history = llm.calls[0][1][:-1]
    assert len(history) == 8
    assert history[-1].text == "r5" and history[0].text == "OPD timings 2"


async def test_conversation_turns_expire_after_a_day() -> None:
    ttl = [
        index.document.get("expireAfterSeconds")
        for index in ConversationTurn.Settings.indexes
        if "expireAfterSeconds" in index.document
    ]

    assert ttl == [int(timedelta(days=1).total_seconds())]
