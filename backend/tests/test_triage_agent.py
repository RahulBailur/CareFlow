"""Guardrail: triage routes to a department and nothing more, end to end."""

import pytest
from httpx import AsyncClient

from agents.guardrails import DISCLAIMERS, SAFE_FALLBACKS, is_safe
from agents.intent_classifier import IntentClassifier
from agents.intent_data import Intent
from agents.orchestrator import run_text_turn
from models.conversation_turn import ConversationTurn
from services.llm import MockLLM
from tests.helpers import auth, make_user
from tests.test_chat import ScriptedLLM, call, say, seed_hospital
from tools.triage_tools import DEFAULT_DEPARTMENT, suggest_department

pytestmark = pytest.mark.guardrail

KEYWORDS_ONLY = IntentClassifier()


@pytest.mark.parametrize(
    ("symptoms", "department"),
    [
        ("I have chest pain since last night", "Cardiology"),
        ("My knee hurts when I climb stairs", "Orthopedics"),
        ("My son has a rash and high fever", "Pediatrics"),
        ("itchy skin and red rash", "Dermatology"),
        ("ear pain and I can't hear properly", "ENT"),
        ("fever and cough for two days", DEFAULT_DEPARTMENT),
        ("मेरे सीने में दर्द हो रहा है", "Cardiology"),
        ("ನನ್ನ ಮೊಣಕಾಲು ತುಂಬಾ ನೋಯುತ್ತಿದೆ", "Orthopedics"),
        ("bacche ko khansi aur bukhar hai", "Pediatrics"),
        ("kaan mein dard ho raha hai", "ENT"),
        ("something feels off", DEFAULT_DEPARTMENT),
    ],
)
def test_symptoms_map_to_a_department(symptoms: str, department: str) -> None:
    assert suggest_department(symptoms)[0] == department


@pytest.mark.parametrize(
    "symptoms",
    ["I have chest pain", "I can't breathe properly", "he fainted", "सांस लेने में तकलीफ है"],
)
def test_red_flag_symptoms_are_marked_urgent(symptoms: str) -> None:
    assert suggest_department(symptoms)[1] is True


def test_ordinary_symptoms_are_not_marked_urgent() -> None:
    assert suggest_department("mild cough since yesterday")[1] is False


def test_short_keywords_do_not_match_inside_other_words() -> None:
    # "ear" inside "year", "son" inside "person"
    assert suggest_department("for a year this person has had a cough")[0] == DEFAULT_DEPARTMENT


async def test_a_diagnosing_model_reply_is_replaced_before_it_reaches_the_patient(
    client: AsyncClient,
) -> None:
    patient = await make_user()
    llm = ScriptedLLM(
        call("symptom_to_department", symptoms="chest pain"),
        say("You probably have angina. Take 75 mg of aspirin twice a day."),
    )

    result = await run_text_turn(patient, "I have chest pain", "session-1", llm, KEYWORDS_ONLY)

    assert result.intent == Intent.TRIAGE
    assert result.blocked
    assert result.reply == SAFE_FALLBACKS["en"]
    assert "angina" not in result.reply and "aspirin" not in result.reply


async def test_the_blocked_text_is_not_kept_in_conversation_memory(client: AsyncClient) -> None:
    patient = await make_user()
    llm = ScriptedLLM(say("You have pneumonia. Take amoxicillin."))

    await run_text_turn(patient, "I have a bad cough", "session-1", llm, KEYWORDS_ONLY)

    stored = " ".join([turn.text async for turn in ConversationTurn.find_all()])
    assert "pneumonia" not in stored and "amoxicillin" not in stored


async def test_a_safe_model_reply_passes_with_the_disclaimer(client: AsyncClient) -> None:
    patient = await make_user()
    reply = "For knee pain, Orthopedics on the ground floor is the right place to start."
    llm = ScriptedLLM(call("symptom_to_department", symptoms="knee pain"), say(reply))

    result = await run_text_turn(patient, "My knee hurts", "session-1", llm, KEYWORDS_ONLY)

    assert result.reply == reply
    assert not result.blocked
    assert result.disclaimer == DISCLAIMERS["en"]


@pytest.mark.parametrize(
    ("question", "language"),
    [("मुझे दो दिन से बुखार है", "hi"), ("ನನಗೆ ಜ್ವರ ಇದೆ", "kn"), ("I have a fever", "en")],
)
async def test_the_disclaimer_follows_the_language_of_an_llm_reply(
    client: AsyncClient, question: str, language: str
) -> None:
    patient = await make_user()
    llm = ScriptedLLM(say("General Medicine."))

    result = await run_text_turn(patient, question, "session-1", llm, KEYWORDS_ONLY)

    assert result.disclaimer == DISCLAIMERS[language]  # type: ignore[index]


async def test_the_rule_based_triage_reply_names_a_department_and_is_safe(
    client: AsyncClient,
) -> None:
    await seed_hospital()
    patient = await make_user()

    result = await run_text_turn(
        patient, "I have chest pain since last night", "session-1", MockLLM(), KEYWORDS_ONLY
    )

    assert "Cardiology" in result.reply and "First floor" in result.reply
    assert "1800-000-0000" in result.reply  # chest pain is urgent: emergency number given
    assert is_safe(result.reply) and not result.blocked
    assert result.disclaimer == DISCLAIMERS["en"]


async def test_every_triage_reply_from_the_api_carries_a_disclaimer(client: AsyncClient) -> None:
    patient = await make_user()

    response = await client.post(
        "/api/chat", json={"message": "I have a fever and cough"}, headers=auth(patient)
    )

    body = response.json()
    assert body["intent"] == "triage"
    assert body["disclaimer"] == DISCLAIMERS["en"]


async def test_non_triage_replies_carry_no_disclaimer(client: AsyncClient) -> None:
    patient = await make_user()

    response = await client.post("/api/chat", json={"message": "hello"}, headers=auth(patient))

    assert response.json()["disclaimer"] is None


async def test_the_triage_agent_has_no_booking_or_records_tools(client: AsyncClient) -> None:
    """Even if the model asks for one, it is refused and nothing is booked."""
    patient = await make_user()
    llm = ScriptedLLM(call("book_slot", doctor_id="x", slot_start="y"), say("Cardiology."))

    result = await run_text_turn(patient, "I have chest pain", "session-1", llm, KEYWORDS_ONLY)

    assert llm.calls[1][1][-1].tool_result == {"error": "book_slot is not available here."}
    assert result.tools_called == ["book_slot"]
