"""Guardrail: CareBot cannot book, cancel or move an appointment without a turn to confirm.

Found on the first real conversation with Gemini: told "the first one please" it booked at
once, then booked a second slot when the patient said "yes, confirm it". The rule is now in
code: a changing tool only takes effect in the turn after the one that proposed it.
"""

from typing import Any

import pytest
from httpx import AsyncClient

from agents.booking_agent import BOOKING_AGENT
from agents.intent_classifier import IntentClassifier
from agents.orchestrator import run_text_turn
from models.appointment import Appointment, AppointmentStatus
from models.pending_action import PendingAction
from models.user import User
from services.llm import LLMUnavailable
from tests.helpers import make_appointment, make_doctor, make_user, slot
from tests.test_chat import ScriptedLLM, call, say
from tools import ToolContext, run_tool

pytestmark = pytest.mark.guardrail

KEYWORDS_ONLY = IntentClassifier()
TOOLS = {tool.name: tool for tool in BOOKING_AGENT.tools}


def turn(user: User, turn_id: str, session_id: str = "session-1") -> ToolContext:
    return ToolContext(user=user, session_id=session_id, turn_id=turn_id)


async def booking_args() -> tuple[User, dict[str, Any]]:
    patient, doctor = await make_user(), await make_doctor("Dr. Confirm")
    return patient, {"doctor_id": str(doctor.id), "slot_start": slot(1, 2).isoformat()}


async def test_the_first_call_proposes_and_changes_nothing(client: AsyncClient) -> None:
    patient, args = await booking_args()

    result = await run_tool(TOOLS["book_slot"], turn(patient, "t1"), args)

    assert result["needs_confirmation"] is True
    assert "Dr. Confirm" in result["not_done_yet"]
    assert await Appointment.count() == 0


async def test_repeating_the_call_in_the_same_turn_still_changes_nothing(
    client: AsyncClient,
) -> None:
    patient, args = await booking_args()

    for _ in range(3):
        result = await run_tool(TOOLS["book_slot"], turn(patient, "t1"), args)

    assert result["needs_confirmation"] is True
    assert await Appointment.count() == 0


async def test_the_same_call_in_the_next_turn_goes_through_once(client: AsyncClient) -> None:
    patient, args = await booking_args()
    await run_tool(TOOLS["book_slot"], turn(patient, "t1"), args)

    result = await run_tool(TOOLS["book_slot"], turn(patient, "t2"), args)
    again = await run_tool(TOOLS["book_slot"], turn(patient, "t2"), args)

    assert "booked" in result
    assert await Appointment.count() == 1
    assert "booked" not in again  # the confirmation was used up


async def test_different_arguments_in_the_next_turn_are_a_new_proposal(
    client: AsyncClient,
) -> None:
    patient, args = await booking_args()
    await run_tool(TOOLS["book_slot"], turn(patient, "t1"), args)

    other = {**args, "slot_start": slot(1, 5).isoformat()}
    result = await run_tool(TOOLS["book_slot"], turn(patient, "t2"), other)

    assert result["needs_confirmation"] is True
    assert await Appointment.count() == 0


async def test_a_confirmation_does_not_carry_to_another_session_or_user(
    client: AsyncClient,
) -> None:
    patient, args = await booking_args()
    someone_else = await make_user()
    await run_tool(TOOLS["book_slot"], turn(patient, "t1"), args)

    in_other_session = await run_tool(TOOLS["book_slot"], turn(patient, "t2", "session-2"), args)
    as_other_user = await run_tool(TOOLS["book_slot"], turn(someone_else, "t2"), args)

    assert in_other_session["needs_confirmation"] and as_other_user["needs_confirmation"]
    assert await Appointment.count() == 0


async def test_cancelling_needs_a_confirming_turn(client: AsyncClient) -> None:
    patient, doctor = await make_user(), await make_doctor("Dr. Cancel")
    appointment = await make_appointment(patient, doctor, slot(2))
    args = {"appointment_id": str(appointment.id)}

    proposal = await run_tool(TOOLS["cancel_appointment"], turn(patient, "t1"), args)
    unchanged = await Appointment.get(appointment.id)
    assert proposal["needs_confirmation"] and "Dr. Cancel" in proposal["not_done_yet"]
    assert unchanged is not None and unchanged.status == AppointmentStatus.BOOKED

    result = await run_tool(TOOLS["cancel_appointment"], turn(patient, "t2"), args)
    cancelled = await Appointment.get(appointment.id)
    assert "cancelled" in result
    assert cancelled is not None and cancelled.status == AppointmentStatus.CANCELLED


async def test_rescheduling_needs_a_confirming_turn(client: AsyncClient) -> None:
    patient, doctor = await make_user(), await make_doctor()
    appointment = await make_appointment(patient, doctor, slot(2))
    args = {"appointment_id": str(appointment.id), "slot_start": slot(3, 1).isoformat()}

    proposal = await run_tool(TOOLS["reschedule_appointment"], turn(patient, "t1"), args)
    unchanged = await Appointment.get(appointment.id)
    assert proposal["needs_confirmation"]
    assert unchanged is not None and unchanged.slot_key == appointment.slot_key

    result = await run_tool(TOOLS["reschedule_appointment"], turn(patient, "t2"), args)
    assert "rescheduled" in result


async def test_a_taken_or_invalid_slot_is_refused_before_asking_to_confirm(
    client: AsyncClient,
) -> None:
    patient, args = await booking_args()
    other = await make_user()
    await Appointment(
        patient_id=other.id,
        doctor_id=args["doctor_id"],
        department="Cardiology",
        slot_start=slot(1, 2),
        slot_end=slot(1, 3),
    ).insert()

    taken = await run_tool(TOOLS["book_slot"], turn(patient, "t1"), args)
    off_schedule = await run_tool(
        TOOLS["book_slot"], turn(patient, "t1"), {**args, "slot_start": slot(-1).isoformat()}
    )

    assert taken == {"error": "That slot has just been taken"}
    assert "error" in off_schedule
    assert await PendingAction.count() == 0


async def test_a_proposal_expires_if_the_next_turn_does_not_confirm_it(
    client: AsyncClient,
) -> None:
    patient, args = await booking_args()
    book = call("book_slot", **args)
    await run_text_turn(patient, "book an appointment", "s-1", ScriptedLLM(book, say("OK?")))
    await run_text_turn(patient, "what are the OPD timings?", "s-1", ScriptedLLM(say("9 to 1.")))

    await run_text_turn(patient, "yes book it", "s-1", ScriptedLLM(book, say("OK?")), KEYWORDS_ONLY)

    assert await Appointment.count() == 0


async def test_a_model_that_books_at_once_and_then_again_books_exactly_once(
    client: AsyncClient,
) -> None:
    """The conversation that went wrong with the real model, replayed."""
    patient, doctor = await make_user(), await make_doctor("Dr. Once")
    first, second = slot(1, 0), slot(1, 1)
    book_first = call("book_slot", doctor_id=str(doctor.id), slot_start=first.isoformat())
    book_second = call("book_slot", doctor_id=str(doctor.id), slot_start=second.isoformat())

    # "The first one please": the model tries to book straight away, twice
    eager = ScriptedLLM(book_first, book_first, say("I am booking 9:00 AM."))
    await run_text_turn(patient, "book the first one please", "s-1", eager, KEYWORDS_ONLY)
    assert await Appointment.count() == 0

    # "Yes, confirm it": it books the proposed slot, then tries another one as well
    greedy = ScriptedLLM(book_first, book_second, say("Booked."))
    await run_text_turn(patient, "yes, book it", "s-1", greedy, KEYWORDS_ONLY)

    booked = await Appointment.find_all().to_list()
    assert len(booked) == 1
    assert (
        booked[0].slot_key
        == Appointment(
            patient_id=patient.id,
            doctor_id=doctor.id,
            department="",
            slot_start=first,
            slot_end=second,
        ).slot_key
    )


async def test_what_was_done_is_remembered_for_the_next_turn(client: AsyncClient) -> None:
    patient, args = await booking_args()
    book = call("book_slot", **args)
    await run_text_turn(patient, "book an appointment", "s-1", ScriptedLLM(book, say("Shall I?")))
    await run_text_turn(patient, "yes book it", "s-1", ScriptedLLM(book, say("Booked.")))

    llm = ScriptedLLM(say("You're welcome."))
    await run_text_turn(patient, "book it, thanks", "s-1", llm, KEYWORDS_ONLY)

    history = [message.text for message in llm.calls[0][1]]
    assert any("awaiting the patient's yes for book_slot" in text for text in history)
    assert any("[booked Dr. Confirm at" in text for text in history)


async def test_the_note_for_the_model_is_not_part_of_the_reply(client: AsyncClient) -> None:
    patient, args = await booking_args()
    llm = ScriptedLLM(call("book_slot", **args), say("Shall I book it?"))

    result = await run_text_turn(patient, "book an appointment", "s-1", llm, KEYWORDS_ONLY)

    assert result.reply == "Shall I book it?"


async def test_a_proposal_the_patient_never_heard_cannot_be_confirmed(client: AsyncClient) -> None:
    """The model proposes, then fails before it can ask. A later "yes" must not book."""
    patient, args = await booking_args()
    book = call("book_slot", **args)
    failed = ScriptedLLM(book, LLMUnavailable("quota"))
    await run_text_turn(patient, "book an appointment", "s-1", failed, KEYWORDS_ONLY)
    assert await PendingAction.count() == 0

    await run_text_turn(patient, "yes book it", "s-1", ScriptedLLM(book, say("Shall I?")))

    assert await Appointment.count() == 0


async def test_a_change_made_before_the_model_failed_is_still_reported(
    client: AsyncClient,
) -> None:
    patient, doctor = await make_user(), await make_doctor("Dr. Reported")
    appointment = await make_appointment(patient, doctor, slot(2))
    cancel = call("cancel_appointment", appointment_id=str(appointment.id))
    await run_text_turn(patient, "cancel my appointment", "s-1", ScriptedLLM(cancel, say("Sure?")))

    result = await run_text_turn(
        patient, "yes cancel it", "s-1", ScriptedLLM(cancel, LLMUnavailable("quota")), KEYWORDS_ONLY
    )

    assert result.reply.startswith("Done: cancelled Dr. Reported at")
    assert not result.used_llm
