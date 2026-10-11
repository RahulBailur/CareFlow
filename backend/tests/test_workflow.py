"""The Redis booking workflow tracker: received -> intent_classified -> slot_checked ->
awaiting_confirmation -> confirmed / failed, per conversation, queryable while it runs."""

import asyncio
from typing import Any

import fakeredis
import pytest
from httpx import AsyncClient

from agents.intent_classifier import IntentClassifier
from agents.orchestrator import run_text_turn
from services.redis_client import WORKFLOW_EVENTS_KEPT, WorkflowTracker
from tests.helpers import auth, make_appointment, make_doctor, make_user, slot
from tests.test_chat import ScriptedLLM, call, say, seed_hospital
from tests.test_live_pipeline import DONE, Rig, audio, heard, said, tool_call
from time_utils import IST

KEYWORDS_ONLY = IntentClassifier()


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> WorkflowTracker:
    tracker = WorkflowTracker(fakeredis.FakeAsyncRedis(decode_responses=True))
    monkeypatch.setattr("tools.get_workflow_tracker", lambda: tracker)
    monkeypatch.setattr("routes.chat.get_workflow_tracker", lambda: tracker)
    return tracker


async def stages(tracker: WorkflowTracker, user: Any, session: str = "s-1") -> list[str]:
    return [event["stage"] for event in await tracker.events(user.id, session)]


async def test_a_confirmed_booking_moves_through_every_stage(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    patient, doctor = await make_user(), await make_doctor("Dr. Flow")
    start = slot(1, 2)
    book = call("book_slot", doctor_id=str(doctor.id), slot_start=start.isoformat())
    find = call("find_slots", date=start.astimezone(IST).date().isoformat())

    await run_text_turn(
        patient,
        "book an appointment",
        "s-1",
        ScriptedLLM(find, book, say("Shall I?")),
        KEYWORDS_ONLY,
    )
    assert await stages(tracker, patient) == [
        "received", "intent_classified", "slot_checked", "awaiting_confirmation",
    ]  # fmt: skip

    await run_text_turn(
        patient, "yes book it", "s-1", ScriptedLLM(book, say("Booked.")), KEYWORDS_ONLY
    )
    assert (await stages(tracker, patient))[-3:] == ["received", "intent_classified", "confirmed"]


async def test_each_event_says_which_turn_and_tool_it_came_from(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    patient = await make_user()
    find = call("find_slots", date=slot(1).astimezone(IST).date().isoformat())

    await run_text_turn(
        patient, "book an appointment", "s-1", ScriptedLLM(find, say("ok")), KEYWORDS_ONLY
    )

    events = await tracker.events(patient.id, "s-1")
    assert len({event["turn"] for event in events}) == 1
    assert events[-1]["tool"] == "find_slots" and "at" in events[-1]
    assert "text" not in str(events) and "book an appointment" not in str(events)


async def test_a_taken_slot_ends_the_workflow_as_failed(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    patient, other, doctor = await make_user(), await make_user(), await make_doctor()
    start = slot(1, 4)
    await make_appointment(other, doctor, start)
    book = call("book_slot", doctor_id=str(doctor.id), slot_start=start.isoformat())

    await run_text_turn(
        patient, "book an appointment", "s-1", ScriptedLLM(book, say("Gone.")), KEYWORDS_ONLY
    )

    assert (await stages(tracker, patient))[-1] == "failed"


async def test_turns_that_are_not_about_booking_leave_no_trace(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    await seed_hospital()
    patient = await make_user()

    for question in ("What are the OPD timings?", "I have a fever", "my prescription", "hello"):
        await run_text_turn(patient, question, "s-1", ScriptedLLM(say("ok")), KEYWORDS_ONLY)

    assert await stages(tracker, patient) == []


async def test_a_booking_turn_without_a_model_still_records_its_start(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    patient = await make_user()

    await run_text_turn(patient, "book an appointment", "s-1", ScriptedLLM(), KEYWORDS_ONLY)

    assert await stages(tracker, patient) == ["received", "intent_classified"]


async def test_conversations_are_tracked_separately(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    patient = await make_user()
    find = call("find_slots", date=slot(1).astimezone(IST).date().isoformat())

    await run_text_turn(
        patient, "book an appointment", "s-1", ScriptedLLM(find, say("ok")), KEYWORDS_ONLY
    )

    assert await stages(tracker, patient, "s-2") == []


async def test_only_the_latest_events_are_kept_and_they_expire(tracker: WorkflowTracker) -> None:
    for index in range(WORKFLOW_EVENTS_KEPT + 10):
        await tracker.advance("u1", "s-1", f"t{index}", "slot_checked")

    events = await tracker.events("u1", "s-1")
    assert (
        len(events) == WORKFLOW_EVENTS_KEPT and events[-1]["turn"] == f"t{WORKFLOW_EVENTS_KEPT + 9}"
    )
    assert 0 < await tracker._client.ttl(tracker.key("u1", "s-1")) <= 24 * 3600


async def test_an_unknown_stage_is_a_programming_error(tracker: WorkflowTracker) -> None:
    with pytest.raises(ValueError, match="unknown workflow stage"):
        await tracker.advance("u1", "s-1", "t1", "teleported")


class BrokenRedis:
    async def rpush(self, *args: Any) -> None:
        raise ConnectionError("down")

    async def lrange(self, *args: Any) -> None:
        raise ConnectionError("down")


async def test_redis_being_down_never_breaks_a_booking(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tools.get_workflow_tracker", lambda: WorkflowTracker(BrokenRedis()))
    patient, doctor = await make_user(), await make_doctor()
    book = call("book_slot", doctor_id=str(doctor.id), slot_start=slot(1, 3).isoformat())

    result = await run_text_turn(
        patient, "book an appointment", "s-1", ScriptedLLM(book, say("Shall I?")), KEYWORDS_ONLY
    )

    assert result.reply == "Shall I?"
    assert await WorkflowTracker(BrokenRedis()).events("u", "s") == []


async def test_with_no_redis_the_tracker_is_simply_off() -> None:
    off = WorkflowTracker(None)

    await off.advance("u1", "s-1", "t1", "received")

    assert not off.enabled and await off.events("u1", "s-1") == []


# --- the API ----------------------------------------------------------------------------


async def test_the_api_shows_the_callers_own_workflow(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    patient = await make_user()
    find = call("find_slots", date=slot(1).astimezone(IST).date().isoformat())
    await run_text_turn(
        patient, "book an appointment", "s-12345678", ScriptedLLM(find, say("ok")), KEYWORDS_ONLY
    )

    response = await client.get("/api/chat/s-12345678/workflow", headers=auth(patient))

    body = response.json()
    assert response.status_code == 200 and body["session_id"] == "s-12345678"
    assert [e["stage"] for e in body["events"]] == ["received", "intent_classified", "slot_checked"]


@pytest.mark.guardrail
async def test_the_api_never_shows_another_users_workflow(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    patient, nosy = await make_user(), await make_user()
    find = call("find_slots", date=slot(1).astimezone(IST).date().isoformat())
    await run_text_turn(
        patient, "book an appointment", "s-12345678", ScriptedLLM(find, say("ok")), KEYWORDS_ONLY
    )

    response = await client.get("/api/chat/s-12345678/workflow", headers=auth(nosy))

    assert response.json()["events"] == []


async def test_the_api_needs_a_login_and_a_well_formed_session(client: AsyncClient) -> None:
    assert (await client.get("/api/chat/s-12345678/workflow")).status_code == 401
    bad = await client.get("/api/chat/x/workflow", headers=auth(await make_user()))
    assert bad.status_code == 422


# --- Pipeline A -------------------------------------------------------------------------


async def test_pipeline_a_bookings_are_tracked_too(
    client: AsyncClient, tracker: WorkflowTracker
) -> None:
    patient, doctor = await make_user(), await make_doctor()
    args = {"doctor_id": str(doctor.id), "slot_start": slot(1, 2).isoformat()}
    rig = await Rig(patient).begin()

    await rig.turn(
        heard("book the first one"),
        tool_call("book_slot", **args),
        DONE,
        audio(),
        said("Shall I?"),
        DONE,
    )
    await asyncio.sleep(0.05)

    assert await stages(tracker, patient, "live-session-1") == [
        "received", "intent_classified", "awaiting_confirmation",
    ]  # fmt: skip
    await rig.session.close()
