"""Guardrail: who the patient is comes from the session, never from the model.

A model can be talked into sending anything. These tests send tool arguments naming a
different patient and check that the tools act for the logged-in user regardless.
"""

from typing import Any

import pytest
from httpx import AsyncClient

from agents.booking_agent import BOOKING_AGENT
from agents.orchestrator import AGENTS
from agents.records_agent import RECORDS_AGENT
from models.appointment import Appointment, AppointmentStatus
from models.user import Role
from tests.helpers import confirmed_tool_call, make_appointment, make_doctor, make_user, slot
from tools import ToolContext, run_tool

pytestmark = pytest.mark.guardrail

IDENTITY_WORDS = ("patient", "user_id", "userid", "email", "phone")


def tool(agent_tools: tuple[Any, ...], name: str) -> Any:
    return next(t for t in agent_tools if t.name == name)


def test_no_tool_declares_a_parameter_that_identifies_a_person() -> None:
    declared = [
        (t.name, parameter)
        for agent in AGENTS.values()
        for t in agent.tools
        for parameter in t.parameters["properties"]
    ]

    assert declared, "expected the agents to have tools"
    offending = [d for d in declared if any(word in d[1].lower() for word in IDENTITY_WORDS)]
    assert offending == []


def test_every_tool_schema_refuses_undeclared_arguments() -> None:
    for agent in AGENTS.values():
        for t in agent.tools:
            assert t.parameters["additionalProperties"] is False, t.name


async def test_booking_ignores_a_patient_id_supplied_by_the_model(client: AsyncClient) -> None:
    caller, victim, doctor = await make_user(), await make_user(), await make_doctor()
    start = slot(1, 3)

    result = await confirmed_tool_call(
        tool(BOOKING_AGENT.tools, "book_slot"),
        caller,
        {
            "doctor_id": str(doctor.id),
            "slot_start": start.isoformat(),
            "patient_id": str(victim.id),
            "user_id": str(victim.id),
        },
    )

    assert "booked" in result
    appointments = await Appointment.find_all().to_list()
    assert [a.patient_id for a in appointments] == [caller.id]


async def test_the_model_cannot_cancel_another_patients_appointment(client: AsyncClient) -> None:
    caller, victim, doctor = await make_user(), await make_user(), await make_doctor()
    theirs = await make_appointment(victim, doctor, slot(2))

    result = await run_tool(
        tool(BOOKING_AGENT.tools, "cancel_appointment"),
        ToolContext(user=caller),
        {"appointment_id": str(theirs.id), "patient_id": str(victim.id)},
    )

    assert result == {"error": "Appointment not found"}
    unchanged = await Appointment.get(theirs.id)
    assert unchanged is not None and unchanged.status == AppointmentStatus.BOOKED


async def test_the_model_cannot_reschedule_another_patients_appointment(
    client: AsyncClient,
) -> None:
    caller, victim, doctor = await make_user(), await make_user(), await make_doctor()
    original = slot(2)
    theirs = await make_appointment(victim, doctor, original)

    result = await run_tool(
        tool(BOOKING_AGENT.tools, "reschedule_appointment"),
        ToolContext(user=caller),
        {"appointment_id": str(theirs.id), "slot_start": slot(3).isoformat()},
    )

    assert result == {"error": "Appointment not found"}
    unchanged = await Appointment.get(theirs.id)
    assert unchanged is not None and unchanged.slot_key == theirs.slot_key


async def test_listing_appointments_shows_only_the_callers_own(client: AsyncClient) -> None:
    caller, victim = await make_user(), await make_user()
    doctor = await make_doctor("Dr. Listed")
    mine = await make_appointment(caller, doctor, slot(2, 0))
    await make_appointment(victim, doctor, slot(2, 1))

    result = await run_tool(
        tool(BOOKING_AGENT.tools, "list_my_appointments"),
        ToolContext(user=caller),
        {"patient_id": str(victim.id)},
    )

    assert [a["appointment_id"] for a in result["upcoming"]] == [str(mine.id)]


@pytest.mark.parametrize("role", [Role.DOCTOR, Role.ADMIN])
@pytest.mark.parametrize(
    ("agent_tools", "name"),
    [
        (BOOKING_AGENT.tools, "book_slot"),
        (BOOKING_AGENT.tools, "list_my_appointments"),
        (BOOKING_AGENT.tools, "cancel_appointment"),
        (BOOKING_AGENT.tools, "reschedule_appointment"),
        (RECORDS_AGENT.tools, "get_my_visits"),
    ],
)
async def test_patient_tools_refuse_staff_accounts(
    client: AsyncClient, role: Role, agent_tools: tuple[Any, ...], name: str
) -> None:
    staff = await make_user(role)

    result = await run_tool(tool(agent_tools, name), ToolContext(user=staff), {})

    assert result == {"error": "Only a logged-in patient can do this."}
    assert await Appointment.count() == 0
