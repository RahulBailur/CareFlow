from datetime import date, datetime
from typing import Any

from beanie import PydanticObjectId
from pydantic import AwareDatetime, Field

from models.appointment import Appointment, AppointmentStatus
from models.user import Role, User
from services import appointments as booking
from services.scheduling import bookable_slot, open_slots
from time_utils import IST, as_utc, now_utc
from tools import PATIENTS_ONLY, Tool, ToolArgs, ToolContext, ToolResult, schema
from tools.confirmation import confirm_or_propose

MAX_SLOTS_PER_DOCTOR = 6


def local_time(moment: datetime) -> str:
    """How a time is said to a patient: hospital-local, e.g. 'Tue 13 Oct, 10:15 AM'."""
    local = as_utc(moment).astimezone(IST)
    return f"{local:%a} {local.day} {local:%b}, {local:%I:%M %p}".replace(", 0", ", ")


def _describe(appointment: Appointment, doctor_name: str) -> dict[str, Any]:
    return {
        "appointment_id": str(appointment.id),
        "doctor_name": doctor_name,
        "department": appointment.department,
        "slot_start": as_utc(appointment.slot_start).isoformat(),
        "local_time": local_time(appointment.slot_start),
        "status": appointment.status.value,
    }


async def _match_department(name: str) -> str | None:
    doctors = await User.find(User.role == Role.DOCTOR).to_list()
    wanted = name.strip().lower()
    return next((d.department for d in doctors if (d.department or "").lower() == wanted), None)


async def _require_open(doctor_id: PydanticObjectId, slot_start: datetime) -> None:
    """Refuse before asking the patient to confirm something that cannot be done."""
    slot = await bookable_slot(doctor_id, slot_start)
    if slot is None:
        raise booking.slot_unavailable()
    day = slot[0].astimezone(IST).date()
    if slot not in await open_slots(doctor_id, day):
        raise booking.BookingError(409, "That slot has just been taken")


class FindSlotsArgs(ToolArgs):
    date: date
    department: str | None = None
    doctor_name: str | None = None


async def find_slots(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    parsed = FindSlotsArgs.model_validate(args)
    department = None
    if parsed.department:
        department = await _match_department(parsed.department)
        if department is None:
            return {"error": f"There is no department called {parsed.department}."}
    found = await booking.find_availability(parsed.date, department)
    if parsed.doctor_name:
        wanted = parsed.doctor_name.lower().removeprefix("dr.").removeprefix("dr").strip()
        found = [f for f in found if wanted in f.doctor.name.lower()]
    return {
        "date": parsed.date.isoformat(),
        "doctors": [
            {
                "doctor_id": str(f.doctor.id),
                "doctor_name": f.doctor.name,
                "department": f.doctor.department,
                "open_slots": len(f.slots),
                "first_slots": [
                    {"slot_start": start.isoformat(), "local_time": local_time(start)}
                    for start, _ in f.slots[:MAX_SLOTS_PER_DOCTOR]
                ],
            }
            for f in found
        ],
    }


class BookSlotArgs(ToolArgs):
    doctor_id: PydanticObjectId
    slot_start: AwareDatetime
    reason: str | None = Field(default=None, max_length=300)


async def book_slot(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not context.is_patient:
        return PATIENTS_ONLY
    parsed = BookSlotArgs.model_validate(args)
    doctor = await booking.get_doctor(parsed.doctor_id)
    await _require_open(parsed.doctor_id, parsed.slot_start)
    summary = f"Book {doctor.name} ({doctor.department}) on {local_time(parsed.slot_start)}"
    if proposal := await confirm_or_propose(context, "book_slot", parsed, summary):
        return proposal
    appointment, doctor = await booking.book(
        context.user, parsed.doctor_id, parsed.slot_start, parsed.reason
    )
    return {"booked": _describe(appointment, doctor.name)}


async def list_my_appointments(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not context.is_patient:
        return PATIENTS_ONLY
    now = now_utc()
    upcoming = [
        _describe(appointment, doctor_name)
        for appointment, doctor_name in await booking.history(context.user)
        if appointment.status == AppointmentStatus.BOOKED and as_utc(appointment.slot_start) > now
    ]
    return {"upcoming": sorted(upcoming, key=lambda a: a["slot_start"])}


class AppointmentArgs(ToolArgs):
    appointment_id: PydanticObjectId


async def cancel_appointment(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not context.is_patient:
        return PATIENTS_ONLY
    parsed = AppointmentArgs.model_validate(args)
    current = await booking.own_changeable(context.user, parsed.appointment_id)
    doctor = await booking.get_doctor(current.doctor_id)
    summary = f"Cancel the appointment with {doctor.name} on {local_time(current.slot_start)}"
    if proposal := await confirm_or_propose(context, "cancel_appointment", parsed, summary):
        return proposal
    appointment, doctor = await booking.cancel(context.user, parsed.appointment_id)
    return {"cancelled": _describe(appointment, doctor.name)}


class RescheduleArgs(AppointmentArgs):
    slot_start: AwareDatetime


async def reschedule_appointment(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not context.is_patient:
        return PATIENTS_ONLY
    parsed = RescheduleArgs.model_validate(args)
    current = await booking.own_changeable(context.user, parsed.appointment_id)
    doctor = await booking.get_doctor(current.doctor_id)
    await _require_open(current.doctor_id, parsed.slot_start)
    summary = (
        f"Move the appointment with {doctor.name} from {local_time(current.slot_start)} "
        f"to {local_time(parsed.slot_start)}"
    )
    if proposal := await confirm_or_propose(context, "reschedule_appointment", parsed, summary):
        return proposal
    appointment, doctor = await booking.reschedule(
        context.user, parsed.appointment_id, parsed.slot_start
    )
    return {"rescheduled": _describe(appointment, doctor.name)}


_SLOT_START = {
    "type": "string",
    "description": "Exact slot_start value returned by find_slots (ISO 8601 with offset).",
}
_APPOINTMENT_ID = {
    "type": "string",
    "description": "appointment_id returned by list_my_appointments.",
}

BOOKING_TOOLS: tuple[Tool, ...] = (
    Tool(
        "find_slots",
        "List open appointment slots on one date, optionally for one department or doctor.",
        schema(
            {
                "date": {"type": "string", "description": "Date as YYYY-MM-DD."},
                "department": {"type": "string", "description": "Department name, if given."},
                "doctor_name": {"type": "string", "description": "Doctor's name, if given."},
            },
            ["date"],
        ),
        find_slots,
    ),
    Tool(
        "book_slot",
        "Book one open slot for the logged-in patient. The first call only proposes it.",
        schema(
            {
                "doctor_id": {"type": "string", "description": "doctor_id from find_slots."},
                "slot_start": _SLOT_START,
                "reason": {"type": "string", "description": "Short reason for the visit."},
            },
            ["doctor_id", "slot_start"],
        ),
        book_slot,
    ),
    Tool(
        "list_my_appointments",
        "List the logged-in patient's upcoming appointments.",
        schema({}),
        list_my_appointments,
    ),
    Tool(
        "cancel_appointment",
        "Cancel one of the logged-in patient's upcoming appointments.",
        schema({"appointment_id": _APPOINTMENT_ID}, ["appointment_id"]),
        cancel_appointment,
    ),
    Tool(
        "reschedule_appointment",
        "Move one of the logged-in patient's upcoming appointments to another open slot "
        "with the same doctor.",
        schema(
            {"appointment_id": _APPOINTMENT_ID, "slot_start": _SLOT_START},
            ["appointment_id", "slot_start"],
        ),
        reschedule_appointment,
    ),
)
