from datetime import datetime, time, timedelta
from typing import Any

from auth_utils import create_access_token, hash_password
from models.appointment import Appointment, AppointmentStatus
from models.schedule import Schedule
from models.user import Role, User
from time_utils import IST, as_utc, today_ist

_counter = 0


async def make_user(role: Role = Role.PATIENT, name: str | None = None) -> User:
    global _counter
    _counter += 1
    user = User(
        name=name or f"{role.value.title()} {_counter}",
        email=f"{role.value}{_counter}@example.com",
        phone=f"9{_counter:09d}",
        password_hash=hash_password("password-1"),
        role=role,
        department="Cardiology" if role == Role.DOCTOR else None,
        medical_registration_number="SYN-TEST" if role == Role.DOCTOR else None,
    )
    await user.insert()
    return user


async def make_doctor(name: str | None = None) -> User:
    """A doctor with a 09:00–13:00 session, 15-minute slots, every day of the week."""
    doctor = await make_user(Role.DOCTOR, name)
    assert doctor.id is not None
    for weekday in range(7):
        await Schedule(
            doctor_id=doctor.id,
            department="Cardiology",
            weekday=weekday,
            start_time="09:00",
            end_time="13:00",
            slot_minutes=15,
        ).insert()
    return doctor


def auth(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user)}"}


def slot(days_ahead: int, index: int = 0) -> datetime:
    """UTC start of the Nth schedule slot, `days_ahead` days from today (IST)."""
    day = today_ist() + timedelta(days=days_ahead)
    start = datetime.combine(day, time(9, 0), tzinfo=IST) + timedelta(minutes=15 * index)
    return as_utc(start)


async def make_appointment(
    patient: User,
    doctor: User,
    slot_start: datetime,
    status: AppointmentStatus = AppointmentStatus.BOOKED,
    prescription: str | None = None,
) -> Appointment:
    """Insert directly, bypassing the API — lets tests place appointments today or in the past."""
    assert patient.id is not None and doctor.id is not None
    appointment = Appointment(
        patient_id=patient.id,
        doctor_id=doctor.id,
        department="Cardiology",
        slot_start=slot_start,
        slot_end=slot_start + timedelta(minutes=15),
        status=status,
        prescription=prescription,
    )
    await appointment.insert()
    return appointment


async def confirmed_tool_call(tool: Any, user: User, args: dict[str, Any]) -> dict[str, Any]:
    """Call a changing tool as a confirmed conversation does: propose, then repeat next turn."""
    from tools import ToolContext, run_tool

    proposal = await run_tool(tool, ToolContext(user=user, session_id="s", turn_id="t1"), args)
    if not proposal.get("needs_confirmation"):
        return proposal
    return await run_tool(tool, ToolContext(user=user, session_id="s", turn_id="t2"), args)
