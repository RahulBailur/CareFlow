from datetime import date, datetime, time, timedelta

from beanie import PydanticObjectId

from models.appointment import Appointment, AppointmentStatus
from models.schedule import Schedule
from time_utils import IST, as_utc, ist_day_bounds, now_utc

Slot = tuple[datetime, datetime]  # UTC start, UTC end


async def day_slots(doctor_id: PydanticObjectId, day: date) -> list[Slot]:
    """Every slot the doctor's schedule defines for a hospital-local day, booked or not."""
    sessions = await Schedule.find(
        Schedule.doctor_id == doctor_id, Schedule.weekday == day.weekday()
    ).to_list()
    slots: list[Slot] = []
    for session in sessions:
        step = timedelta(minutes=session.slot_minutes)
        start = datetime.combine(day, time.fromisoformat(session.start_time), tzinfo=IST)
        close = datetime.combine(day, time.fromisoformat(session.end_time), tzinfo=IST)
        while start + step <= close:
            slots.append((as_utc(start), as_utc(start + step)))
            start += step
    return sorted(slots)


async def open_slots(doctor_id: PydanticObjectId, day: date) -> list[Slot]:
    """Future slots on that day that nobody holds."""
    day_start, day_end = ist_day_bounds(day)
    held = await Appointment.find(
        Appointment.doctor_id == doctor_id,
        Appointment.slot_start >= day_start,
        Appointment.slot_start < day_end,
        Appointment.status != AppointmentStatus.CANCELLED,
    ).to_list()
    taken = {as_utc(appointment.slot_start) for appointment in held}
    now = now_utc()
    slots = await day_slots(doctor_id, day)
    return [slot for slot in slots if slot[0] > now and slot[0] not in taken]


async def bookable_slot(doctor_id: PydanticObjectId, slot_start: datetime) -> Slot | None:
    """The schedule slot starting exactly then, if it exists and is still in the future."""
    start = as_utc(slot_start)
    if start <= now_utc():
        return None
    for slot in await day_slots(doctor_id, start.astimezone(IST).date()):
        if slot[0] == start:
            return slot
    return None
