"""Appointment rules, shared by the HTTP routes and CareBot's booking tools.

Every function that acts for a patient takes the `User` it is acting for. That user always
comes from the verified session, so neither a request body nor an LLM can choose it.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from beanie import PydanticObjectId
from beanie.exceptions import RevisionIdWasChanged
from beanie.operators import In
from pymongo.errors import DuplicateKeyError

from models.appointment import Appointment, AppointmentStatus
from models.user import Role, User
from services.scheduling import Slot, bookable_slot, open_slots
from sockets import broadcast_queue
from time_utils import now_utc, today_ist

BOOKING_WINDOW_DAYS = 30
HISTORY_DAYS = 365


class BookingError(Exception):
    """A request the appointment rules refuse. Carries the HTTP status the API answers with."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _slot_taken() -> BookingError:
    return BookingError(409, "That slot has just been taken")


def _slot_unavailable() -> BookingError:
    return BookingError(400, "That is not an open slot in the doctor's schedule")


def _not_found() -> BookingError:
    return BookingError(404, "Appointment not found")


@dataclass(frozen=True)
class DoctorSlots:
    doctor: User
    slots: list[Slot]


async def get_doctor(doctor_id: PydanticObjectId) -> User:
    doctor = await User.get(doctor_id)
    if doctor is None or doctor.role != Role.DOCTOR:
        raise BookingError(404, "Doctor not found")
    return doctor


def check_bookable_day(day: date) -> None:
    today = today_ist()
    if not today <= day <= today + timedelta(days=BOOKING_WINDOW_DAYS):
        raise BookingError(400, f"Date must be between today and {BOOKING_WINDOW_DAYS} days ahead")


async def find_availability(
    day: date | None = None,
    department: str | None = None,
    doctor_id: PydanticObjectId | None = None,
) -> list[DoctorSlots]:
    day = day or today_ist()
    check_bookable_day(day)
    if doctor_id is not None:
        doctors = [await get_doctor(doctor_id)]
    elif department is not None:
        doctors = await User.find(User.role == Role.DOCTOR, User.department == department).to_list()
    else:
        doctors = await User.find(User.role == Role.DOCTOR).to_list()
    return [
        DoctorSlots(doctor, await open_slots(doctor.id, day))
        for doctor in doctors
        if doctor.id is not None
    ]


async def book(
    patient: User, doctor_id: PydanticObjectId, slot_start: datetime, reason: str | None = None
) -> tuple[Appointment, User]:
    doctor = await get_doctor(doctor_id)
    slot = await bookable_slot(doctor_id, slot_start)
    if slot is None or patient.id is None:
        raise _slot_unavailable()
    appointment = Appointment(
        patient_id=patient.id,
        doctor_id=doctor_id,
        department=doctor.department or "",
        slot_start=slot[0],
        slot_end=slot[1],
        reason=reason,
    )
    try:
        await appointment.insert()
    except DuplicateKeyError:
        raise _slot_taken() from None
    await broadcast_queue(doctor)
    return appointment, doctor


async def _own_changeable(patient: User, appointment_id: PydanticObjectId) -> Appointment:
    """The patient's own appointment, if it can still be changed. Anyone else's is 'not found'."""
    appointment = await Appointment.get(appointment_id)
    if appointment is None or appointment.patient_id != patient.id:
        raise _not_found()
    if appointment.status != AppointmentStatus.BOOKED:
        raise BookingError(409, f"A {appointment.status} appointment cannot be changed")
    return appointment


async def cancel(patient: User, appointment_id: PydanticObjectId) -> tuple[Appointment, User]:
    appointment = await _own_changeable(patient, appointment_id)
    doctor = await get_doctor(appointment.doctor_id)
    appointment.cancel()
    await appointment.save()
    await broadcast_queue(doctor, also=appointment)
    return appointment, doctor


async def reschedule(
    patient: User, appointment_id: PydanticObjectId, slot_start: datetime
) -> tuple[Appointment, User]:
    appointment = await _own_changeable(patient, appointment_id)
    doctor = await get_doctor(appointment.doctor_id)
    slot = await bookable_slot(appointment.doctor_id, slot_start)
    if slot is None:
        raise _slot_unavailable()
    appointment.move_to(*slot)
    try:
        await appointment.save()
    except (DuplicateKeyError, RevisionIdWasChanged):  # beanie re-raises the former as the latter
        raise _slot_taken() from None
    await broadcast_queue(doctor)
    return appointment, doctor


async def history(patient: User) -> list[tuple[Appointment, str]]:
    """The patient's own visits with the doctor's name: the last 12 months plus upcoming."""
    since = now_utc() - timedelta(days=HISTORY_DAYS)
    appointments = (
        await Appointment.find(
            Appointment.patient_id == patient.id, Appointment.slot_start >= since
        )
        .sort("-slot_start")
        .to_list()
    )
    doctors = await User.find(In(User.id, [a.doctor_id for a in appointments])).to_list()
    names = {doctor.id: doctor.name for doctor in doctors}
    return [(a, names.get(a.doctor_id, "Unknown")) for a in appointments]
