from datetime import date, datetime, timedelta
from typing import Annotated, Literal

from beanie import PydanticObjectId
from beanie.exceptions import RevisionIdWasChanged
from beanie.operators import In
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import AwareDatetime, BaseModel, Field, model_validator
from pymongo.errors import DuplicateKeyError

from auth_utils import CurrentUser, require_role
from models.appointment import Appointment, AppointmentStatus
from models.user import Role, User
from services.queue import (
    DoctorQueueView,
    PatientQueueView,
    doctor_view,
    load_queue,
    patient_view,
)
from services.scheduling import bookable_slot, open_slots
from sockets import broadcast_queue
from time_utils import as_utc, now_utc, today_ist

router = APIRouter(prefix="/api/appointments", tags=["appointments"])

PatientUser = Annotated[User, Depends(require_role(Role.PATIENT))]
DoctorUser = Annotated[User, Depends(require_role(Role.DOCTOR))]

BOOKING_WINDOW_DAYS = 30
HISTORY_DAYS = 365

SLOT_TAKEN = HTTPException(status.HTTP_409_CONFLICT, "That slot has just been taken")
SLOT_UNAVAILABLE = HTTPException(
    status.HTTP_400_BAD_REQUEST, "That is not an open slot in the doctor's schedule"
)
NOT_FOUND = HTTPException(status.HTTP_404_NOT_FOUND, "Appointment not found")

# What a doctor may move an appointment to, from each state
STATUS_TRANSITIONS: dict[AppointmentStatus, set[AppointmentStatus]] = {
    AppointmentStatus.BOOKED: {AppointmentStatus.IN_CONSULTATION, AppointmentStatus.NO_SHOW},
    AppointmentStatus.IN_CONSULTATION: {AppointmentStatus.DONE},
}


class SlotOut(BaseModel):
    start: datetime
    end: datetime


class DoctorAvailability(BaseModel):
    doctor_id: str
    doctor_name: str
    department: str | None
    slots: list[SlotOut]


class BookRequest(BaseModel):
    """There is no patient field: the appointment is always for the logged-in patient."""

    doctor_id: PydanticObjectId
    slot_start: AwareDatetime
    reason: str | None = Field(default=None, max_length=300)


class ChangeRequest(BaseModel):
    action: Literal["reschedule", "cancel"]
    slot_start: AwareDatetime | None = None

    @model_validator(mode="after")
    def _reschedule_needs_slot(self) -> "ChangeRequest":
        if self.action == "reschedule" and self.slot_start is None:
            raise ValueError("slot_start is required to reschedule")
        return self


class StatusRequest(BaseModel):
    status: Literal[
        AppointmentStatus.IN_CONSULTATION, AppointmentStatus.DONE, AppointmentStatus.NO_SHOW
    ]


class AppointmentOut(BaseModel):
    id: str
    doctor_id: str
    doctor_name: str
    department: str
    slot_start: datetime
    slot_end: datetime
    status: AppointmentStatus
    reason: str | None
    prescription: str | None

    @classmethod
    def build(cls, appointment: Appointment, doctor_name: str) -> "AppointmentOut":
        return cls(
            id=str(appointment.id),
            doctor_id=str(appointment.doctor_id),
            doctor_name=doctor_name,
            department=appointment.department,
            slot_start=as_utc(appointment.slot_start),
            slot_end=as_utc(appointment.slot_end),
            status=appointment.status,
            reason=appointment.reason,
            prescription=appointment.prescription,
        )


async def _get_doctor(doctor_id: PydanticObjectId) -> User:
    doctor = await User.get(doctor_id)
    if doctor is None or doctor.role != Role.DOCTOR:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Doctor not found")
    return doctor


@router.get("/availability")
async def availability(
    user: CurrentUser,
    day: Annotated[date | None, Query(alias="date")] = None,
    department: str | None = None,
    doctor_id: PydanticObjectId | None = None,
) -> list[DoctorAvailability]:
    today = today_ist()
    day = day or today
    if not today <= day <= today + timedelta(days=BOOKING_WINDOW_DAYS):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Date must be between today and {BOOKING_WINDOW_DAYS} days ahead",
        )
    if doctor_id is not None:
        doctors = [await _get_doctor(doctor_id)]
    elif department is not None:
        doctors = await User.find(User.role == Role.DOCTOR, User.department == department).to_list()
    else:
        doctors = await User.find(User.role == Role.DOCTOR).to_list()

    result = []
    for doctor in doctors:
        if doctor.id is None:
            continue
        slots = await open_slots(doctor.id, day)
        result.append(
            DoctorAvailability(
                doctor_id=str(doctor.id),
                doctor_name=doctor.name,
                department=doctor.department,
                slots=[SlotOut(start=start, end=end) for start, end in slots],
            )
        )
    return result


@router.post("/book", status_code=status.HTTP_201_CREATED)
async def book(patient: PatientUser, body: BookRequest) -> AppointmentOut:
    doctor = await _get_doctor(body.doctor_id)
    slot = await bookable_slot(body.doctor_id, body.slot_start)
    if slot is None or patient.id is None:
        raise SLOT_UNAVAILABLE
    appointment = Appointment(
        patient_id=patient.id,
        doctor_id=body.doctor_id,
        department=doctor.department or "",
        slot_start=slot[0],
        slot_end=slot[1],
        reason=body.reason,
    )
    try:
        await appointment.insert()
    except DuplicateKeyError:
        raise SLOT_TAKEN from None
    await broadcast_queue(doctor)
    return AppointmentOut.build(appointment, doctor.name)


@router.get("/me")
async def my_appointments(patient: PatientUser) -> list[AppointmentOut]:
    """The logged-in patient's own visits: the last 12 months plus anything upcoming."""
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
    return [AppointmentOut.build(a, names.get(a.doctor_id, "Unknown")) for a in appointments]


@router.get("/queue/{doctor_id}")
async def queue(
    user: CurrentUser, doctor_id: PydanticObjectId
) -> DoctorQueueView | PatientQueueView:
    """Doctors (own queue) and admins get the full list; a patient gets only their own place."""
    doctor = await _get_doctor(doctor_id)
    if user.role == Role.DOCTOR and user.id != doctor.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not permitted")
    snapshot = await load_queue(doctor)
    if user.role != Role.PATIENT:
        return await doctor_view(snapshot)
    mine = [a for a in snapshot.appointments if a.patient_id == user.id]
    if not mine:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "You have no appointment with this doctor today"
        )
    # Prefer the appointment still waiting, if the patient has more than one today
    current = next((a for a in mine if a.status == AppointmentStatus.BOOKED), mine[-1])
    return patient_view(snapshot, current)


@router.put("/{appointment_id}/status")
async def update_status(
    doctor: DoctorUser, appointment_id: PydanticObjectId, body: StatusRequest
) -> AppointmentOut:
    appointment = await Appointment.get(appointment_id)
    if appointment is None or appointment.doctor_id != doctor.id:
        raise NOT_FOUND
    if body.status not in STATUS_TRANSITIONS.get(appointment.status, set()):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Cannot change status from {appointment.status} to {body.status}",
        )
    appointment.status = AppointmentStatus(body.status)
    await appointment.save()
    await broadcast_queue(doctor)
    return AppointmentOut.build(appointment, doctor.name)


@router.put("/{appointment_id}")
async def change_appointment(
    patient: PatientUser, appointment_id: PydanticObjectId, body: ChangeRequest
) -> AppointmentOut:
    """Reschedule or cancel: only by the patient who owns it, and only before it starts."""
    appointment = await Appointment.get(appointment_id)
    if appointment is None or appointment.patient_id != patient.id:
        raise NOT_FOUND
    if appointment.status != AppointmentStatus.BOOKED:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A {appointment.status} appointment cannot be changed"
        )
    doctor = await _get_doctor(appointment.doctor_id)

    if body.action == "cancel":
        appointment.cancel()
        await appointment.save()
        await broadcast_queue(doctor, also=appointment)
        return AppointmentOut.build(appointment, doctor.name)

    slot = await bookable_slot(appointment.doctor_id, body.slot_start) if body.slot_start else None
    if slot is None:
        raise SLOT_UNAVAILABLE
    appointment.move_to(*slot)
    try:
        await appointment.save()
    except (DuplicateKeyError, RevisionIdWasChanged):  # beanie re-raises the former as the latter
        raise SLOT_TAKEN from None
    await broadcast_queue(doctor)
    return AppointmentOut.build(appointment, doctor.name)
