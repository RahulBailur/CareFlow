from datetime import date, datetime
from typing import Annotated, Literal

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from auth_utils import CurrentUser, require_role
from models.appointment import Appointment, AppointmentStatus
from models.user import Role, User
from services import appointments as booking
from services.queue import (
    DoctorQueueView,
    PatientQueueView,
    doctor_view,
    load_queue,
    patient_view,
)
from sockets import broadcast_queue
from time_utils import as_utc

router = APIRouter(prefix="/api/appointments", tags=["appointments"])

PatientUser = Annotated[User, Depends(require_role(Role.PATIENT))]
DoctorUser = Annotated[User, Depends(require_role(Role.DOCTOR))]

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


@router.get("/availability")
async def availability(
    user: CurrentUser,
    day: Annotated[date | None, Query(alias="date")] = None,
    department: str | None = None,
    doctor_id: PydanticObjectId | None = None,
) -> list[DoctorAvailability]:
    return [
        DoctorAvailability(
            doctor_id=str(found.doctor.id),
            doctor_name=found.doctor.name,
            department=found.doctor.department,
            slots=[SlotOut(start=start, end=end) for start, end in found.slots],
        )
        for found in await booking.find_availability(day, department, doctor_id)
    ]


@router.post("/book", status_code=status.HTTP_201_CREATED)
async def book(patient: PatientUser, body: BookRequest) -> AppointmentOut:
    appointment, doctor = await booking.book(patient, body.doctor_id, body.slot_start, body.reason)
    return AppointmentOut.build(appointment, doctor.name)


@router.get("/me")
async def my_appointments(patient: PatientUser) -> list[AppointmentOut]:
    """The logged-in patient's own visits: the last 12 months plus anything upcoming."""
    return [AppointmentOut.build(a, name) for a, name in await booking.history(patient)]


@router.get("/queue/{doctor_id}")
async def queue(
    user: CurrentUser, doctor_id: PydanticObjectId
) -> DoctorQueueView | PatientQueueView:
    """Doctors (own queue) and admins get the full list; a patient gets only their own place."""
    doctor = await booking.get_doctor(doctor_id)
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
    if body.action == "cancel":
        appointment, doctor = await booking.cancel(patient, appointment_id)
    else:
        assert body.slot_start is not None  # noqa: S101 — enforced by ChangeRequest
        appointment, doctor = await booking.reschedule(patient, appointment_id, body.slot_start)
    return AppointmentOut.build(appointment, doctor.name)
