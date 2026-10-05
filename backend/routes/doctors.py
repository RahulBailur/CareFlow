from typing import Annotated

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth_utils import CurrentUser, require_role
from models.doctor_status import DoctorStatus
from models.schedule import Schedule
from models.user import Role, User
from sockets import broadcast_queue
from time_utils import now_utc, today_ist

router = APIRouter(prefix="/api/doctors", tags=["doctors"])

DoctorUser = Annotated[User, Depends(require_role(Role.DOCTOR))]


class ScheduleOut(BaseModel):
    weekday: int
    start_time: str
    end_time: str
    slot_minutes: int


class DoctorScheduleOut(BaseModel):
    doctor_id: str
    doctor_name: str
    department: str | None
    sessions: list[ScheduleOut]


class DelayRequest(BaseModel):
    delay_minutes: int = Field(ge=0, le=240)
    message: str | None = Field(default=None, max_length=200)


class DelayOut(BaseModel):
    delay_minutes: int
    message: str | None


@router.get("/{doctor_id}/schedule")
async def schedule(user: CurrentUser, doctor_id: PydanticObjectId) -> DoctorScheduleOut:
    doctor = await User.get(doctor_id)
    if doctor is None or doctor.role != Role.DOCTOR:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Doctor not found")
    sessions = await Schedule.find(Schedule.doctor_id == doctor_id).sort("+weekday").to_list()
    return DoctorScheduleOut(
        doctor_id=str(doctor.id),
        doctor_name=doctor.name,
        department=doctor.department,
        sessions=[
            ScheduleOut(
                weekday=s.weekday,
                start_time=s.start_time,
                end_time=s.end_time,
                slot_minutes=s.slot_minutes,
            )
            for s in sessions
        ],
    )


@router.post("/status")
async def broadcast_delay(doctor: DoctorUser, body: DelayRequest) -> DelayOut:
    """Set today's delay for the logged-in doctor and push it to every waiting patient."""
    if doctor.id is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Doctor not found")
    current = await DoctorStatus.find_one(DoctorStatus.doctor_id == doctor.id)
    if current is None:
        current = DoctorStatus(doctor_id=doctor.id, day=today_ist().isoformat())
    current.day = today_ist().isoformat()
    current.delay_minutes = body.delay_minutes
    current.message = body.message
    current.updated_at = now_utc()
    await current.save()
    await broadcast_queue(doctor)
    return DelayOut(delay_minutes=current.delay_minutes, message=current.message)
