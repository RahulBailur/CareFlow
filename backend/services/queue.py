from dataclasses import dataclass
from datetime import datetime, timedelta

from beanie import PydanticObjectId
from beanie.operators import In
from pydantic import BaseModel

from models.appointment import Appointment, AppointmentStatus
from models.doctor_status import DoctorStatus
from models.user import User
from time_utils import as_utc, ist_day_bounds, today_ist

WAITING_STATUSES = (AppointmentStatus.BOOKED, AppointmentStatus.IN_CONSULTATION)


class PatientQueueView(BaseModel):
    """What one patient may see: their own appointment and position, nobody else's."""

    appointment_id: str
    doctor_id: str
    doctor_name: str
    department: str
    status: AppointmentStatus
    in_queue: bool
    patients_ahead: int | None
    slot_start: datetime
    eta: datetime | None
    delay_minutes: int
    delay_message: str | None


class DoctorQueueEntry(BaseModel):
    appointment_id: str
    patient_name: str
    slot_start: datetime
    status: AppointmentStatus
    reason: str | None


class DoctorQueueView(BaseModel):
    doctor_id: str
    doctor_name: str
    department: str | None
    delay_minutes: int
    delay_message: str | None
    waiting: int
    entries: list[DoctorQueueEntry]


@dataclass
class QueueSnapshot:
    doctor: User
    appointments: list[Appointment]  # today's, not cancelled, in slot order
    delay_minutes: int
    delay_message: str | None


async def current_delay(doctor_id: PydanticObjectId | None) -> tuple[int, str | None]:
    status = await DoctorStatus.find_one(DoctorStatus.doctor_id == doctor_id)
    if status is None or status.day != today_ist().isoformat():
        return 0, None
    return status.delay_minutes, status.message


async def load_queue(doctor: User) -> QueueSnapshot:
    day_start, day_end = ist_day_bounds(today_ist())
    appointments = (
        await Appointment.find(
            Appointment.doctor_id == doctor.id,
            Appointment.slot_start >= day_start,
            Appointment.slot_start < day_end,
            Appointment.status != AppointmentStatus.CANCELLED,
        )
        .sort("+slot_start")
        .to_list()
    )
    delay_minutes, delay_message = await current_delay(doctor.id)
    return QueueSnapshot(doctor, appointments, delay_minutes, delay_message)


def patient_view(snapshot: QueueSnapshot, appointment: Appointment) -> PatientQueueView:
    in_queue = appointment.status in WAITING_STATUSES
    ahead = sum(
        1
        for other in snapshot.appointments
        if other.status in WAITING_STATUSES and other.slot_start < appointment.slot_start
    )
    slot_start = as_utc(appointment.slot_start)
    return PatientQueueView(
        appointment_id=str(appointment.id),
        doctor_id=str(snapshot.doctor.id),
        doctor_name=snapshot.doctor.name,
        department=appointment.department,
        status=appointment.status,
        in_queue=in_queue,
        patients_ahead=ahead if in_queue else None,
        slot_start=slot_start,
        eta=slot_start + timedelta(minutes=snapshot.delay_minutes) if in_queue else None,
        delay_minutes=snapshot.delay_minutes,
        delay_message=snapshot.delay_message,
    )


async def doctor_view(snapshot: QueueSnapshot) -> DoctorQueueView:
    patient_ids = [appointment.patient_id for appointment in snapshot.appointments]
    patients = await User.find(In(User.id, patient_ids)).to_list()
    names = {patient.id: patient.name for patient in patients}
    return DoctorQueueView(
        doctor_id=str(snapshot.doctor.id),
        doctor_name=snapshot.doctor.name,
        department=snapshot.doctor.department,
        delay_minutes=snapshot.delay_minutes,
        delay_message=snapshot.delay_message,
        waiting=sum(1 for a in snapshot.appointments if a.status == AppointmentStatus.BOOKED),
        entries=[
            DoctorQueueEntry(
                appointment_id=str(appointment.id),
                patient_name=names.get(appointment.patient_id, "Unknown"),
                slot_start=as_utc(appointment.slot_start),
                status=appointment.status,
                reason=appointment.reason,
            )
            for appointment in snapshot.appointments
        ],
    )
