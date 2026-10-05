from datetime import UTC, datetime
from enum import StrEnum

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel


class AppointmentStatus(StrEnum):
    BOOKED = "booked"
    IN_CONSULTATION = "in_consultation"
    DONE = "done"
    NO_SHOW = "no_show"
    CANCELLED = "cancelled"


class Appointment(Document):
    patient_id: PydanticObjectId
    doctor_id: PydanticObjectId
    department: str
    slot_start: datetime  # UTC
    slot_end: datetime  # UTC
    status: AppointmentStatus = AppointmentStatus.BOOKED
    reason: str | None = None
    prescription: str | None = None  # view-only in visit history
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Settings:
        name = "appointments"
        indexes = [
            # No double booking: a second insert for the same slot fails
            IndexModel([("doctor_id", ASCENDING), ("slot_start", ASCENDING)], unique=True),
            IndexModel([("patient_id", ASCENDING), ("slot_start", ASCENDING)]),
        ]
