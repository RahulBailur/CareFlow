from datetime import datetime
from enum import StrEnum
from typing import Self

from beanie import Document, PydanticObjectId
from pydantic import Field, model_validator
from pymongo import ASCENDING, IndexModel

from time_utils import as_utc, now_utc


class AppointmentStatus(StrEnum):
    BOOKED = "booked"
    IN_CONSULTATION = "in_consultation"
    DONE = "done"
    NO_SHOW = "no_show"
    CANCELLED = "cancelled"


def slot_key_for(slot_start: datetime) -> str:
    return as_utc(slot_start).strftime("%Y-%m-%dT%H:%MZ")


class Appointment(Document):
    patient_id: PydanticObjectId
    doctor_id: PydanticObjectId
    department: str
    slot_start: datetime  # UTC
    slot_end: datetime  # UTC
    status: AppointmentStatus = AppointmentStatus.BOOKED
    reason: str | None = None
    prescription: str | None = None  # view-only in visit history
    created_at: datetime = Field(default_factory=now_utc)
    # What the no-double-booking index is built on. It equals the slot start while the
    # appointment holds its slot; cancelling rewrites it to a per-appointment value, which
    # frees the slot for someone else without deleting the record.
    slot_key: str = ""

    @model_validator(mode="after")
    def _default_slot_key(self) -> Self:
        if not self.slot_key:
            self.slot_key = slot_key_for(self.slot_start)
        return self

    def move_to(self, slot_start: datetime, slot_end: datetime) -> None:
        self.slot_start = slot_start
        self.slot_end = slot_end
        self.slot_key = slot_key_for(slot_start)

    def cancel(self) -> None:
        self.status = AppointmentStatus.CANCELLED
        self.slot_key = f"cancelled:{self.id}"

    class Settings:
        name = "appointments"
        indexes = [
            # No double booking: a second insert for the same doctor and slot fails
            IndexModel([("doctor_id", ASCENDING), ("slot_key", ASCENDING)], unique=True),
            IndexModel([("doctor_id", ASCENDING), ("slot_start", ASCENDING)]),
            IndexModel([("patient_id", ASCENDING), ("slot_start", ASCENDING)]),
        ]
