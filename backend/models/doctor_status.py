from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel

from time_utils import now_utc


class DoctorStatus(Document):
    """A doctor's delay broadcast. Only counts on the hospital-local day it was set."""

    doctor_id: PydanticObjectId
    day: str  # IST date, YYYY-MM-DD
    delay_minutes: int = 0
    message: str | None = None
    updated_at: datetime = Field(default_factory=now_utc)

    class Settings:
        name = "doctor_status"
        indexes = [IndexModel([("doctor_id", ASCENDING)], unique=True)]
