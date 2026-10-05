from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel

HHMM = r"^([01]\d|2[0-3]):[0-5]\d$"


class Schedule(Document):
    """A doctor's recurring OPD session on one weekday. Seeded; not editable in the POC."""

    doctor_id: PydanticObjectId
    department: str
    weekday: int = Field(ge=0, le=6)  # Monday = 0
    start_time: str = Field(pattern=HHMM)  # hospital local time (IST)
    end_time: str = Field(pattern=HHMM)
    slot_minutes: int = Field(default=15, gt=0)

    class Settings:
        name = "schedules"
        indexes = [
            IndexModel(
                [("doctor_id", ASCENDING), ("weekday", ASCENDING), ("start_time", ASCENDING)],
                unique=True,
            ),
        ]
