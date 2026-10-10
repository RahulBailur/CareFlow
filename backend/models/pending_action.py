from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel

from time_utils import now_utc

PENDING_TTL_SECONDS = 15 * 60


class PendingAction(Document):
    """A change CareBot has proposed and the patient has not yet confirmed.

    One per conversation. It only becomes executable in the turn after the one that
    proposed it, so the patient always hears the proposal before anything changes.
    """

    user_id: PydanticObjectId
    session_id: str
    tool: str
    args_key: str  # canonical JSON of the validated arguments
    turn_id: str  # the turn that proposed it
    created_at: datetime = Field(default_factory=now_utc)

    class Settings:
        name = "pending_actions"
        indexes = [
            IndexModel([("user_id", ASCENDING), ("session_id", ASCENDING)], unique=True),
            IndexModel([("created_at", ASCENDING)], expireAfterSeconds=PENDING_TTL_SECONDS),
        ]
