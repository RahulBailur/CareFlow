from datetime import datetime
from typing import Literal

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel

from time_utils import now_utc

MEMORY_TTL_SECONDS = 24 * 60 * 60


class ConversationTurn(Document):
    """One line of a CareBot conversation, kept so follow-up questions resolve.

    Keyed by user as well as session, so a session ID alone never opens someone else's
    conversation. MongoDB deletes turns a day after they were written.
    """

    user_id: PydanticObjectId
    session_id: str
    role: Literal["user", "assistant"]
    text: str
    intent: str
    # What CareBot's tools did or proposed in this turn; shown to the model, not the patient
    note: str = ""
    created_at: datetime = Field(default_factory=now_utc)

    class Settings:
        name = "conversation_turns"
        indexes = [
            IndexModel(
                [("user_id", ASCENDING), ("session_id", ASCENDING), ("created_at", ASCENDING)]
            ),
            IndexModel([("created_at", ASCENDING)], expireAfterSeconds=MEMORY_TTL_SECONDS),
        ]
