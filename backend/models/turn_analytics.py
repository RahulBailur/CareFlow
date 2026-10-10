from datetime import datetime

from beanie import Document
from pydantic import Field
from pymongo import DESCENDING, IndexModel

from time_utils import now_utc


class TurnAnalytics(Document):
    """What one voice turn cost, stage by stage.

    Timings and labels only. No transcript, no reply text, no audio, and no user ID:
    nothing here can be read back as what a patient said.
    """

    pipeline: str  # "cascade" | "live"
    intent: str
    language: str
    routed_by: str
    used_llm: bool
    cache_hit: bool = False
    # The fallback tier that answered: "cache", a model's name, or "rules"
    answered_by: str = ""
    blocked: bool
    tools_called: list[str]
    stt_provider: str
    tts_provider: str  # the voice that actually spoke; empty if none could
    spoken: bool  # False when the reply could only be shown as text
    barged_in: bool
    stages: dict[str, float]  # milliseconds, see voice/timing.py
    created_at: datetime = Field(default_factory=now_utc)

    class Settings:
        name = "turn_analytics"
        indexes = [IndexModel([("created_at", DESCENDING)])]
