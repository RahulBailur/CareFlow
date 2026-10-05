from datetime import UTC, datetime
from enum import StrEnum

from beanie import Document
from pydantic import Field
from pymongo import ASCENDING, IndexModel


class Role(StrEnum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"


class User(Document):
    name: str
    email: str  # stored lowercased
    phone: str
    password_hash: str
    role: Role = Role.PATIENT
    # Doctor-only fields, set by the seed script
    department: str | None = None
    medical_registration_number: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Settings:
        name = "users"
        indexes = [IndexModel([("email", ASCENDING)], unique=True)]
