from beanie import Document
from pydantic import BaseModel


class Department(BaseModel):
    name: str
    location: str
    opd_timings: str


class HospitalConfig(Document):
    """Single seeded document with public hospital info."""

    name: str
    address: str
    opd_timings: str
    emergency_contact: str
    departments: list[Department]

    class Settings:
        name = "hospital_config"
