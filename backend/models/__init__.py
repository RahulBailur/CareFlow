from beanie import Document

from models.appointment import Appointment
from models.hospital_config import HospitalConfig
from models.schedule import Schedule
from models.user import User

DOCUMENT_MODELS: list[type[Document]] = [User, Appointment, Schedule, HospitalConfig]

__all__ = ["DOCUMENT_MODELS", "Appointment", "HospitalConfig", "Schedule", "User"]
