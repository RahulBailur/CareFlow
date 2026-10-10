from beanie import Document

from models.appointment import Appointment
from models.conversation_turn import ConversationTurn
from models.doctor_status import DoctorStatus
from models.hospital_config import HospitalConfig
from models.schedule import Schedule
from models.user import User

DOCUMENT_MODELS: list[type[Document]] = [
    User,
    Appointment,
    Schedule,
    HospitalConfig,
    DoctorStatus,
    ConversationTurn,
]

__all__ = [
    "DOCUMENT_MODELS",
    "Appointment",
    "ConversationTurn",
    "DoctorStatus",
    "HospitalConfig",
    "Schedule",
    "User",
]
