"""Symptoms to a department. Routing only: this never names a condition or a treatment."""

from typing import Any

from pydantic import Field

from agents.intent_classifier import normalise
from models.hospital_config import HospitalConfig
from tools import Tool, ToolArgs, ToolContext, ToolResult, schema

DEFAULT_DEPARTMENT = "General Medicine"

# Checked in order: a child goes to Pediatrics whatever the symptom.
DEPARTMENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Pediatrics",
        (
            "child",
            "son",
            "daughter",
            "baby",
            "kid",
            "infant",
            "toddler",
            "bacche",
            "baccha",
            "bachche",
            "magu",
            "बच्चे",
            "बच्चा",
            "बच्ची",
            "बेटे",
            "बेटी",
            "ಮಗು",
            "ಮಕ್ಕಳ",
        ),
    ),
    (
        "Cardiology",
        (
            "chest",
            "heart",
            "palpitation",
            "seene",
            "chhati",
            "ede",
            "सीने",
            "छाती",
            "दिल",
            "ಎದೆ",
            "ಹೃದಯ",
        ),
    ),
    (
        "Orthopedics",
        (
            "knee",
            "bone",
            "joint",
            "back pain",
            "backache",
            "fracture",
            "ankle",
            "shoulder",
            "sprain",
            "ghutne",
            "haddi",
            "kamar",
            "monakalu",
            "घुटने",
            "हड्डी",
            "कमर",
            "जोड़",
            "ಮೊಣಕಾಲು",
            "ಮೂಳೆ",
            "ಬೆನ್ನು",
        ),
    ),
    (
        "Dermatology",
        (
            "skin",
            "rash",
            "itch",
            "acne",
            "pimple",
            "khujli",
            "twacha",
            "त्वचा",
            "खुजली",
            "दाने",
            "ಚರ್ಮ",
            "ತುರಿಕೆ",
        ),
    ),
    (
        "ENT",
        (
            "ear",
            "nose",
            "throat",
            "hearing",
            "sinus",
            "tonsil",
            "kaan",
            "naak",
            "gale",
            "gala",
            "kivi",
            "कान",
            "नाक",
            "गला",
            "गले",
            "ಕಿವಿ",
            "ಮೂಗು",
            "ಗಂಟಲು",
        ),
    ),
]

# Signs that should not wait for an appointment
URGENT_KEYWORDS = (
    "chest pain",
    "cant breathe",
    "can t breathe",
    "cannot breathe",
    "difficulty breathing",
    "shortness of breath",
    "unconscious",
    "fainted",
    "seizure",
    "heavy bleeding",
    "severe bleeding",
    "stroke",
    "saans",
    "सीने में दर्द",
    "सांस",
    "साँस",
    "बेहोश",
    "ಎದೆ ನೋವು",
    "ಉಸಿರಾಡ",
    "ede novu",
)


def _mentions(text: str, keyword: str) -> bool:
    if keyword.isascii():
        return f" {keyword} " in f" {text} " or f" {keyword}s " in f" {text} "
    return normalise(keyword) in text


def suggest_department(symptoms: str) -> tuple[str, bool]:
    """(department, urgent). Falls back to General Medicine when nothing more specific fits."""
    text = normalise(symptoms)
    urgent = any(_mentions(text, keyword) for keyword in URGENT_KEYWORDS)
    for department, keywords in DEPARTMENT_KEYWORDS:
        if any(_mentions(text, keyword) for keyword in keywords):
            return department, urgent
    return DEFAULT_DEPARTMENT, urgent


class SymptomArgs(ToolArgs):
    symptoms: str = Field(min_length=1, max_length=500)


async def symptom_to_department(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    parsed = SymptomArgs.model_validate(args)
    department, urgent = suggest_department(parsed.symptoms)
    config = await HospitalConfig.find_one()
    location = next(
        (d.location for d in (config.departments if config else []) if d.name == department), None
    )
    result: ToolResult = {"department": department, "location": location, "urgent": urgent}
    if urgent and config:
        result["emergency_contact"] = config.emergency_contact
    return result


TRIAGE_TOOLS: tuple[Tool, ...] = (
    Tool(
        "symptom_to_department",
        "Suggest which hospital department to visit for the symptoms the patient described. "
        "Returns a department only. It does not diagnose.",
        schema(
            {
                "symptoms": {
                    "type": "string",
                    "description": "The symptoms, in the patient's words.",
                }
            },
            ["symptoms"],
        ),
        symptom_to_department,
    ),
)
