import hashlib
import json
from typing import Any

from agents.base import Agent
from agents.intent_classifier import normalise
from agents.intent_data import Intent
from tools import ToolContext
from tools.support_tools import SUPPORT_TOOLS, get_hospital_info

INSTRUCTIONS = """You answer questions about the hospital itself: OPD timings, where a \
department is, the address and the emergency contact.
- Always call get_hospital_info and answer only from what it returns.
- If the answer is not in it, say you do not have that information."""

_EMERGENCY = ("emergency", "ambulance", "helpline", "आपातकालीन", "इमरजेंसी", "एम्बुलेंस", "ತುರ್ತು")
_TIMINGS = ("timing", "time", "open", "close", "hours", "sunday", "khula", "band", "समय", "ಸಮಯ")
_ADDRESS = ("address", "reach", "directions", "pata", "पता", "ವಿಳಾಸ")


def topic(text: str, info: dict[str, Any]) -> str | None:
    """What a question about the hospital is asking for, if the question itself says so.

    None means it cannot be answered without the conversation around it ("where is it?").
    """
    asked = normalise(text)
    if any(word in asked for word in _EMERGENCY):
        return "emergency"
    for department in info["departments"]:
        # Whole words only: "ENT" must not match inside "department"
        if f" {department['name'].lower()} " in f" {asked} ":
            return f"department:{department['name']}"
    if any(word in asked for word in _ADDRESS):
        return "address"
    if any(word in asked for word in _TIMINGS):
        return "timings"
    return None


def facts_fingerprint(info: dict[str, Any]) -> str:
    """Changes whenever the hospital information does."""
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]


async def fallback(context: ToolContext, text: str) -> str:
    """The rule-based knowledge base: answers straight from the seeded hospital config."""
    info = await get_hospital_info(context, {})
    if "error" in info:
        return str(info["error"])
    asked_for = topic(text, info)
    if asked_for == "emergency":
        return f"The emergency contact number is {info['emergency_contact']}."
    if asked_for and asked_for.startswith("department:"):
        department = next(d for d in info["departments"] if f"department:{d['name']}" == asked_for)
        return (
            f"{department['name']} is at {department['location']}. "
            f"Its OPD timings are {department['opd_timings']}."
        )
    if asked_for == "address":
        return f"{info['name']} is at {info['address']}."
    if asked_for == "timings":
        return f"OPD timings are {info['opd_timings']}."
    return (
        f"{info['name']}: OPD timings are {info['opd_timings']}, and the emergency contact "
        f"is {info['emergency_contact']}."
    )


SUPPORT_AGENT = Agent(Intent.SUPPORT, INSTRUCTIONS, SUPPORT_TOOLS, fallback)
