from agents.base import Agent
from agents.intent_data import Intent
from tools import ToolContext
from tools.triage_tools import TRIAGE_TOOLS, symptom_to_department

INSTRUCTIONS = """You suggest which department a patient should visit for the symptoms \
they describe. That is all you do.
- Always call symptom_to_department and name the department and location it returns.
- Never say what the condition might be, what causes it, or what to take for it.
- If the tool marks the symptoms urgent, tell the patient to call the emergency number it \
returns, or go to the emergency department, right away.
- Then offer to help book an appointment in that department."""


async def fallback(context: ToolContext, text: str) -> str:
    result = await symptom_to_department(context, {"symptoms": text or "unspecified"})
    where = f" ({result['location']})" if result.get("location") else ""
    department = f"the {result['department']} department{where}"
    reply = f"For these symptoms, {department} is the right place to start."
    if result.get("urgent"):
        contact = result.get("emergency_contact")
        call = (
            f"call the emergency number {contact}" if contact else "go to the emergency department"
        )
        reply = f"If this is severe or came on suddenly, please {call} right away. {reply}"
    return reply


TRIAGE_AGENT = Agent(Intent.TRIAGE, INSTRUCTIONS, TRIAGE_TOOLS, fallback)
