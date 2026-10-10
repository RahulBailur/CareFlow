from agents.base import Agent
from agents.intent_data import Intent
from services.llm import Message
from tools import ToolContext, ToolResult, run_tool
from tools.triage_tools import DEFAULT_DEPARTMENT, TRIAGE_TOOLS, symptom_to_department

SYMPTOM_TO_DEPARTMENT = TRIAGE_TOOLS[0]

INSTRUCTIONS = """You suggest which department a patient should visit for the symptoms \
they describe. That is all you do.
- Name the department and location given in the facts below, and no other.
- Never say what the condition might be, what causes it, or what to take for it.
- If the facts mark the symptoms urgent, tell the patient to call the emergency number \
given there, or go to the emergency department, right away.
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


async def preload(context: ToolContext, text: str, history: list[Message]) -> dict[str, ToolResult]:
    """Route on what the patient just said; if that names no particular department,
    on what they said the turn before as well ("it also hurts when I breathe")."""
    result = await run_tool(SYMPTOM_TO_DEPARTMENT, context, {"symptoms": text or "unspecified"})
    earlier = [message.text for message in history if message.role == "user"]
    if result.get("department") == DEFAULT_DEPARTMENT and earlier:
        together = f"{earlier[-1]} {text}"[:500]
        result = await run_tool(SYMPTOM_TO_DEPARTMENT, context, {"symptoms": together})
    return {SYMPTOM_TO_DEPARTMENT.name: result}


TRIAGE_AGENT = Agent(Intent.TRIAGE, INSTRUCTIONS, TRIAGE_TOOLS, fallback, preload)
