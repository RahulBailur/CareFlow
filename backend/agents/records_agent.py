from agents.base import Agent
from agents.intent_data import Intent
from tools import ToolContext
from tools.records_tools import RECORDS_TOOLS, get_my_visits

INSTRUCTIONS = """You read back the logged-in patient's own visit history and the \
prescription notes recorded on those visits.
- Always call get_my_visits. You can only ever see this patient's records.
- Read a prescription note back as it was recorded. Do not explain, change or add to it.
- If asked about anyone else's records, say you can only show the patient their own."""


async def fallback(context: ToolContext, text: str) -> str:
    if not context.is_patient:
        return "Visit history is only available from a patient account."
    visits = (await get_my_visits(context, {"limit": 1})).get("visits", [])
    if not visits:
        return "You have no visits recorded in the last 12 months."
    last = visits[0]
    doctor = f"{last['doctor_name']} ({last['department']})"
    reply = f"Your last visit was with {doctor} on {last['local_time']}."
    if last.get("prescription"):
        reply += f" The note recorded was: {last['prescription'].rstrip('.')}."
    return f"{reply} Your full history is on the Visit history page."


RECORDS_AGENT = Agent(Intent.RECORDS, INSTRUCTIONS, RECORDS_TOOLS, fallback)
