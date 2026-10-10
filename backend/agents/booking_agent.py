from agents.base import Agent
from agents.intent_data import Intent
from tools import ToolContext
from tools.booking_tools import BOOKING_TOOLS, list_my_appointments

INSTRUCTIONS = """You handle appointments: checking open slots, booking, rescheduling and \
cancelling.
- Call find_slots before offering or booking any time. Offer at most three times.
- To cancel or reschedule, call list_my_appointments first to get the appointment.
- Before booking, cancelling or rescheduling, say exactly what you are about to do and wait \
for the patient to confirm, unless they have already confirmed it in this conversation.
- After a change, confirm the doctor, date and time.
- If a tool returns an error, explain it plainly and offer another option."""


async def fallback(context: ToolContext, text: str) -> str:
    if not context.is_patient:
        return "Appointments can only be booked from a patient account."
    upcoming = (await list_my_appointments(context, {})).get("upcoming", [])
    how = "To book, change or cancel one, please use the Book page for now."
    if not upcoming:
        return f"You have no upcoming appointments. {how}"
    listed = "; ".join(f"{a['doctor_name']} on {a['local_time']}" for a in upcoming[:3])
    return f"Your upcoming appointments: {listed}. {how}"


BOOKING_AGENT = Agent(Intent.BOOKING, INSTRUCTIONS, BOOKING_TOOLS, fallback)
