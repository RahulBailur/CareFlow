from agents.base import Agent
from agents.intent_data import Intent
from tools import ToolContext

INSTRUCTIONS = """You handle greetings, thanks and anything that is not about appointments, \
symptoms, records or hospital information.
- Be brief and friendly, and say what you can help with: booking or changing appointments, \
suggesting a department for symptoms, visit history, and hospital timings and locations.
- Politely decline anything outside that."""

CAPABILITIES = (
    "I can help you book, change or cancel an appointment, suggest which department to visit, "
    "show your visit history, and answer questions about hospital timings and locations."
)


async def fallback(context: ToolContext, text: str) -> str:
    return CAPABILITIES


GENERAL_AGENT = Agent(Intent.GENERAL, INSTRUCTIONS, (), fallback)
