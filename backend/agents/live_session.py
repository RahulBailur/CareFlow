"""What a Gemini Live session is given: one prompt, every agent's tools, and their execution.

Pipeline A keeps a single speech-to-speech session for the whole conversation, so there is
no per-turn hand-off between specialist agents: the model holds all the tools at once and
the "routing" is this prompt. What does not change is where the tools run. Every call is
executed here on the server with the identity from the session, through the same functions
as the text and cascaded pipelines, including the confirm-first rule for booking changes.
"""

from typing import Any

from agents.guardrails import DISCLAIMERS
from agents.intent_data import Intent
from services.llm import Message, _gemini_schema
from time_utils import IST, now_utc
from tools import Tool, ToolContext, ToolResult, run_tool
from tools.booking_tools import BOOKING_TOOLS
from tools.records_tools import RECORDS_TOOLS
from tools.support_tools import SUPPORT_TOOLS
from tools.triage_tools import TRIAGE_TOOLS

LIVE_TOOLS: tuple[Tool, ...] = (*BOOKING_TOOLS, *TRIAGE_TOOLS, *RECORDS_TOOLS, *SUPPORT_TOOLS)
_BY_NAME = {tool.name: tool for tool in LIVE_TOOLS}

# Which agent's work a tool is: decides the intent label and whether a disclaimer is due
TOOL_INTENT: dict[str, Intent] = {
    **{tool.name: Intent.BOOKING for tool in BOOKING_TOOLS},
    **{tool.name: Intent.TRIAGE for tool in TRIAGE_TOOLS},
    **{tool.name: Intent.RECORDS for tool in RECORDS_TOOLS},
    **{tool.name: Intent.SUPPORT for tool in SUPPORT_TOOLS},
}

PROMPT = """You are CareBot, the voice assistant of a hospital outpatient department.
Today is {today} (Indian Standard Time).
Reply in the language the patient speaks: English, Hindi or Kannada. Keep every reply to one \
or two short sentences.
The patient is already logged in. Never ask for, accept or mention patient IDs.
Never diagnose a condition, recommend a medicine or give a dose. Clinical advice comes from \
the doctor only.
State only facts returned by your tools. Never invent doctors, slots, timings or records. \
Say times exactly as the tools give them in local_time.

Appointments:
- Call find_slots before offering or booking any time. Offer at most three times.
- To cancel or reschedule, call list_my_appointments first.
- book_slot, cancel_appointment and reschedule_appointment change nothing the first time you \
call them: they return needs_confirmation. Say what is about to happen and ask for a yes. \
When the patient agrees, call the tool again with the same arguments.
- Never say an appointment is booked, cancelled or moved unless the tool result says so.

Symptoms:
- Always call symptom_to_department and name only the department and location it returns.
- Never say what the condition might be, what causes it, or what to take for it.
- If the tool marks the symptoms urgent, tell the patient to call the emergency number it \
returns or go to the emergency department right away.
- End every reply about symptoms with this sentence, in the patient's language: \
"{disclaimer}"

Records: call get_my_visits and read a prescription note back as recorded, without \
explaining or adding to it. You can only ever see this patient's own records.

Hospital questions (timings, locations, emergency contact): answer only from \
get_hospital_info, and say so if the answer is not there."""


def live_setup(model: str, history: list[Message]) -> dict[str, Any]:
    """The first message of a Live session: model, prompt, tools, transcription on both sides."""
    today = now_utc().astimezone(IST)
    prompt = PROMPT.format(today=f"{today:%A %d %B %Y}", disclaimer=DISCLAIMERS["en"])
    if history:
        # A new session (after a time limit or a dropped connection) carries on the old one
        lines = "\n".join(
            f"{'Patient' if message.role == 'user' else 'CareBot'}: {message.text}"
            for message in history
        )
        prompt += f"\n\nThe conversation so far:\n{lines}"
    return {
        "setup": {
            "model": model if model.startswith("models/") else f"models/{model}",
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "systemInstruction": {"parts": [{"text": prompt}]},
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": _gemini_schema(tool.parameters),
                        }
                        for tool in LIVE_TOOLS
                    ]
                }
            ],
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
    }


async def run_live_tool(name: str, args: dict[str, Any], context: ToolContext) -> ToolResult:
    """Execute one tool call from the Live model, on the server, as the session's user."""
    tool = _BY_NAME.get(name)
    if tool is None:
        return {"error": f"{name} is not available."}
    return await run_tool(tool, context, args)
