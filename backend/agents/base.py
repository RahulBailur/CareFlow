"""The tool-calling loop every specialist agent shares."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agents.intent_classifier import Language
from agents.intent_data import Intent
from services.llm import LLMProvider, LLMUnavailable, Message
from time_utils import IST, now_utc
from tools import OUTCOME_KEYS, Tool, ToolContext, outcome_note, run_tool
from tools.confirmation import drop_proposals

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4

LANGUAGE_NAMES: dict[Language, str] = {"en": "English", "hi": "Hindi", "kn": "Kannada"}

COMMON_RULES = """You are CareBot, the assistant of a hospital outpatient department.
Today is {today} (Indian Standard Time).
Reply in {language}. Keep replies to one or two short sentences: they may be read aloud.
The patient is already logged in. Never ask for, accept or mention patient IDs.
Never diagnose a condition, recommend a medicine or give a dose. Clinical advice comes \
from the doctor only.
State only facts returned by your tools. Never invent doctors, slots, timings or records.
Say times exactly as the tools give them in local_time."""

Fallback = Callable[[ToolContext, str], Awaitable[str]]


@dataclass(frozen=True)
class Agent:
    intent: Intent
    instructions: str
    tools: tuple[Tool, ...]
    # Rule-based reply for when no LLM answer is available. Always in English.
    fallback: Fallback


@dataclass
class AgentReply:
    text: str
    used_llm: bool
    tools_called: list[str] = field(default_factory=list)
    # What the tools did or proposed this turn, for conversation memory
    notes: list[str] = field(default_factory=list)


def system_prompt(agent: Agent, language: Language) -> str:
    today = now_utc().astimezone(IST)
    rules = COMMON_RULES.format(today=f"{today:%A %d %B %Y}", language=LANGUAGE_NAMES[language])
    return f"{rules}\n\n{agent.instructions}"


async def run_agent(
    agent: Agent, context: ToolContext, text: str, history: list[Message], llm: LLMProvider
) -> AgentReply:
    """Let the LLM answer with this agent's tools; fall back to rules if it cannot."""
    by_name = {tool.name: tool for tool in agent.tools}
    specs = [tool.spec for tool in agent.tools]
    messages = [*history, Message("user", text)]
    called: list[str] = []
    notes: list[str] = []
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = await llm.generate(system_prompt(agent, context.language), messages, specs)
            if not response.tool_calls:
                if response.text.strip():
                    return AgentReply(response.text.strip(), True, called, notes)
                break
            messages.append(
                Message("assistant", response.text, response.tool_calls, raw=response.raw)
            )
            for call in response.tool_calls:
                tool = by_name.get(call.name)
                # A tool outside this agent's set is refused, whatever the model asks for
                result = (
                    await run_tool(tool, context, call.args)
                    if tool
                    else {"error": f"{call.name} is not available here."}
                )
                called.append(call.name)
                if note := outcome_note(call.name, result):
                    notes.append(note)
                messages.append(Message("tool", tool_name=call.name, tool_result=result))
    except LLMUnavailable as error:
        logger.warning("LLM unavailable for the %s agent: %s", agent.intent.value, error)

    # No final answer from the model. Changes already made are reported plainly, and a
    # proposal the patient never got to hear is withdrawn so a later "yes" cannot confirm it.
    done = [note for note in notes if note.split(" ", 1)[0] in OUTCOME_KEYS]
    if len(done) != len(notes):
        await drop_proposals(context)
    reply = await agent.fallback(context, text)
    if done:
        reply = f"Done: {'; '.join(done)}. {reply}"
    return AgentReply(reply, False, called, done)
