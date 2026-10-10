"""The tool-calling loop every specialist agent shares."""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agents.intent_classifier import Language
from agents.intent_data import Intent
from services.llm import LLMProvider, LLMUnavailable, Message
from time_utils import IST, now_utc
from tools import OUTCOME_KEYS, Tool, ToolContext, ToolResult, outcome_note, run_tool
from tools.confirmation import drop_proposals

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
RULES = "rules"  # the last tier: no model answered

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
# Told "routed", "tool" and "generating" as a turn moves along (drives the voice UI)
Progress = Callable[[str], Awaitable[None]]
# Looks up what an agent needs before the model is asked: {tool name: its result}
Preload = Callable[[ToolContext, str, list[Message]], Awaitable[dict[str, ToolResult]]]

FACTS_HEADER = (
    "Facts for this turn, already looked up from the hospital's systems. Answer only from "
    "these, and if they hold an error, tell the patient plainly:"
)


@dataclass(frozen=True)
class Agent:
    intent: Intent
    instructions: str
    tools: tuple[Tool, ...]
    # Rule-based reply for when no LLM answer is available. Always in English.
    fallback: Fallback
    # For an agent whose tools need no decision from the model (their arguments are the
    # patient's own words, or nothing): the server runs them first and hands over the
    # results, so the turn costs one model call instead of two. No tools are then offered.
    preload: Preload | None = None


@dataclass
class AgentReply:
    text: str
    used_llm: bool
    tools_called: list[str] = field(default_factory=list)
    # What the tools did or proposed this turn, for conversation memory
    notes: list[str] = field(default_factory=list)
    # Which tier produced the reply: a model's name, or "rules"
    answered_by: str = RULES


def system_prompt(
    agent: Agent, language: Language, facts: dict[str, ToolResult] | None = None
) -> str:
    today = now_utc().astimezone(IST)
    rules = COMMON_RULES.format(today=f"{today:%A %d %B %Y}", language=LANGUAGE_NAMES[language])
    prompt = f"{rules}\n\n{agent.instructions}"
    if facts is not None:
        prompt += f"\n\n{FACTS_HEADER}\n{json.dumps(facts, ensure_ascii=False, default=str)}"
    return prompt


async def run_agent(
    agent: Agent,
    context: ToolContext,
    text: str,
    history: list[Message],
    llm: LLMProvider,
    progress: Progress | None = None,
) -> AgentReply:
    """Let the LLM answer with this agent's tools; fall back to rules if it cannot."""
    by_name = {tool.name: tool for tool in agent.tools}
    specs = [tool.spec for tool in agent.tools]
    messages = [*history, Message("user", text)]
    called: list[str] = []
    notes: list[str] = []
    facts: dict[str, ToolResult] | None = None
    if agent.preload is not None:
        if progress:
            await progress("tool")
        facts = await agent.preload(context, text, history)
        called.extend(facts)
        by_name, specs = {}, []  # the facts are in hand: nothing left to call
        if progress:
            await progress("generating")
    prompt = system_prompt(agent, context.language, facts)
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = await llm.generate(prompt, messages, specs)
            if not response.tool_calls:
                if response.text.strip():
                    tier = response.provider or llm.name
                    return AgentReply(response.text.strip(), True, called, notes, tier)
                break
            messages.append(
                Message(
                    "assistant",
                    response.text,
                    response.tool_calls,
                    raw=response.raw,
                    provider=response.provider,
                )
            )
            if progress:
                await progress("tool")
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
            if progress:
                await progress("generating")
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
