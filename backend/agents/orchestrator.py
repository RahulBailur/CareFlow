"""One text turn: route it to a specialist agent, guard the reply, remember it.

Used by the chat endpoint now and by the cascaded voice pipeline later.
"""

import logging
import secrets
from dataclasses import dataclass, field

from agents.base import Agent, run_agent
from agents.booking_agent import BOOKING_AGENT
from agents.general_agent import GENERAL_AGENT
from agents.guardrails import guard_triage_reply
from agents.intent_classifier import (
    IntentClassifier,
    IntentResult,
    Language,
    get_classifier,
    keyword_scores,
)
from agents.intent_data import Intent
from agents.records_agent import RECORDS_AGENT
from agents.support_agent import SUPPORT_AGENT
from agents.triage_agent import TRIAGE_AGENT
from memory import conversation_memory as memory
from models.user import User
from services.llm import LLMProvider, LLMUnavailable, Message, get_llm
from tools import ToolContext
from tools.confirmation import expire_older_proposals

logger = logging.getLogger(__name__)

AGENTS: dict[Intent, Agent] = {
    agent.intent: agent
    for agent in (BOOKING_AGENT, TRIAGE_AGENT, RECORDS_AGENT, SUPPORT_AGENT, GENERAL_AGENT)
}

ROUTER_PROMPT = """Classify the patient's latest message for a hospital assistant.
Answer with exactly one word from this list:
booking  - book, change, cancel or ask about an appointment or a doctor's availability
triage   - describes symptoms or asks which department or doctor to see for a problem
records  - asks about their own past visits, prescriptions or reports
support  - hospital timings, locations, address, parking, emergency contact
general  - greetings, thanks, or anything else"""

# How the turn's intent was decided, most local first
Routing = str  # "keywords" | "embedding" | "llm" | "previous" | "guess"


@dataclass
class TurnResult:
    reply: str
    intent: Intent
    language: Language
    routed_by: Routing
    used_llm: bool
    disclaimer: str | None = None
    blocked: bool = False
    tools_called: list[str] = field(default_factory=list)


async def _route(
    local: IntentResult,
    text: str,
    history: list[Message],
    previous: Intent | None,
    llm: LLMProvider,
) -> tuple[Intent, Routing]:
    if not local.needs_llm:
        return local.intent, local.source
    try:
        response = await llm.generate(ROUTER_PROMPT, [*history, Message("user", text)])
        words = response.text.strip().lower().split()
        if words and words[0].strip(".,:") in Intent._value2member_map_:
            return Intent(words[0].strip(".,:")), "llm"
    except LLMUnavailable as error:
        logger.warning("LLM unavailable for routing: %s", error)
    # No LLM verdict. A follow-up with no signal of its own ("what time was that again?")
    # stays with the agent that handled the turn before it.
    no_signal = not any(keyword_scores(text).values())
    if no_signal and previous is not None and previous != Intent.GENERAL:
        return previous, "previous"
    return local.intent, "guess"


async def run_text_turn(
    user: User,
    text: str,
    session_id: str,
    llm: LLMProvider | None = None,
    classifier: IntentClassifier | None = None,
) -> TurnResult:
    assert user.id is not None  # noqa: S101 — loaded from the database
    llm = llm or get_llm()
    local = (classifier or get_classifier()).classify(text)
    history = await memory.recent_messages(user.id, session_id)
    previous = await memory.last_intent(user.id, session_id)

    intent, routed_by = await _route(local, text, history, previous, llm)
    context = ToolContext(
        user=user,
        language=local.language,
        session_id=session_id,
        turn_id=secrets.token_hex(8),
    )
    answer = await run_agent(AGENTS[intent], context, text, history, llm)
    await expire_older_proposals(context)

    result = TurnResult(
        reply=answer.text,
        intent=intent,
        language=local.language,
        routed_by=routed_by,
        used_llm=answer.used_llm or routed_by == "llm",
        tools_called=answer.tools_called,
    )
    if intent == Intent.TRIAGE:
        # Rule-based replies are English, so the disclaimer matches the reply, not the question
        guarded = guard_triage_reply(answer.text, local.language if answer.used_llm else "en")
        result.reply, result.disclaimer, result.blocked = (
            guarded.text,
            guarded.disclaimer,
            guarded.blocked,
        )

    await memory.remember(user.id, session_id, text, result.reply, intent, "; ".join(answer.notes))
    return result
