"""One text turn: route it to a specialist agent, guard the reply, remember it.

Used by the chat endpoint now and by the cascaded voice pipeline later.
"""

import logging
import secrets
from dataclasses import dataclass, field

from agents.base import Agent, Progress, run_agent
from agents.booking_agent import BOOKING_AGENT
from agents.general_agent import GENERAL_AGENT
from agents.guardrails import guard_triage_reply
from agents.intent_classifier import (
    IntentClassifier,
    IntentResult,
    Language,
    get_classifier,
    keyword_scores,
    normalise,
)
from agents.intent_data import Intent
from agents.records_agent import RECORDS_AGENT
from agents.support_agent import SUPPORT_AGENT, facts_fingerprint, topic
from agents.triage_agent import TRIAGE_AGENT
from memory import conversation_memory as memory
from models.user import User
from services.llm import LLMProvider, LLMUnavailable, Message, get_llm
from services.redis_client import ResponseCache, get_response_cache, reply_key
from tools import ToolContext
from tools.confirmation import expire_older_proposals
from tools.support_tools import get_hospital_info

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
    language: Language  # what the patient wrote or said
    reply_language: Language  # what the reply is in: rule-based replies are English
    routed_by: Routing
    used_llm: bool
    disclaimer: str | None = None
    blocked: bool = False
    tools_called: list[str] = field(default_factory=list)
    cached: bool = False  # answered from the response cache, with no agent run
    # The tier that produced the reply: "cache", a model's name, or "rules"
    answered_by: str = "rules"


async def _cache_key(
    intent: Intent, routed_by: Routing, text: str, context: ToolContext
) -> str | None:
    """The cache key for this turn, or None if its reply must not be shared.

    Shared replies are only those about the hospital itself, to a question that says
    what it is asking (so the answer does not depend on the conversation), routed
    without the LLM. Anything about a patient is never cached.
    """
    if intent != Intent.SUPPORT or routed_by not in ("keywords", "embedding"):
        return None
    info = await get_hospital_info(context, {})
    if "error" in info or topic(text, info) is None:
        return None
    return reply_key(normalise(text), context.language, facts_fingerprint(info))


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
    progress: Progress | None = None,
    cache: ResponseCache | None = None,
) -> TurnResult:
    assert user.id is not None  # noqa: S101 — loaded from the database
    llm = llm or get_llm()
    local = (classifier or get_classifier()).classify(text)
    history = await memory.recent_messages(user.id, session_id)
    previous = await memory.last_intent(user.id, session_id)

    intent, routed_by = await _route(local, text, history, previous, llm)
    if progress:
        await progress("routed")
    context = ToolContext(
        user=user,
        language=local.language,
        session_id=session_id,
        turn_id=secrets.token_hex(8),
    )
    cache = cache or get_response_cache()
    key = await _cache_key(intent, routed_by, text, context) if cache.enabled else None
    if key and (hit := await cache.get(key)):
        result = TurnResult(
            reply=hit["reply"],
            intent=intent,
            language=local.language,
            reply_language=hit["reply_language"],
            routed_by=routed_by,
            used_llm=False,
            cached=True,
            answered_by="cache",
        )
        await expire_older_proposals(context)
        await memory.remember(user.id, session_id, text, result.reply, intent)
        return result

    answer = await run_agent(AGENTS[intent], context, text, history, llm, progress)
    await expire_older_proposals(context)

    result = TurnResult(
        reply=answer.text,
        intent=intent,
        language=local.language,
        reply_language=local.language if answer.used_llm else "en",
        routed_by=routed_by,
        used_llm=answer.used_llm or routed_by == "llm",
        tools_called=answer.tools_called,
        answered_by=answer.answered_by,
    )
    if intent == Intent.TRIAGE:
        guarded = guard_triage_reply(answer.text, result.reply_language)
        result.reply, result.disclaimer, result.blocked = (
            guarded.text,
            guarded.disclaimer,
            guarded.blocked,
        )

    # Only the LLM's wording is worth keeping: a rule-based reply costs nothing to redo
    if key and answer.used_llm:
        await cache.set(key, {"reply": result.reply, "reply_language": result.reply_language})
    await memory.remember(user.id, session_id, text, result.reply, intent, "; ".join(answer.notes))
    return result
