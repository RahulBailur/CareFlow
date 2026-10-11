import secrets
from typing import Annotated

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, StringConstraints

from agents.orchestrator import run_text_turn
from auth_utils import CurrentUser
from rate_limit import chat_limit, limiter
from services.redis_client import get_workflow_tracker

router = APIRouter(prefix="/api/chat", tags=["chat"])

SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{8,64}$")]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    # Omit on the first turn; send back the one returned to continue the conversation
    session_id: SessionId | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    language: str
    # Set on every triage reply; the UI must show it
    disclaimer: str | None
    blocked: bool
    routed_by: str
    used_llm: bool
    cached: bool
    answered_by: str
    tools_called: list[str]


class WorkflowEvent(BaseModel):
    stage: str
    turn: str
    at: str
    tool: str | None = None


class WorkflowOut(BaseModel):
    session_id: str
    events: list[WorkflowEvent]


@router.get("/{session_id}/workflow")
async def booking_workflow(user: CurrentUser, session_id: SessionId) -> WorkflowOut:
    """Where the booking requests in one of the caller's own conversations have got to."""
    events = await get_workflow_tracker().events(user.id, session_id)
    return WorkflowOut(session_id=session_id, events=[WorkflowEvent(**e) for e in events])


@router.post("")
@limiter.limit(chat_limit)
async def chat(request: Request, user: CurrentUser, body: ChatRequest) -> ChatResponse:
    """One CareBot text turn for the logged-in user."""
    session_id = body.session_id or secrets.token_urlsafe(16)
    result = await run_text_turn(user, body.message.strip() or body.message, session_id)
    return ChatResponse(
        reply=result.reply,
        session_id=session_id,
        intent=result.intent.value,
        language=result.language,
        disclaimer=result.disclaimer,
        blocked=result.blocked,
        routed_by=result.routed_by,
        used_llm=result.used_llm,
        cached=result.cached,
        answered_by=result.answered_by,
        tools_called=result.tools_called,
    )
