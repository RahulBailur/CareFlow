"""Tools the agents may call.

Who the patient is never comes from the model. Every handler receives a `ToolContext` built
from the verified session, and no tool declares a patient or user ID parameter. If a model
sends one anyway, it is dropped before the handler sees it.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from agents.intent_classifier import Language
from models.user import Role, User
from services.appointments import BookingError
from services.llm import ToolSpec

ToolResult = dict[str, Any]
Handler = Callable[["ToolContext", dict[str, Any]], Awaitable[ToolResult]]

PATIENTS_ONLY = {"error": "Only a logged-in patient can do this."}


@dataclass(frozen=True)
class ToolContext:
    user: User  # from the JWT, never from model output
    language: Language = "en"
    # Which conversation and which turn of it: used to hold changes for confirmation
    session_id: str = "no-session"
    turn_id: str = ""

    @property
    def is_patient(self) -> bool:
        return self.user.role == Role.PATIENT


class ToolArgs(BaseModel):
    """Base for tool argument models: anything the tool did not declare is discarded."""

    model_config = ConfigDict(extra="ignore")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


def schema(
    properties: dict[str, dict[str, Any]], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


async def run_tool(tool: Tool, context: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Run a tool. Refusals and bad arguments come back as data for the model to explain."""
    try:
        return await tool.handler(context, args)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        return {"error": f"Invalid arguments: {problems}"}
    except BookingError as error:
        return {"error": error.detail}


OUTCOME_KEYS = ("booked", "cancelled", "rescheduled")


def outcome_note(tool_name: str, result: ToolResult) -> str | None:
    """A one-line record of what a tool did or proposed, kept in conversation memory."""
    for key in OUTCOME_KEYS:
        if key in result:
            done = result[key]
            return f"{key} {done['doctor_name']} at {done['local_time']}"
    if result.get("needs_confirmation"):
        return f"awaiting the patient's yes for {tool_name} {result['repeat_with']['arguments']}"
    return None
