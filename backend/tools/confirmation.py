"""Two-step confirmation for tools that change something.

The first call only records the proposal and tells the model to ask the patient. The same
call with the same arguments goes through in the *next* turn, and only that one. A model
therefore cannot book, cancel or move an appointment in the turn where the idea first came
up, however the patient or the prompt is worded.
"""

import json

from pydantic import BaseModel

from models.pending_action import PendingAction
from tools import ToolContext, ToolResult


def _scope(context: ToolContext) -> dict[str, object]:
    return {"user_id": context.user.id, "session_id": context.session_id}


async def confirm_or_propose(
    context: ToolContext, tool: str, args: BaseModel, summary: str
) -> ToolResult | None:
    """None when the patient has had a turn to confirm this exact action; else the proposal."""
    assert context.user.id is not None  # noqa: S101 — loaded from the database
    args_key = json.dumps(args.model_dump(mode="json"), sort_keys=True)
    pending = await PendingAction.find_one(_scope(context))
    if pending is not None:
        await pending.delete()
        if (
            pending.tool == tool
            and pending.args_key == args_key
            and pending.turn_id != context.turn_id
        ):
            return None
    await PendingAction(
        user_id=context.user.id,
        session_id=context.session_id,
        tool=tool,
        args_key=args_key,
        turn_id=context.turn_id,
    ).insert()
    return {
        "needs_confirmation": True,
        "not_done_yet": summary,
        "instruction": (
            "Nothing has changed yet. Tell the patient exactly what is about to happen and ask "
            "them to confirm. Only after they say yes, call this tool again with the same "
            "arguments."
        ),
        "repeat_with": {"tool": tool, "arguments": args.model_dump(mode="json")},
    }


async def expire_older_proposals(context: ToolContext) -> None:
    """End of a turn: a proposal the patient did not confirm straight away is dropped."""
    stale = {**_scope(context), "turn_id": {"$ne": context.turn_id}}
    await PendingAction.find(stale).delete()


async def drop_proposals(context: ToolContext) -> None:
    """Withdraw whatever is awaiting confirmation in this conversation."""
    await PendingAction.find(_scope(context)).delete()
