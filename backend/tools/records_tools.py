from typing import Any

from pydantic import Field

from models.appointment import AppointmentStatus
from services import appointments as booking
from time_utils import as_utc, now_utc
from tools import PATIENTS_ONLY, Tool, ToolArgs, ToolContext, ToolResult, schema
from tools.booking_tools import local_time


class VisitArgs(ToolArgs):
    limit: int = Field(default=5, ge=1, le=20)


async def get_my_visits(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Past visits of the session's patient. There is no way to name another patient."""
    if not context.is_patient:
        return PATIENTS_ONLY
    parsed = VisitArgs.model_validate(args)
    now = now_utc()
    past = [
        {
            "doctor_name": doctor_name,
            "department": appointment.department,
            "local_time": local_time(appointment.slot_start),
            "status": appointment.status.value,
            "prescription": appointment.prescription,
        }
        for appointment, doctor_name in await booking.history(context.user)
        if as_utc(appointment.slot_start) <= now and appointment.status == AppointmentStatus.DONE
    ]
    return {"visits": past[: parsed.limit], "total_in_last_12_months": len(past)}


RECORDS_TOOLS: tuple[Tool, ...] = (
    Tool(
        "get_my_visits",
        "The logged-in patient's own completed visits from the last 12 months, newest first, "
        "with the prescription note recorded for each.",
        schema({"limit": {"type": "integer", "description": "How many visits, 1 to 20."}}),
        get_my_visits,
    ),
)
