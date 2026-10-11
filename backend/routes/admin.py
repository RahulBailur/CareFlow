from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from auth_utils import require_role
from models.user import Role, User
from services.analytics import department_load, turn_summary

router = APIRouter(prefix="/api/admin", tags=["admin"])

AdminUser = Annotated[User, Depends(require_role(Role.ADMIN))]


@router.get("/stats")
async def stats(admin: AdminUser) -> dict[str, Any]:
    """Department load today: doctors on duty, queue depth, appointments, delays."""
    departments = await department_load()
    keys = (
        "doctors_on_duty",
        "appointments_today",
        "waiting",
        "in_consultation",
        "done",
        "no_show",
    )
    totals = {key: sum(row[key] for row in departments) for key in keys}
    return {"departments": departments, "totals": totals}


@router.get("/analytics")
async def analytics(
    admin: AdminUser, days: Annotated[int, Query(ge=1, le=90)] = 7
) -> dict[str, Any]:
    """CareBot over the last `days`: latency per stage and pipeline, intent mix, which
    tier answered (the fallback rate), and LLM errors. Timings and labels only."""
    return await turn_summary(days)
