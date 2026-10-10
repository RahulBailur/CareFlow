from typing import Any

from models.hospital_config import HospitalConfig
from tools import Tool, ToolContext, ToolResult, schema

NOT_SEEDED = {"error": "Hospital information is not available right now."}


async def get_hospital_info(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    config = await HospitalConfig.find_one()
    if config is None:
        return NOT_SEEDED
    return {
        "name": config.name,
        "address": config.address,
        "opd_timings": config.opd_timings,
        "emergency_contact": config.emergency_contact,
        "departments": [
            {"name": d.name, "location": d.location, "opd_timings": d.opd_timings}
            for d in config.departments
        ],
    }


SUPPORT_TOOLS: tuple[Tool, ...] = (
    Tool(
        "get_hospital_info",
        "Hospital name, address, OPD timings, emergency contact, and each department's "
        "location and timings.",
        schema({}),
        get_hospital_info,
    ),
)
