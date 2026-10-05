from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from models.hospital_config import Department, HospitalConfig

router = APIRouter(prefix="/api/hospital", tags=["hospital"])


class HospitalOut(BaseModel):
    name: str
    address: str
    opd_timings: str
    emergency_contact: str
    departments: list[Department]


@router.get("/config")
async def hospital_config() -> HospitalOut:
    """Public hospital info: no login needed."""
    config = await HospitalConfig.find_one()
    if config is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Hospital info has not been seeded")
    return HospitalOut(
        name=config.name,
        address=config.address,
        opd_timings=config.opd_timings,
        emergency_contact=config.emergency_contact,
        departments=config.departments,
    )
