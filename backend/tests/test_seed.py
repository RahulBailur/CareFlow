from httpx import AsyncClient

from models.appointment import Appointment
from models.hospital_config import HospitalConfig
from models.schedule import Schedule
from models.user import Role, User
from scripts.seed_data import DOCTORS, OPD_WEEKDAYS, PATIENTS, seed


async def test_seed_creates_the_synthetic_dataset(client: AsyncClient) -> None:
    created = await seed()

    assert await User.find(User.role == Role.DOCTOR).count() == len(DOCTORS)
    assert await User.find(User.role == Role.ADMIN).count() == 1
    assert await User.find(User.role == Role.PATIENT).count() == len(PATIENTS)
    assert await Schedule.count() == len(DOCTORS) * len(OPD_WEEKDAYS)
    assert await Appointment.count() == created["appointments"] == len(PATIENTS) * 2
    assert await HospitalConfig.count() == 1

    doctors = await User.find(User.role == Role.DOCTOR).to_list()
    assert all(d.department and d.medical_registration_number for d in doctors)


async def test_seed_is_idempotent(client: AsyncClient) -> None:
    await seed()
    users, appointments = await User.count(), await Appointment.count()

    second = await seed()

    assert second == {"users": 0, "schedules": 0, "appointments": 0, "hospital_config": 0}
    assert await User.count() == users
    assert await Appointment.count() == appointments
