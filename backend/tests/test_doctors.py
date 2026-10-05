from datetime import timedelta

from httpx import AsyncClient

from models.doctor_status import DoctorStatus
from models.user import Role
from scripts.seed_data import seed
from tests.helpers import auth, make_appointment, make_doctor, make_user, slot
from time_utils import today_ist


async def test_schedule_lists_the_doctors_sessions(client: AsyncClient) -> None:
    doctor, patient = await make_doctor(), await make_user()

    response = await client.get(f"/api/doctors/{doctor.id}/schedule", headers=auth(patient))

    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert [s["weekday"] for s in sessions] == list(range(7))
    assert sessions[0] == {
        "weekday": 0,
        "start_time": "09:00",
        "end_time": "13:00",
        "slot_minutes": 15,
    }


async def test_schedule_of_a_non_doctor_is_404(client: AsyncClient) -> None:
    patient = await make_user()

    response = await client.get(f"/api/doctors/{patient.id}/schedule", headers=auth(patient))

    assert response.status_code == 404


async def test_delay_broadcast_shifts_the_patients_eta(client: AsyncClient) -> None:
    doctor, patient = await make_doctor(), await make_user()
    await make_appointment(patient, doctor, slot(0, 2))

    posted = await client.post(
        "/api/doctors/status",
        json={"delay_minutes": 20, "message": "Running late"},
        headers=auth(doctor),
    )
    view = await client.get(f"/api/appointments/queue/{doctor.id}", headers=auth(patient))

    assert posted.status_code == 200
    body = view.json()
    assert body["delay_minutes"] == 20
    assert body["delay_message"] == "Running late"
    assert body["eta"].startswith((slot(0, 2) + timedelta(minutes=20)).isoformat()[:16])


async def test_delay_can_be_cleared(client: AsyncClient) -> None:
    doctor = await make_doctor()
    await client.post("/api/doctors/status", json={"delay_minutes": 30}, headers=auth(doctor))

    await client.post("/api/doctors/status", json={"delay_minutes": 0}, headers=auth(doctor))
    view = await client.get(f"/api/appointments/queue/{doctor.id}", headers=auth(doctor))

    assert view.json()["delay_minutes"] == 0
    assert await DoctorStatus.count() == 1


async def test_yesterdays_delay_does_not_carry_over(client: AsyncClient) -> None:
    doctor = await make_doctor()
    assert doctor.id is not None
    yesterday = (today_ist() - timedelta(days=1)).isoformat()
    await DoctorStatus(doctor_id=doctor.id, day=yesterday, delay_minutes=45).insert()

    view = await client.get(f"/api/appointments/queue/{doctor.id}", headers=auth(doctor))

    assert view.json()["delay_minutes"] == 0


async def test_only_doctors_can_broadcast_a_delay(client: AsyncClient) -> None:
    patient, admin = await make_user(), await make_user(Role.ADMIN)

    for user in (patient, admin):
        response = await client.post(
            "/api/doctors/status", json={"delay_minutes": 10}, headers=auth(user)
        )
        assert response.status_code == 403


async def test_delay_must_be_within_bounds(client: AsyncClient) -> None:
    doctor = await make_doctor()

    for minutes in (-5, 500):
        response = await client.post(
            "/api/doctors/status", json={"delay_minutes": minutes}, headers=auth(doctor)
        )
        assert response.status_code == 422


async def test_hospital_config_is_public_once_seeded(client: AsyncClient) -> None:
    missing = await client.get("/api/hospital/config")
    await seed()
    found = await client.get("/api/hospital/config")

    assert missing.status_code == 404
    assert found.status_code == 200
    assert found.json()["name"] == "CareFlow Demo Hospital"
    assert len(found.json()["departments"]) == 6
