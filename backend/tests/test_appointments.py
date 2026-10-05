import asyncio
from datetime import timedelta

import pytest
from httpx import AsyncClient

from models.appointment import Appointment, AppointmentStatus
from models.user import Role
from tests.helpers import auth, make_appointment, make_doctor, make_user, slot
from time_utils import today_ist

TOMORROW = (today_ist() + timedelta(days=1)).isoformat()


def _book_body(doctor_id: object, days_ahead: int = 1, index: int = 0) -> dict[str, str]:
    return {"doctor_id": str(doctor_id), "slot_start": slot(days_ahead, index).isoformat()}


# --- availability ---------------------------------------------------------------------------


async def test_availability_lists_open_slots_for_the_day(client: AsyncClient) -> None:
    doctor, patient = await make_doctor(), await make_user()

    response = await client.get(
        "/api/appointments/availability", params={"date": TOMORROW}, headers=auth(patient)
    )

    assert response.status_code == 200
    [entry] = response.json()
    assert entry["doctor_id"] == str(doctor.id)
    assert len(entry["slots"]) == 16  # 09:00–13:00 in 15-minute slots


async def test_availability_hides_booked_slots_and_shows_cancelled_ones_again(
    client: AsyncClient,
) -> None:
    doctor, patient = await make_doctor(), await make_user()
    booked = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(patient)
    )

    async def open_count() -> int:
        response = await client.get(
            "/api/appointments/availability", params={"date": TOMORROW}, headers=auth(patient)
        )
        return len(response.json()[0]["slots"])

    assert await open_count() == 15
    await client.put(
        f"/api/appointments/{booked.json()['id']}",
        json={"action": "cancel"},
        headers=auth(patient),
    )
    assert await open_count() == 16


async def test_availability_filters_by_department(client: AsyncClient) -> None:
    await make_doctor()
    patient = await make_user()

    response = await client.get(
        "/api/appointments/availability",
        params={"date": TOMORROW, "department": "Dermatology"},
        headers=auth(patient),
    )

    assert response.json() == []


async def test_availability_rejects_past_dates_and_requires_login(client: AsyncClient) -> None:
    patient = await make_user()
    yesterday = (today_ist() - timedelta(days=1)).isoformat()

    past = await client.get(
        "/api/appointments/availability", params={"date": yesterday}, headers=auth(patient)
    )
    anonymous = await client.get("/api/appointments/availability")

    assert past.status_code == 400
    assert anonymous.status_code == 401


# --- booking --------------------------------------------------------------------------------


async def test_patient_books_a_slot(client: AsyncClient) -> None:
    doctor, patient = await make_doctor("Dr. Test"), await make_user()

    response = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(patient)
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "booked"
    assert body["doctor_name"] == "Dr. Test"
    stored = await Appointment.get(body["id"])
    assert stored is not None and stored.patient_id == patient.id


async def test_booking_a_taken_slot_returns_409(client: AsyncClient) -> None:
    doctor = await make_doctor()
    first, second = await make_user(), await make_user()
    await client.post("/api/appointments/book", json=_book_body(doctor.id), headers=auth(first))

    response = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(second)
    )

    assert response.status_code == 409


@pytest.mark.guardrail
async def test_concurrent_bookings_for_one_slot_produce_exactly_one_appointment(
    client: AsyncClient,
) -> None:
    doctor = await make_doctor()
    patients = [await make_user() for _ in range(8)]

    responses = await asyncio.gather(
        *(
            client.post("/api/appointments/book", json=_book_body(doctor.id), headers=auth(p))
            for p in patients
        )
    )

    assert sorted(r.status_code for r in responses) == [201] + [409] * 7
    assert await Appointment.find(Appointment.doctor_id == doctor.id).count() == 1


@pytest.mark.guardrail
async def test_booking_ignores_a_patient_id_supplied_by_the_client(client: AsyncClient) -> None:
    doctor = await make_doctor()
    caller, victim = await make_user(), await make_user()

    response = await client.post(
        "/api/appointments/book",
        json={**_book_body(doctor.id), "patient_id": str(victim.id)},
        headers=auth(caller),
    )

    stored = await Appointment.get(response.json()["id"])
    assert stored is not None and stored.patient_id == caller.id


@pytest.mark.parametrize(
    "slot_start",
    [
        slot(-1).isoformat(),  # in the past
        (slot(1) + timedelta(minutes=7)).isoformat(),  # not on a slot boundary
        (slot(1) + timedelta(hours=8)).isoformat(),  # outside the session
    ],
)
async def test_booking_rejects_slots_that_are_not_bookable(
    client: AsyncClient, slot_start: str
) -> None:
    doctor, patient = await make_doctor(), await make_user()

    response = await client.post(
        "/api/appointments/book",
        json={"doctor_id": str(doctor.id), "slot_start": slot_start},
        headers=auth(patient),
    )

    assert response.status_code == 400
    assert await Appointment.count() == 0


async def test_only_patients_can_book(client: AsyncClient) -> None:
    doctor = await make_doctor()
    admin = await make_user(Role.ADMIN)

    as_doctor = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(doctor)
    )
    as_admin = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(admin)
    )

    assert as_doctor.status_code == as_admin.status_code == 403


async def test_booking_an_unknown_doctor_returns_404(client: AsyncClient) -> None:
    patient = await make_user()

    response = await client.post(
        "/api/appointments/book", json=_book_body(patient.id), headers=auth(patient)
    )

    assert response.status_code == 404


# --- reschedule / cancel --------------------------------------------------------------------


async def test_reschedule_moves_the_appointment_and_frees_the_old_slot(
    client: AsyncClient,
) -> None:
    doctor, patient, other = await make_doctor(), await make_user(), await make_user()
    booked = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id, index=0), headers=auth(patient)
    )

    moved = await client.put(
        f"/api/appointments/{booked.json()['id']}",
        json={"action": "reschedule", "slot_start": slot(1, 3).isoformat()},
        headers=auth(patient),
    )
    old_slot = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id, index=0), headers=auth(other)
    )

    assert moved.status_code == 200
    assert moved.json()["slot_start"].startswith(slot(1, 3).isoformat()[:16])
    assert old_slot.status_code == 201


async def test_reschedule_into_a_taken_slot_returns_409_and_keeps_the_original(
    client: AsyncClient,
) -> None:
    doctor, patient, other = await make_doctor(), await make_user(), await make_user()
    mine = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id, index=0), headers=auth(patient)
    )
    await client.post(
        "/api/appointments/book", json=_book_body(doctor.id, index=1), headers=auth(other)
    )

    response = await client.put(
        f"/api/appointments/{mine.json()['id']}",
        json={"action": "reschedule", "slot_start": slot(1, 1).isoformat()},
        headers=auth(patient),
    )

    assert response.status_code == 409
    stored = await Appointment.get(mine.json()["id"])
    assert stored is not None and stored.slot_start == slot(1, 0)


async def test_cancelled_slot_can_be_booked_by_someone_else(client: AsyncClient) -> None:
    doctor, patient, other = await make_doctor(), await make_user(), await make_user()
    booked = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(patient)
    )

    cancelled = await client.put(
        f"/api/appointments/{booked.json()['id']}",
        json={"action": "cancel"},
        headers=auth(patient),
    )
    rebooked = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(other)
    )

    assert cancelled.json()["status"] == "cancelled"
    assert rebooked.status_code == 201


@pytest.mark.guardrail
@pytest.mark.parametrize("action", ["cancel", "reschedule"])
async def test_a_patient_cannot_change_someone_elses_appointment(
    client: AsyncClient, action: str
) -> None:
    doctor, owner, intruder = await make_doctor(), await make_user(), await make_user()
    booked = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(owner)
    )

    response = await client.put(
        f"/api/appointments/{booked.json()['id']}",
        json={"action": action, "slot_start": slot(1, 5).isoformat()},
        headers=auth(intruder),
    )

    assert response.status_code == 404
    stored = await Appointment.get(booked.json()["id"])
    assert stored is not None
    assert stored.status == AppointmentStatus.BOOKED and stored.slot_start == slot(1, 0)


async def test_a_finished_appointment_cannot_be_cancelled(client: AsyncClient) -> None:
    doctor, patient = await make_doctor(), await make_user()
    done = await make_appointment(patient, doctor, slot(-1), AppointmentStatus.DONE)

    response = await client.put(
        f"/api/appointments/{done.id}", json={"action": "cancel"}, headers=auth(patient)
    )

    assert response.status_code == 409


async def test_reschedule_without_a_slot_is_rejected(client: AsyncClient) -> None:
    doctor, patient = await make_doctor(), await make_user()
    booked = await client.post(
        "/api/appointments/book", json=_book_body(doctor.id), headers=auth(patient)
    )

    response = await client.put(
        f"/api/appointments/{booked.json()['id']}",
        json={"action": "reschedule"},
        headers=auth(patient),
    )

    assert response.status_code == 422


# --- history --------------------------------------------------------------------------------


@pytest.mark.guardrail
async def test_history_shows_only_the_callers_own_visits(client: AsyncClient) -> None:
    doctor, patient, other = await make_doctor(), await make_user(), await make_user()
    await make_appointment(patient, doctor, slot(-10), AppointmentStatus.DONE, "Mine")
    await make_appointment(other, doctor, slot(-10, 1), AppointmentStatus.DONE, "Not mine")

    response = await client.get("/api/appointments/me", headers=auth(patient))

    assert [a["prescription"] for a in response.json()] == ["Mine"]


async def test_history_covers_twelve_months_newest_first(client: AsyncClient) -> None:
    doctor, patient = await make_doctor(), await make_user()
    await make_appointment(patient, doctor, slot(-400), AppointmentStatus.DONE, "Too old")
    await make_appointment(patient, doctor, slot(-200), AppointmentStatus.DONE, "Older")
    await make_appointment(patient, doctor, slot(-5), AppointmentStatus.DONE, "Recent")

    response = await client.get("/api/appointments/me", headers=auth(patient))

    assert [a["prescription"] for a in response.json()] == ["Recent", "Older"]


# --- queue ----------------------------------------------------------------------------------


@pytest.mark.guardrail
async def test_patient_sees_only_their_own_queue_position(client: AsyncClient) -> None:
    doctor = await make_doctor()
    first = await make_user(name="First Patient")
    second = await make_user(name="Second Patient")
    third = await make_user(name="Third Patient")
    await make_appointment(first, doctor, slot(0, 0))
    await make_appointment(second, doctor, slot(0, 1))
    await make_appointment(third, doctor, slot(0, 2))

    response = await client.get(f"/api/appointments/queue/{doctor.id}", headers=auth(third))

    assert response.status_code == 200
    body = response.json()
    assert body["patients_ahead"] == 2
    assert "entries" not in body
    assert "First Patient" not in response.text and "Second Patient" not in response.text
    assert str(first.id) not in response.text and str(second.id) not in response.text


async def test_queue_position_skips_finished_and_cancelled_patients(client: AsyncClient) -> None:
    doctor = await make_doctor()
    patients = [await make_user() for _ in range(4)]
    await make_appointment(patients[0], doctor, slot(0, 0), AppointmentStatus.DONE)
    await make_appointment(patients[1], doctor, slot(0, 1), AppointmentStatus.NO_SHOW)
    await make_appointment(patients[2], doctor, slot(0, 2), AppointmentStatus.IN_CONSULTATION)
    await make_appointment(patients[3], doctor, slot(0, 3))

    response = await client.get(f"/api/appointments/queue/{doctor.id}", headers=auth(patients[3]))

    assert response.json()["patients_ahead"] == 1


async def test_patient_without_an_appointment_today_gets_404(client: AsyncClient) -> None:
    doctor, waiting, outsider = await make_doctor(), await make_user(), await make_user()
    await make_appointment(waiting, doctor, slot(0, 0))

    response = await client.get(f"/api/appointments/queue/{doctor.id}", headers=auth(outsider))

    assert response.status_code == 404
    assert waiting.name not in response.text


async def test_doctor_sees_their_full_queue_in_slot_order(client: AsyncClient) -> None:
    doctor = await make_doctor()
    late, early = await make_user(name="Late"), await make_user(name="Early")
    await make_appointment(late, doctor, slot(0, 5))
    await make_appointment(early, doctor, slot(0, 1))

    response = await client.get(f"/api/appointments/queue/{doctor.id}", headers=auth(doctor))

    body = response.json()
    assert [e["patient_name"] for e in body["entries"]] == ["Early", "Late"]
    assert body["waiting"] == 2


async def test_a_doctor_cannot_view_another_doctors_queue(client: AsyncClient) -> None:
    doctor, other_doctor = await make_doctor(), await make_doctor()

    response = await client.get(f"/api/appointments/queue/{doctor.id}", headers=auth(other_doctor))

    assert response.status_code == 403


# --- doctor status updates ------------------------------------------------------------------


async def test_doctor_walks_an_appointment_through_consultation(client: AsyncClient) -> None:
    doctor, patient = await make_doctor(), await make_user()
    appointment = await make_appointment(patient, doctor, slot(0, 0))
    url = f"/api/appointments/{appointment.id}/status"

    started = await client.put(url, json={"status": "in_consultation"}, headers=auth(doctor))
    finished = await client.put(url, json={"status": "done"}, headers=auth(doctor))

    assert started.json()["status"] == "in_consultation"
    assert finished.json()["status"] == "done"


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (AppointmentStatus.BOOKED, "done"),  # must go through consultation
        (AppointmentStatus.DONE, "in_consultation"),
        (AppointmentStatus.CANCELLED, "no_show"),
    ],
)
async def test_invalid_status_transitions_are_refused(
    client: AsyncClient, start: AppointmentStatus, target: str
) -> None:
    doctor, patient = await make_doctor(), await make_user()
    appointment = await make_appointment(patient, doctor, slot(0, 0), start)

    response = await client.put(
        f"/api/appointments/{appointment.id}/status",
        json={"status": target},
        headers=auth(doctor),
    )

    assert response.status_code == 409


async def test_status_cannot_be_set_to_cancelled_or_booked(client: AsyncClient) -> None:
    doctor, patient = await make_doctor(), await make_user()
    appointment = await make_appointment(patient, doctor, slot(0, 0))

    for target in ("cancelled", "booked"):
        response = await client.put(
            f"/api/appointments/{appointment.id}/status",
            json={"status": target},
            headers=auth(doctor),
        )
        assert response.status_code == 422


async def test_only_the_appointments_own_doctor_can_update_status(client: AsyncClient) -> None:
    doctor, other_doctor, patient = await make_doctor(), await make_doctor(), await make_user()
    appointment = await make_appointment(patient, doctor, slot(0, 0))
    url = f"/api/appointments/{appointment.id}/status"
    body = {"status": "in_consultation"}

    as_other_doctor = await client.put(url, json=body, headers=auth(other_doctor))
    as_patient = await client.put(url, json=body, headers=auth(patient))

    assert as_other_doctor.status_code == 404
    assert as_patient.status_code == 403
    stored = await Appointment.get(appointment.id)
    assert stored is not None and stored.status == AppointmentStatus.BOOKED
