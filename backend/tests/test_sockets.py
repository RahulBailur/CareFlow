"""Socket.IO tests against a real server on a local port, with real Socket.IO clients."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
import socketio
import uvicorn
from httpx import AsyncClient

from auth_utils import create_access_token
from main import app
from models.user import User
from sockets import QUEUE_EVENT
from tests.helpers import auth, make_appointment, make_doctor, make_user, slot

pytestmark = pytest.mark.guardrail

TIMEOUT = 3.0
QUIET = 0.4  # how long to wait before deciding nothing was sent


@pytest.fixture
async def server_url(client: AsyncClient) -> AsyncIterator[str]:
    """The app served over real HTTP/WebSocket; shares the test's in-memory database."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 — uvicorn exposes a flag, not an event
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await task


class Listener:
    """A connected Socket.IO client that records every queue update it receives."""

    def __init__(self) -> None:
        self.client = socketio.AsyncClient(reconnection=False)
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.client.on(QUEUE_EVENT, self.events.put)

    async def connect(self, url: str, user: User) -> "Listener":
        await self.client.connect(
            url, auth={"token": create_access_token(user)}, transports=["websocket"]
        )
        return self

    async def next(self) -> dict[str, Any]:
        return await asyncio.wait_for(self.events.get(), TIMEOUT)

    async def received_nothing(self) -> bool:
        await asyncio.sleep(QUIET)
        return self.events.empty()

    async def close(self) -> None:
        await self.client.disconnect()


@pytest.mark.parametrize("auth_payload", [None, {}, {"token": "not-a-jwt"}, {"token": 12345}])
async def test_socket_without_a_valid_token_is_rejected(
    server_url: str, auth_payload: dict[str, Any] | None
) -> None:
    client = socketio.AsyncClient(reconnection=False)

    with pytest.raises(socketio.exceptions.ConnectionError):
        await client.connect(server_url, auth=auth_payload, transports=["websocket"])

    assert not client.connected


async def test_each_patient_receives_only_their_own_position(
    client: AsyncClient, server_url: str
) -> None:
    doctor = await make_doctor()
    first = await make_user(name="First Patient")
    second = await make_user(name="Second Patient")
    await make_appointment(first, doctor, slot(0, 0))
    await make_appointment(second, doctor, slot(0, 1))
    first_socket = await Listener().connect(server_url, first)
    second_socket = await Listener().connect(server_url, second)

    await client.post("/api/doctors/status", json={"delay_minutes": 15}, headers=auth(doctor))
    to_first, to_second = await first_socket.next(), await second_socket.next()

    assert to_first["patients_ahead"] == 0
    assert to_second["patients_ahead"] == 1
    assert to_first["delay_minutes"] == to_second["delay_minutes"] == 15
    # Nothing about the other patient leaks into either payload
    assert "entries" not in to_first and "entries" not in to_second
    assert "First Patient" not in json.dumps(to_second)
    assert str(first.id) not in json.dumps(to_second)
    assert "Second Patient" not in json.dumps(to_first)
    assert await first_socket.received_nothing() and await second_socket.received_nothing()

    await first_socket.close()
    await second_socket.close()


async def test_patient_not_in_the_queue_receives_nothing(
    client: AsyncClient, server_url: str
) -> None:
    doctor, waiting, outsider = await make_doctor(), await make_user(), await make_user()
    await make_appointment(waiting, doctor, slot(0, 0))
    outsider_socket = await Listener().connect(server_url, outsider)

    await client.post("/api/doctors/status", json={"delay_minutes": 10}, headers=auth(doctor))

    assert await outsider_socket.received_nothing()
    await outsider_socket.close()


async def test_doctor_receives_their_full_queue_and_other_doctors_do_not(
    client: AsyncClient, server_url: str
) -> None:
    doctor, other_doctor, patient = await make_doctor(), await make_doctor(), await make_user()
    appointment = await make_appointment(patient, doctor, slot(0, 0))
    doctor_socket = await Listener().connect(server_url, doctor)
    other_socket = await Listener().connect(server_url, other_doctor)

    await client.put(
        f"/api/appointments/{appointment.id}/status",
        json={"status": "in_consultation"},
        headers=auth(doctor),
    )
    update = await doctor_socket.next()

    assert [e["status"] for e in update["entries"]] == ["in_consultation"]
    assert update["entries"][0]["patient_name"] == patient.name
    assert await other_socket.received_nothing()

    await doctor_socket.close()
    await other_socket.close()


async def test_queue_moves_up_live_when_the_patient_ahead_is_seen(
    client: AsyncClient, server_url: str
) -> None:
    doctor, ahead, behind = await make_doctor(), await make_user(), await make_user()
    first = await make_appointment(ahead, doctor, slot(0, 0))
    await make_appointment(behind, doctor, slot(0, 1))
    socket = await Listener().connect(server_url, behind)
    url = f"/api/appointments/{first.id}/status"

    await client.put(url, json={"status": "in_consultation"}, headers=auth(doctor))
    while_consulting = await socket.next()
    await client.put(url, json={"status": "done"}, headers=auth(doctor))
    after_done = await socket.next()

    assert while_consulting["patients_ahead"] == 1
    assert after_done["patients_ahead"] == 0

    await socket.close()


async def test_patient_is_told_when_their_own_appointment_is_cancelled(
    client: AsyncClient, server_url: str
) -> None:
    doctor, patient = await make_doctor(), await make_user()
    booked = await client.post(
        "/api/appointments/book",
        json={"doctor_id": str(doctor.id), "slot_start": slot(1, 0).isoformat()},
        headers=auth(patient),
    )
    socket = await Listener().connect(server_url, patient)

    await client.put(
        f"/api/appointments/{booked.json()['id']}",
        json={"action": "cancel"},
        headers=auth(patient),
    )
    update = await socket.next()

    assert update["status"] == "cancelled"
    assert update["in_queue"] is False and update["patients_ahead"] is None

    await socket.close()
