"""JWT-authenticated Socket.IO for live queue updates.

There is no subscribe event. On connect the server places each socket in a room for its own
user (and, for a doctor, their own queue), and only the server decides what is sent where.
A patient therefore has no way to ask for anyone else's position.
"""

from typing import Any

import socketio

from auth_utils import user_from_token
from models.appointment import Appointment
from models.user import Role, User
from services.queue import doctor_view, load_queue, patient_view

QUEUE_EVENT = "queue:update"

sio = socketio.AsyncServer(async_mode="asgi")


def user_room(user_id: object) -> str:
    return f"user:{user_id}"


def doctor_room(doctor_id: object) -> str:
    return f"doctor:{doctor_id}"


@sio.event
async def connect(sid: str, environ: dict[str, Any], auth: Any = None) -> None:
    token = auth.get("token") if isinstance(auth, dict) else None
    user = await user_from_token(token) if isinstance(token, str) else None
    if user is None:
        raise socketio.exceptions.ConnectionRefusedError("unauthorized")
    await sio.enter_room(sid, user_room(user.id))
    if user.role == Role.DOCTOR:
        await sio.enter_room(sid, doctor_room(user.id))


async def broadcast_queue(doctor: User, also: Appointment | None = None) -> None:
    """Push today's queue: the full list to the doctor, and to each patient only their own view.

    `also` is an appointment that just left the queue (e.g. cancelled), so its patient hears too.
    """
    snapshot = await load_queue(doctor)
    full = await doctor_view(snapshot)
    await sio.emit(QUEUE_EVENT, full.model_dump(mode="json"), room=doctor_room(doctor.id))
    for appointment in [*snapshot.appointments, *([also] if also else [])]:
        view = patient_view(snapshot, appointment)
        await sio.emit(
            QUEUE_EVENT, view.model_dump(mode="json"), room=user_room(appointment.patient_id)
        )
