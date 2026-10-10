import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import socketio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from agents.intent_classifier import get_classifier
from config import get_settings
from database import close_db, init_db
from frontend_app import mount_frontend
from rate_limit import limiter
from routes import appointments, auth, chat, doctors, hospital, voice_ws
from services.appointments import BookingError
from services.stt import get_stt
from services.tts import get_tts
from sockets import sio

logger = logging.getLogger(__name__)


def _warm_up() -> None:
    try:
        get_classifier()
        get_stt().warm_up()
        get_tts().warm_up()
    except Exception:
        logger.exception("Model warm-up failed; models will load on first use")
    else:
        logger.info("Models ready")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # fail fast on invalid configuration
    await init_db()
    # Load the speech and embedding models now, off the event loop, so the first
    # patient does not wait for them
    warm_up = (
        asyncio.create_task(asyncio.to_thread(_warm_up)) if settings.environment != "test" else None
    )
    yield
    if warm_up is not None:
        warm_up.cancel()
    close_db()


api = FastAPI(title="CareFlow", version="0.2.0", lifespan=lifespan)
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


@api.exception_handler(BookingError)
async def booking_error(request: Request, error: BookingError) -> JSONResponse:
    return JSONResponse({"detail": error.detail}, status_code=error.status_code)


api.include_router(auth.router)
api.include_router(appointments.router)
api.include_router(doctors.router)
api.include_router(hospital.router)
api.include_router(chat.router)
api.include_router(voice_ws.router)


@api.get("/api/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Last, so its catch-all route never shadows an API route
mount_frontend(api, Path(get_settings().frontend_build_dir))

# The ASGI entry point: Socket.IO handles /socket.io, everything else goes to FastAPI
app = socketio.ASGIApp(sio, other_asgi_app=api)
