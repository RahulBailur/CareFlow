from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import socketio
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import get_settings
from database import close_db, init_db
from frontend_app import mount_frontend
from rate_limit import limiter
from routes import appointments, auth, doctors, hospital
from sockets import sio


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_settings()  # fail fast on invalid configuration
    await init_db()
    yield
    close_db()


api = FastAPI(title="CareFlow", version="0.2.0", lifespan=lifespan)
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

api.include_router(auth.router)
api.include_router(appointments.router)
api.include_router(doctors.router)
api.include_router(hospital.router)


@api.get("/api/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Last, so its catch-all route never shadows an API route
mount_frontend(api, Path(get_settings().frontend_build_dir))

# The ASGI entry point: Socket.IO handles /socket.io, everything else goes to FastAPI
app = socketio.ASGIApp(sio, other_asgi_app=api)
