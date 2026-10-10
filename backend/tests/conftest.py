import os
from collections.abc import AsyncIterator

# Must be set before any application module reads its settings
os.environ["ENVIRONMENT"] = "test"
os.environ["USE_MOCK_DB"] = "true"
os.environ["JWT_SECRET"] = "test-secret-that-is-at-least-32-bytes-long"
os.environ["BCRYPT_ROUNDS"] = "4"
os.environ["RATE_LIMIT_AUTH"] = "5/minute"
os.environ["SEED_DEFAULT_PASSWORD"] = "seed-password-1"
# Tests must behave the same whether or not a frontend build happens to exist locally
os.environ["FRONTEND_BUILD_DIR"] = "no-frontend-build-in-tests"
# Never a real model in tests, whatever the developer's own .env says
os.environ["LLM_PROVIDER"] = "mock"
os.environ["REDIS_URL"] = ""  # tests that want a cache build one on fakeredis
os.environ["GEMINI_API_KEY"] = ""
os.environ["INTENT_EMBEDDING_MODEL"] = ""
os.environ["STT_PROVIDER"] = "mock"
os.environ["TTS_PROVIDER"] = "mock"
os.environ["VAD_PROVIDER"] = "energy"
os.environ["VAD_SILENCE_MS"] = "200"
os.environ["RATE_LIMIT_VOICE_SESSIONS"] = "3/minute"
os.environ["RATE_LIMIT_CHAT"] = "1000/minute"

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from database import close_db, init_db  # noqa: E402
from main import app  # noqa: E402
from rate_limit import limiter  # noqa: E402

PATIENT = {
    "name": "Test Patient",
    "email": "patient@example.com",
    "phone": "9876543210",
    "password": "correct-horse-1",
}


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """HTTP client against the app, with a fresh in-memory database per test."""
    await init_db()
    limiter.reset()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
    close_db()
