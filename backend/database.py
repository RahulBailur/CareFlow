import logging
from typing import Any

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from config import get_settings
from models import DOCUMENT_MODELS

logger = logging.getLogger(__name__)
_client: Any = None


async def init_db() -> None:
    global _client
    settings = get_settings()
    database: Any
    if settings.use_mock_db:
        from mongomock_motor import AsyncMongoMockClient

        _client = AsyncMongoMockClient(tz_aware=True)
        database = _client["careflow_test"]
    else:
        _client = AsyncIOMotorClient(settings.mongo_uri, tz_aware=True)
        database = _client.get_default_database(default="careflow")
    await init_beanie(database=database, document_models=DOCUMENT_MODELS)
    if not settings.use_mock_db:
        # Open the connection now, so the first patient does not wait for the TLS handshake
        try:
            await _client.admin.command("ping")
        except Exception as error:  # noqa: BLE001 — the app still starts; requests will retry
            logger.warning("Database not reachable at start-up: %s", type(error).__name__)


def close_db() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
