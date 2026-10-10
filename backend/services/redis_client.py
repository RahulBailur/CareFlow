"""Redis: the response cache for questions whose answer is the same for everyone.

Only replies about the hospital itself are cached (timings, locations, contacts). Nothing
about a patient ever is. Keys are hashes, so a patient's wording is not stored either.
Redis being down is never an error for the patient: the turn just goes to the LLM.
"""

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)

KEY_PREFIX = "careflow:reply:v1"


def reply_key(question: str, language: str, facts: str) -> str:
    """One key per (question as normalised, language, version of the facts answered from).

    `facts` is a fingerprint of the hospital information, so editing it retires every
    cached reply that was written from the old version.
    """
    digest = hashlib.sha256(f"{language}\n{question}\n{facts}".encode()).hexdigest()[:40]
    return f"{KEY_PREFIX}:{digest}"


class ResponseCache:
    def __init__(self, client: Any, ttl_seconds: int) -> None:
        self._client, self._ttl = client, ttl_seconds

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def get(self, key: str) -> dict[str, Any] | None:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as error:  # noqa: BLE001 — a cache must never break a turn
            logger.warning("Response cache read failed: %s", type(error).__name__)
            return None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        if self._client is None:
            return
        try:
            await self._client.set(key, json.dumps(value, ensure_ascii=False), ex=self._ttl)
        except Exception as error:  # noqa: BLE001
            logger.warning("Response cache write failed: %s", type(error).__name__)


@lru_cache
def get_response_cache() -> ResponseCache:
    settings = get_settings()
    if not settings.redis_url:
        return ResponseCache(None, settings.response_cache_ttl_s)
    from redis.asyncio import Redis

    client = Redis.from_url(
        settings.redis_url, decode_responses=True, socket_timeout=0.5, socket_connect_timeout=0.5
    )
    return ResponseCache(client, settings.response_cache_ttl_s)
