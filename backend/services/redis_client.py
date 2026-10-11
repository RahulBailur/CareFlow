"""Redis: the booking workflow tracker, and the response cache for questions whose
answer is the same for everyone.

Only replies about the hospital itself are cached (timings, locations, contacts). Nothing
about a patient ever is. Keys are hashes, so a patient's wording is not stored either.
Redis being down is never an error for the patient: the turn just goes to the LLM.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
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


WORKFLOW_PREFIX = "careflow:workflow:v1"
WORKFLOW_EVENTS_KEPT = 30
WORKFLOW_TTL_S = 24 * 60 * 60

# The stages a booking request moves through. awaiting_confirmation is the confirm-first rule.
WORKFLOW_STAGES = (
    "received",
    "intent_classified",
    "slot_checked",
    "awaiting_confirmation",
    "confirmed",
    "failed",
)


class WorkflowTracker:
    """Where each booking request has got to, per conversation, queryable while it runs.

    Kept as a short Redis list of stage events per (user, session). Stages and times only:
    no wording, and nothing a patient said.
    """

    def __init__(self, client: Any, clock: Any = None) -> None:
        self._client = client
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @staticmethod
    def key(user_id: object, session_id: str) -> str:
        return f"{WORKFLOW_PREFIX}:{user_id}:{session_id}"

    async def advance(
        self, user_id: object, session_id: str, turn_id: str, stage: str, tool: str = ""
    ) -> None:
        if self._client is None:
            return
        if stage not in WORKFLOW_STAGES:
            raise ValueError(f"unknown workflow stage {stage}")
        event = {"stage": stage, "turn": turn_id, "at": self._clock().isoformat()}
        if tool:
            event["tool"] = tool
        key = self.key(user_id, session_id)
        try:
            await self._client.rpush(key, json.dumps(event))
            await self._client.ltrim(key, -WORKFLOW_EVENTS_KEPT, -1)
            await self._client.expire(key, WORKFLOW_TTL_S)
        except Exception as error:  # noqa: BLE001 — tracking must never break a booking
            logger.warning("Workflow tracker write failed: %s", type(error).__name__)

    async def events(self, user_id: object, session_id: str) -> list[dict[str, Any]]:
        if self._client is None:
            return []
        try:
            raw = await self._client.lrange(self.key(user_id, session_id), 0, -1)
        except Exception as error:  # noqa: BLE001
            logger.warning("Workflow tracker read failed: %s", type(error).__name__)
            return []
        return [json.loads(item) for item in raw]


def _client() -> Any:
    settings = get_settings()
    if not settings.redis_url:
        return None
    from redis.asyncio import Redis

    return Redis.from_url(
        settings.redis_url, decode_responses=True, socket_timeout=0.5, socket_connect_timeout=0.5
    )


@lru_cache
def get_workflow_tracker() -> WorkflowTracker:
    return WorkflowTracker(_client())


@lru_cache
def get_response_cache() -> ResponseCache:
    return ResponseCache(_client(), get_settings().response_cache_ttl_s)
