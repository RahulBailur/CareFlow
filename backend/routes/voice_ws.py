"""CareBot voice over a WebSocket: /ws/voice?pipeline=cascade

Client -> server
  first message, text:  {"type": "auth", "token": "<JWT>", "session_id": "<optional>"}
                        (the token is never in the URL)
  binary:               microphone audio, PCM16 mono 16 kHz little-endian, any chunk size
  text:                 {"type": "playback_done"} | {"type": "end"}

Server -> client
  text:    ready, state, transcript, reply, audio_start, audio_end, audio_unavailable,
           stop_audio (barge-in: drop whatever is still queued to play), timing, error
  binary:  the spoken reply, PCM16 mono at the sample rate given in audio_start
"""

import asyncio
import contextlib
import json
import logging
import re
import secrets
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from auth_utils import user_from_token
from config import get_settings
from services.stt import get_stt
from services.tts import get_tts
from voice.audio_utils import SAMPLE_RATE
from voice.cascade_pipeline import VoiceSession
from voice.vad import Probability, SileroVAD, VADConfig, energy_vad

logger = logging.getLogger(__name__)
router = APIRouter()

AUTH_TIMEOUT_S = 5.0
MAX_SESSION_S = 10 * 60
MAX_AUDIO_MESSAGE_BYTES = 64 * 1024  # two seconds of 16 kHz PCM16

# WebSocket close codes in the application range
UNAUTHORIZED, UNSUPPORTED, TOO_MANY = 4401, 4400, 4429

SESSION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")  # the same shape /api/chat accepts

_recent_sessions: dict[str, deque[float]] = defaultdict(deque)


def _allow_session(user_id: str) -> bool:
    """`RATE_LIMIT_VOICE_SESSIONS` per user, e.g. "5/minute". One worker, so in memory."""
    limit, _, period = get_settings().rate_limit_voice_sessions.partition("/")
    window = {"second": 1, "minute": 60, "hour": 3600}.get(period, 60)
    now, started = time.monotonic(), _recent_sessions[user_id]
    while started and now - started[0] > window:
        started.popleft()
    if len(started) >= int(limit):
        return False
    started.append(now)
    return True


def reset_rate_limit() -> None:
    _recent_sessions.clear()


def _vad() -> Probability:
    return SileroVAD() if get_settings().vad_provider == "silero" else energy_vad()


async def _authenticate(websocket: WebSocket) -> tuple[Any, str | None]:
    """(user, session to continue). The session ID lets voice carry on a text conversation;
    memory is keyed by user as well, so naming someone else's session opens nothing."""
    try:
        message = json.loads(await asyncio.wait_for(websocket.receive_text(), AUTH_TIMEOUT_S))
        token = message.get("token") if message.get("type") == "auth" else None
        session_id = message.get("session_id")
    except (TimeoutError, ValueError, AttributeError, KeyError, RuntimeError, WebSocketDisconnect):
        return None, None
    user = await user_from_token(token) if isinstance(token, str) else None
    valid = isinstance(session_id, str) and SESSION_ID.fullmatch(session_id)
    return user, session_id if valid else None


@router.websocket("/ws/voice")
async def voice(websocket: WebSocket, pipeline: str = "cascade") -> None:
    await websocket.accept()
    user, continued_session = await _authenticate(websocket)
    if user is None:
        return await websocket.close(UNAUTHORIZED, "unauthorized")
    if pipeline != "cascade":
        await websocket.send_json({"type": "error", "message": "Only pipeline=cascade exists yet."})
        return await websocket.close(UNSUPPORTED, "unsupported pipeline")
    if not _allow_session(str(user.id)):
        return await websocket.close(TOO_MANY, "too many voice sessions")

    settings = get_settings()
    send_lock = asyncio.Lock()  # the turn task and the receive loop both write

    # A caller who has hung up is not an error: whatever was still on its way is dropped
    async def send_json(message: dict[str, Any]) -> None:
        async with send_lock:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_json(message)

    async def send_audio(pcm16: bytes) -> None:
        async with send_lock:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_bytes(pcm16)

    session_id = continued_session or secrets.token_urlsafe(16)
    session = VoiceSession(
        user,
        session_id,
        send_json,
        send_audio,
        stt=get_stt(),
        tts=get_tts(),
        probability=_vad(),
        vad_config=VADConfig(silence_ms=settings.vad_silence_ms),
        chunking=settings.tts_chunking,
    )
    await send_json({"type": "ready", "session_id": session_id, "sample_rate": SAMPLE_RATE})
    await session.start()
    deadline = time.monotonic() + MAX_SESSION_S
    try:
        while time.monotonic() < deadline:
            message = await asyncio.wait_for(websocket.receive(), deadline - time.monotonic())
            if message["type"] == "websocket.disconnect":
                break
            if (audio := message.get("bytes")) is not None:
                if len(audio) <= MAX_AUDIO_MESSAGE_BYTES:
                    await session.receive_audio(audio)
            elif (text := message.get("text")) is not None:
                kind = json.loads(text).get("type") if text.startswith("{") else None
                if kind == "playback_done":
                    await session.playback_done()
                elif kind == "end":
                    break
    except (WebSocketDisconnect, TimeoutError, RuntimeError, ValueError):
        pass
    finally:
        await session.close()
        await send_json({"type": "state", "state": "END"})
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()
