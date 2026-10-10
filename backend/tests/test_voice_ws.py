"""The voice WebSocket against a real server on a local port, with stand-in speech models."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
import uvicorn
from httpx import AsyncClient

from auth_utils import create_access_token
from main import app
from models.user import User
from routes import voice_ws
from tests.helpers import make_user
from tests.test_chat import seed_hospital
from tests.voice_helpers import UTTERANCE, FakeSTT, FakeTTS, tone

TIMEOUT = 5.0


@pytest.fixture
async def ws_url(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    voice_ws.reset_rate_limit()
    monkeypatch.setattr(voice_ws, "get_stt", lambda: FakeSTT())
    monkeypatch.setattr(voice_ws, "get_tts", lambda: FakeTTS())
    monkeypatch.setattr(voice_ws, "AUTH_TIMEOUT_S", 0.3)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 — uvicorn exposes a flag, not an event
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"ws://127.0.0.1:{port}/ws/voice"
    server.should_exit = True
    await task


class Caller:
    def __init__(self, session: aiohttp.ClientSession, ws: aiohttp.ClientWebSocketResponse) -> None:
        self._session, self.ws = session, ws
        self.events: list[dict[str, Any]] = []
        self.audio: list[bytes] = []

    @classmethod
    async def connect(
        cls,
        url: str,
        user: User | None = None,
        token: str | None = None,
        session_id: str | None = None,
    ) -> "Caller":
        session = aiohttp.ClientSession()
        caller = cls(session, await session.ws_connect(url))
        if user is not None:
            token = create_access_token(user)
        if token is not None:
            auth = {"type": "auth", "token": token}
            await caller.ws.send_json(auth | ({"session_id": session_id} if session_id else {}))
        return caller

    async def until(self, kind: str, **match: Any) -> dict[str, Any]:
        """Read until an event of this type (and with these fields) arrives."""
        async with asyncio.timeout(TIMEOUT):
            while True:
                message = await self.ws.receive()
                if message.type == aiohttp.WSMsgType.BINARY:
                    self.audio.append(message.data)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    self.events.append(event)
                    if event["type"] == kind and all(event.get(k) == v for k, v in match.items()):
                        return event
                else:
                    raise ConnectionError(f"closed with code {self.ws.close_code}")

    async def closed_with(self) -> int | None:
        async with asyncio.timeout(TIMEOUT):
            while not self.ws.closed:
                message = await self.ws.receive()
                if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    break
        return self.ws.close_code

    async def hang_up(self) -> None:
        await self.ws.close()
        await self._session.close()


@pytest.mark.guardrail
async def test_a_connection_that_never_authenticates_is_closed(ws_url: str) -> None:
    caller = await Caller.connect(ws_url)

    assert await caller.closed_with() == voice_ws.UNAUTHORIZED
    await caller.hang_up()


@pytest.mark.guardrail
@pytest.mark.parametrize("token", ["not-a-jwt", ""])
async def test_a_bad_token_is_refused(ws_url: str, token: str) -> None:
    caller = await Caller.connect(ws_url, token=token)

    assert await caller.closed_with() == voice_ws.UNAUTHORIZED
    await caller.hang_up()


@pytest.mark.guardrail
async def test_audio_sent_before_authenticating_is_refused(ws_url: str) -> None:
    caller = await Caller.connect(ws_url)
    await caller.ws.send_bytes(UTTERANCE)

    assert await caller.closed_with() == voice_ws.UNAUTHORIZED
    await caller.hang_up()


async def test_the_live_pipeline_is_refused_until_it_exists(ws_url: str) -> None:
    caller = await Caller.connect(f"{ws_url}?pipeline=live", await make_user())

    assert "cascade" in (await caller.until("error"))["message"]
    assert await caller.closed_with() == voice_ws.UNSUPPORTED
    await caller.hang_up()


async def test_a_spoken_question_gets_a_spoken_answer(ws_url: str) -> None:
    await seed_hospital()
    caller = await Caller.connect(ws_url, await make_user())
    ready = await caller.until("ready")
    assert ready["sample_rate"] == 16_000 and len(ready["session_id"]) >= 8

    # Sent the way a browser does: small chunks, not aligned to VAD frames
    for start in range(0, len(UTTERANCE), 1000):
        await caller.ws.send_bytes(UTTERANCE[start : start + 1000])

    reply = await caller.until("reply")
    await caller.until("timing")
    assert "9 am to 1 pm" in reply["text"] and reply["intent"] == "support"
    assert caller.audio, "expected the reply as audio"
    states = [event["state"] for event in caller.events if event["type"] == "state"]
    assert states == ["IDLE", "LISTENING", "TRANSCRIBING", "THINKING", "SPEAKING"]

    await caller.ws.send_json({"type": "playback_done"})
    await caller.until("state", state="WAITING_FOR_NEXT_INPUT")
    await caller.hang_up()


async def test_speaking_over_the_reply_sends_stop_audio(ws_url: str) -> None:
    await seed_hospital()
    caller = await Caller.connect(ws_url, await make_user())
    await caller.until("ready")
    await caller.ws.send_bytes(UTTERANCE)
    await caller.until("audio_end")

    await caller.ws.send_bytes(tone(400))

    await caller.until("stop_audio")
    await caller.until("state", state="LISTENING")
    await caller.hang_up()


async def test_end_closes_the_session_cleanly(ws_url: str) -> None:
    caller = await Caller.connect(ws_url, await make_user())
    await caller.until("ready")

    await caller.ws.send_json({"type": "end"})

    await caller.until("state", state="END")
    await caller.hang_up()


async def test_junk_text_and_oversized_audio_are_ignored_not_fatal(ws_url: str) -> None:
    await seed_hospital()
    caller = await Caller.connect(ws_url, await make_user())
    await caller.until("ready")

    await caller.ws.send_str("not json")
    await caller.ws.send_str('{"type": "unknown"}')
    await caller.ws.send_bytes(b"\x00" * (voice_ws.MAX_AUDIO_MESSAGE_BYTES + 2))
    await caller.ws.send_bytes(UTTERANCE)

    assert (await caller.until("reply"))["intent"] == "support"
    await caller.hang_up()


@pytest.mark.guardrail
async def test_voice_sessions_are_rate_limited_per_user(ws_url: str) -> None:
    user, someone_else = await make_user(), await make_user()
    for _ in range(3):
        caller = await Caller.connect(ws_url, user)
        await caller.until("ready")
        await caller.hang_up()

    blocked = await Caller.connect(ws_url, user)
    assert await blocked.closed_with() == voice_ws.TOO_MANY
    await blocked.hang_up()

    other = await Caller.connect(ws_url, someone_else)
    await other.until("ready")
    await other.hang_up()


async def test_a_caller_hanging_up_mid_turn_does_not_break_the_server(
    ws_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_hospital()
    slow = FakeSTT()
    slow.gate = asyncio.Event()
    monkeypatch.setattr(voice_ws, "get_stt", lambda: slow)
    caller = await Caller.connect(ws_url, await make_user())
    await caller.until("ready")
    await caller.ws.send_bytes(UTTERANCE)
    await caller.until("state", state="TRANSCRIBING")

    await caller.hang_up()
    slow.gate.set()
    await asyncio.sleep(0.2)

    # The same server still answers the next caller
    monkeypatch.setattr(voice_ws, "get_stt", lambda: FakeSTT())
    again = await Caller.connect(ws_url, await make_user())
    await again.until("ready")
    await again.ws.send_bytes(UTTERANCE)
    assert (await again.until("reply"))["intent"] == "support"
    await again.hang_up()


async def test_voice_can_continue_a_text_conversation(ws_url: str) -> None:
    caller = await Caller.connect(ws_url, await make_user(), session_id="text-session-123")

    assert (await caller.until("ready"))["session_id"] == "text-session-123"
    await caller.hang_up()


@pytest.mark.parametrize("session_id", ["short", "../../etc", "x" * 65, "has space in it"])
async def test_a_malformed_session_id_is_replaced_not_trusted(ws_url: str, session_id: str) -> None:
    caller = await Caller.connect(ws_url, await make_user(), session_id=session_id)

    assert (await caller.until("ready"))["session_id"] != session_id
    await caller.hang_up()
