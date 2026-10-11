"""Pipeline A: the browser's audio bridged to a Gemini Live speech-to-speech session.

The model hears the patient and answers in audio directly, so there is no STT, LLM and TTS
to wait for one after another. What stays on this server, turn by turn:

  * tools run here, as the session's user (see agents/live_session.py);
  * the model's own transcript of what it is saying is checked as it arrives, and a reply
    that diagnoses or prescribes is cut off and replaced;
  * what was said is remembered, so a new session can carry the conversation on;
  * time to first audio is measured the same way as in Pipeline B, with a local VAD.

The client sees exactly the events Pipeline B sends, so the UI does not care which runs.
"""

import asyncio
import base64
import contextlib
import json
import logging
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import aiohttp

from agents.guardrails import DISCLAIMERS, SAFE_FALLBACKS, find_violations
from agents.intent_classifier import IntentClassifier, detect_language, get_classifier
from agents.intent_data import Intent
from agents.live_session import TOOL_INTENT, live_setup, run_live_tool
from config import get_settings
from memory import conversation_memory as memory
from models.turn_analytics import TurnAnalytics
from models.user import User
from services.tts import TTSProvider, TTSUnavailable
from tools import ToolContext, outcome_note
from tools.confirmation import expire_older_proposals
from voice.audio_utils import SAMPLE_RATE, Framer
from voice.cascade_pipeline import SendAudio, SendJson, VoiceState
from voice.timing import TurnTimer
from voice.vad import Probability, SpeechDetector, VADConfig

logger = logging.getLogger(__name__)

PIPELINE = "live"
LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
CONNECT_TIMEOUT_S = 10.0
MAX_RECONNECTS = 3
OUTPUT_SAMPLE_RATE = 24_000
INPUT_MIME = f"audio/pcm;rate={SAMPLE_RATE}"
# Reading a prescription note back is the one place medicine names belong in a reply
UNFILTERED_TOOLS = frozenset({"get_my_visits"})


class LiveUnavailable(Exception):
    """No Live session could be opened or kept. The caller falls back to Pipeline B."""


class LiveConnection(Protocol):
    async def send(self, message: dict[str, Any]) -> None: ...

    async def receive(self) -> dict[str, Any] | None:
        """The next message, or None once the connection has closed."""
        ...

    async def close(self) -> None: ...


Connect = Callable[[dict[str, Any]], Awaitable[LiveConnection]]


class GeminiLiveConnection:
    def __init__(self, http: aiohttp.ClientSession, ws: aiohttp.ClientWebSocketResponse) -> None:
        self._http, self._ws = http, ws

    async def send(self, message: dict[str, Any]) -> None:
        await self._ws.send_json(message)

    async def receive(self) -> dict[str, Any] | None:
        message = await self._ws.receive()
        if message.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
            return None
        data: dict[str, Any] = json.loads(message.data)
        return data

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._ws.close()
        await self._http.close()


async def connect_gemini_live(setup: dict[str, Any]) -> LiveConnection:
    """Open a Live session and wait for it to accept the setup."""
    api_key = get_settings().gemini_api_key
    if not api_key or not setup["setup"]["model"].removeprefix("models/"):
        raise LiveUnavailable("GEMINI_API_KEY and GEMINI_LIVE_MODEL must both be set")
    http = aiohttp.ClientSession()
    try:
        async with asyncio.timeout(CONNECT_TIMEOUT_S):
            # The key travels in a header, never in the URL
            ws = await http.ws_connect(
                LIVE_URL, headers={"x-goog-api-key": api_key}, max_msg_size=0
            )
            connection = GeminiLiveConnection(http, ws)
            await connection.send(setup)
            first = await connection.receive()
    except (TimeoutError, aiohttp.ClientError, OSError) as error:
        await http.close()
        raise LiveUnavailable(f"could not open a Live session: {type(error).__name__}") from error
    if first is None or "setupComplete" not in first:
        code = ws.close_code
        await connection.close()
        raise LiveUnavailable(f"the Live session was refused (close code {code})")
    return connection


def audio_message(pcm16: bytes) -> dict[str, Any]:
    data = base64.b64encode(pcm16).decode()
    return {"realtimeInput": {"audio": {"mimeType": INPUT_MIME, "data": data}}}


@dataclass
class _Turn:
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    heard: str = ""  # the model's transcript of the patient
    said: str = ""  # the model's transcript of itself
    tools: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    transcript_sent: bool = False
    audio_sent: float = 0.0  # seconds forwarded to the client
    blocked: bool = False
    # A tool was just answered and the model has not spoken since: its "turn complete" at
    # that point closes the tool call, not the reply
    awaiting_reply: bool = False
    timer: TurnTimer | None = None
    voice: str = ""


class LiveSession:
    def __init__(
        self,
        user: User,
        session_id: str,
        send_json: SendJson,
        send_audio: SendAudio,
        *,
        connect: Connect,
        model: str,
        probability: Probability,
        tts: TTSProvider,
        vad_config: VADConfig | None = None,
        hold_ms: int = 300,
        classifier: IntentClassifier | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._user, self._session_id = user, session_id
        self._send_json, self._send_audio = send_json, send_audio
        self._connect, self._model, self._tts = connect, model, tts
        self._hold_s, self._clock = hold_ms / 1000, clock
        self._classifier = classifier
        self._detector = SpeechDetector(probability, vad_config, clock)
        self._framer = Framer()
        self._connection: LiveConnection | None = None
        self._pump: asyncio.Task[None] | None = None
        self._flusher: asyncio.Task[None] | None = None
        self._held: deque[tuple[float, bytes]] = deque()
        self._release_lock = asyncio.Lock()
        self._turn = _Turn()
        self._speech_end = 0.0
        self._reconnects = 0
        self._go_away = False
        self._closing = False
        self.state = VoiceState.IDLE
        # Set when the Live session is lost for good: the caller switches to Pipeline B
        self.failed = asyncio.Event()
        self.failure = ""

    # --- lifecycle ----------------------------------------------------------------------

    async def _open(self) -> None:
        assert self._user.id is not None  # noqa: S101 — loaded from the database
        history = await memory.recent_messages(self._user.id, self._session_id)
        self._connection = await self._connect(live_setup(self._model, history))

    async def connect(self) -> None:
        """Open the Live session. Raises LiveUnavailable if it cannot be opened."""
        await self._open()

    async def start(self) -> None:
        self._pump = asyncio.create_task(self._run())
        self._flusher = asyncio.create_task(self._flush_held())
        # Sent even though nothing changed: it tells the client the session is listening
        await self._send_json({"type": "state", "state": VoiceState.IDLE.value})

    async def close(self) -> None:
        self._closing = True
        for task in (self._pump, self._flusher):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        self.state = VoiceState.END

    async def _set_state(self, state: VoiceState) -> None:
        if state != self.state:
            self.state = state
            await self._send_json({"type": "state", "state": state.value})

    def _fail(self, reason: str) -> None:
        logger.warning("Live pipeline lost: %s", reason)
        self.failure = reason
        self.failed.set()

    # --- input --------------------------------------------------------------------------

    async def receive_audio(self, data: bytes) -> None:
        """Microphone audio: forwarded to the model, and watched locally for timing."""
        if self._connection is not None:
            try:
                await self._connection.send(audio_message(data))
            except Exception as error:  # noqa: BLE001 — the reader loop decides what happens next
                logger.warning("Could not forward audio to the Live session: %s", error)
        for frame in self._framer.push(data):
            event = self._detector.feed(frame)
            if event is None:
                continue
            if event.kind == "start":
                self._speech_end = 0.0
                # Mid-reply, the model decides whether this is an interruption. Once the
                # reply has been handed over in full, it is simply the next question.
                mid_reply = self.state == VoiceState.SPEAKING and (
                    self._turn.audio_sent > 0 or bool(self._held)
                )
                if not mid_reply:
                    await self._set_state(VoiceState.LISTENING)
            else:
                self._speech_end = event.last_speech_at
                if self.state == VoiceState.LISTENING:
                    await self._set_state(VoiceState.THINKING)

    async def playback_done(self) -> None:
        if self.state == VoiceState.SPEAKING and self._turn.audio_sent == 0:
            await self._set_state(VoiceState.WAITING_FOR_NEXT_INPUT)

    # --- the model's side ---------------------------------------------------------------

    async def _run(self) -> None:
        while not self._closing:
            connection = self._connection
            message = await connection.receive() if connection is not None else None
            if message is None:
                if self._closing or not await self._reconnect("the connection closed"):
                    return
                continue
            try:
                await self._handle(message)
            except Exception:
                logger.exception("Live message could not be handled")

    async def _reconnect(self, reason: str) -> bool:
        """A session ended (time limit, dropped socket). Open another; memory carries on."""
        self._reconnects += 1
        if self._reconnects > MAX_RECONNECTS:
            self._fail(f"{reason}; too many reconnects")
            return False
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        self._held.clear()
        self._turn = _Turn()
        try:
            await self._open()
        except LiveUnavailable as error:
            self._fail(f"{reason}; {error}")
            return False
        self._go_away = False
        logger.info("Live session reopened after: %s", reason)
        return True

    def _context(self) -> ToolContext:
        language = detect_language(self._turn.heard)
        return ToolContext(self._user, language, self._session_id, self._turn.id)

    async def _handle(self, message: dict[str, Any]) -> None:
        turn = self._turn
        if "toolCall" in message:
            await self._send_transcript(turn)
            await self._set_state(VoiceState.TOOL_EXECUTION)
            responses = []
            for call in message["toolCall"].get("functionCalls", []):
                name, args = call.get("name", ""), dict(call.get("args") or {})
                result = await run_live_tool(name, args, self._context())
                turn.tools.append(name)
                if note := outcome_note(name, result):
                    turn.notes.append(note)
                responses.append({"id": call.get("id"), "name": name, "response": result})
            turn.awaiting_reply = True
            if self._connection is not None:
                await self._connection.send({"toolResponse": {"functionResponses": responses}})
            return
        if "goAway" in message:
            self._go_away = True  # the session is about to end: reopen after this turn
            return
        content = message.get("serverContent")
        if not content:
            return

        if heard := content.get("inputTranscription", {}).get("text"):
            turn.heard += heard
        for part in content.get("modelTurn", {}).get("parts", []):
            if "inlineData" in part:
                turn.awaiting_reply = False
                if not turn.blocked:
                    audio = base64.b64decode(part["inlineData"]["data"])
                    self._held.append((self._clock() + self._hold_s, audio))
        if said := content.get("outputTranscription", {}).get("text"):
            turn.awaiting_reply = False
            turn.said += said
            if not turn.blocked and not UNFILTERED_TOOLS & set(turn.tools):
                if find_violations(turn.said):
                    await self._block(turn)
        if content.get("interrupted"):
            await self._finish(turn, interrupted=True)
        elif content.get("turnComplete") and not turn.awaiting_reply:
            await self._finish(turn)

    # --- output -------------------------------------------------------------------------

    async def _send_transcript(self, turn: _Turn) -> None:
        if not turn.transcript_sent and turn.heard.strip():
            turn.transcript_sent = True
            text = turn.heard.strip()
            await self._send_json(
                {"type": "transcript", "text": text, "language": detect_language(text)}
            )

    async def _forward(self, turn: _Turn, audio: bytes, sample_rate: int, voice: str) -> None:
        if turn.audio_sent == 0:
            await self._send_transcript(turn)
            # Speech stopped at the last speech frame our own VAD saw, as in Pipeline B
            speech_end = self._speech_end or self._detector.last_speech_at or self._clock()
            turn.timer = TurnTimer(speech_end, self._clock)
            turn.timer.mark("first_audio")
            turn.voice = voice
            await self._set_state(VoiceState.SPEAKING)
            await self._send_json({"type": "audio_start", "sample_rate": sample_rate})
        await self._send_audio(audio)
        turn.audio_sent += len(audio) / 2 / sample_rate

    async def _release(self, turn: _Turn, everything: bool = False) -> None:
        """Forward held audio that has waited long enough for its transcript to be checked."""
        async with self._release_lock:  # the flusher and the end of a turn must not interleave
            now = self._clock()
            while self._held and (everything or self._held[0][0] <= now):
                _, audio = self._held.popleft()
                await self._forward(turn, audio, OUTPUT_SAMPLE_RATE, "gemini-live")

    async def _flush_held(self) -> None:
        while True:
            await asyncio.sleep(0.02)
            if self._held and not self._turn.blocked:
                await self._release(self._turn)

    async def _block(self, turn: _Turn) -> None:
        """The reply broke a guardrail: stop it, and say the safe fallback instead."""
        turn.blocked = True
        self._held.clear()
        if turn.audio_sent:
            await self._send_json({"type": "stop_audio"})
        turn.audio_sent, turn.voice = 0.0, ""
        language = detect_language(turn.heard)
        try:
            speech = await self._tts.synthesize(SAFE_FALLBACKS[language], language)
        except TTSUnavailable as error:
            logger.warning("Could not speak the safe fallback: %s", error)
            await self._send_json({"type": "audio_unavailable"})
            return
        await self._forward(turn, speech.pcm16, speech.sample_rate, speech.provider or "fallback")

    async def _finish(self, turn: _Turn, interrupted: bool = False) -> None:
        """The reply is over, or the patient spoke over it."""
        self._turn = _Turn()  # whatever arrives next belongs to the next turn
        if interrupted:
            self._held.clear()
            await self._send_json({"type": "stop_audio"})
        elif not turn.blocked:
            await self._release(turn, everything=True)

        heard = turn.heard.strip()
        language = detect_language(heard)
        reply = SAFE_FALLBACKS[language] if turn.blocked else turn.said.strip()
        if not heard and not reply:
            return
        await self._send_transcript(turn)
        intent = next(
            (TOOL_INTENT[name] for name in turn.tools if name in TOOL_INTENT),
            (self._classifier or get_classifier()).classify(heard).intent,
        )
        if turn.audio_sent and not interrupted:
            if turn.timer:
                turn.timer.mark("last_audio")
            await self._send_json({"type": "audio_end"})
        await self._send_json(
            {
                "type": "reply",
                "text": reply,
                "intent": intent.value,
                "language": language,
                "disclaimer": DISCLAIMERS[language] if intent == Intent.TRIAGE else None,
                "blocked": turn.blocked,
                "tools_called": turn.tools,
            }
        )
        assert self._user.id is not None  # noqa: S101
        if heard and reply:
            await memory.remember(
                self._user.id, self._session_id, heard, reply, intent, "; ".join(turn.notes)
            )
        await expire_older_proposals(ToolContext(self._user, language, self._session_id, turn.id))
        await self._record(turn, intent, language, interrupted)
        if not turn.audio_sent or interrupted:
            await self._set_state(
                VoiceState.LISTENING if interrupted else VoiceState.WAITING_FOR_NEXT_INPUT
            )
        if self._go_away:
            await self._reconnect("the session reached its time limit")

    async def _record(self, turn: _Turn, intent: Intent, language: str, interrupted: bool) -> None:
        stages = turn.timer.stages() if turn.timer else {}
        labels = {
            "routed_by": PIPELINE,
            "used_llm": True,
            "cache_hit": False,
            "answered_by": "gemini-live",
            "stt_provider": "gemini-live",
            "tts_provider": turn.voice,
        }
        try:
            await TurnAnalytics(
                pipeline=PIPELINE,
                intent=intent.value,
                language=language,
                blocked=turn.blocked,
                tools_called=turn.tools,
                spoken=turn.audio_sent > 0,
                barged_in=interrupted,
                stages=stages,
                **labels,
            ).insert()
            await self._send_json({"type": "timing", "stages": stages, **labels})
        except Exception:
            logger.exception("Could not record turn analytics")
