"""Pipeline B: VAD -> STT -> the same agents as text chat -> TTS, one sentence at a time.

One `VoiceSession` per WebSocket. Audio arrives as it is spoken; a turn starts when the VAD
hears the end of an utterance. Speaking can be interrupted by the user (barge-in); the
steps before it cannot, because by then a tool may be halfway through changing a booking.
"""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from agents.intent_classifier import IntentClassifier
from agents.orchestrator import TurnResult, run_text_turn
from models.turn_analytics import TurnAnalytics
from models.user import User
from services.llm import LLMProvider
from services.stt import STTProvider, STTUnavailable
from services.tts import TTSProvider, TTSUnavailable
from voice.audio_utils import Framer
from voice.timing import TurnTimer
from voice.vad import Probability, SpeechDetector, SpeechEvent, VADConfig

logger = logging.getLogger(__name__)

PIPELINE = "cascade"
BARGE_IN_MIN_SPEECH_MS = 256  # stricter than a normal start, so echo and coughs are ignored
PLAYBACK_GRACE_S = 3.0
MIN_CHUNK_CHARS = 24

SendJson = Callable[[dict[str, Any]], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]


class VoiceState(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    GENERATING_RESPONSE = "GENERATING_RESPONSE"
    SPEAKING = "SPEAKING"
    WAITING_FOR_NEXT_INPUT = "WAITING_FOR_NEXT_INPUT"
    ERROR = "ERROR"
    END = "END"


# While a turn is being worked out, incoming audio is ignored
BUSY = frozenset(
    {
        VoiceState.TRANSCRIBING,
        VoiceState.THINKING,
        VoiceState.TOOL_EXECUTION,
        VoiceState.GENERATING_RESPONSE,
    }
)

_SENTENCE_END = re.compile(r"(?<=[.?!।])\s+")
_ABBREVIATIONS = ("dr.", "mr.", "mrs.", "ms.", "no.", "a.m.", "p.m.", "st.")


def split_sentences(text: str, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Sentences to speak one at a time, without cutting at "Dr." or leaving tiny scraps."""
    chunks: list[str] = []
    for piece in filter(None, _SENTENCE_END.split(text.strip())):
        previous = chunks[-1] if chunks else ""
        if previous and (
            previous.lower().endswith(_ABBREVIATIONS)
            or len(previous) < min_chars
            or len(piece) < min_chars
        ):
            chunks[-1] = f"{previous} {piece}"
        else:
            chunks.append(piece)
    return chunks


class VoiceSession:
    def __init__(
        self,
        user: User,
        session_id: str,
        send_json: SendJson,
        send_audio: SendAudio,
        *,
        stt: STTProvider,
        tts: TTSProvider,
        probability: Probability,
        vad_config: VADConfig | None = None,
        chunking: str = "sentence",
        llm: LLMProvider | None = None,
        classifier: IntentClassifier | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._user, self._session_id = user, session_id
        self._send_json, self._send_audio = send_json, send_audio
        self._stt, self._tts = stt, tts
        self._chunking = chunking
        self._llm, self._classifier = llm, classifier
        self._clock = clock
        self._vad_config = vad_config or VADConfig()
        self._detector = SpeechDetector(probability, self._vad_config, clock)
        self._framer = Framer()
        self._turn: asyncio.Task[None] | None = None
        self._playback_timer: asyncio.TimerHandle | None = None
        self._timer: TurnTimer | None = None
        self.state = VoiceState.IDLE

    # --- state --------------------------------------------------------------------------

    async def _set_state(self, state: VoiceState) -> None:
        self.state = state
        speaking = state == VoiceState.SPEAKING
        self._detector.min_speech_ms = (
            BARGE_IN_MIN_SPEECH_MS if speaking else self._vad_config.min_speech_ms
        )
        await self._send_json({"type": "state", "state": state.value})

    async def start(self) -> None:
        await self._set_state(VoiceState.IDLE)

    async def close(self) -> None:
        self._cancel_playback_timer()
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
            await asyncio.gather(self._turn, return_exceptions=True)
        self.state = VoiceState.END

    # --- input --------------------------------------------------------------------------

    async def receive_audio(self, data: bytes) -> None:
        """PCM16 mono 16 kHz from the microphone, in chunks of any size."""
        for frame in self._framer.push(data):
            if self.state in BUSY:
                continue
            event = self._detector.feed(frame)
            if event is None:
                continue
            if event.kind == "start":
                if self.state == VoiceState.SPEAKING:
                    await self._barge_in()
                await self._set_state(VoiceState.LISTENING)
            else:
                # Set before the task first runs, so the very next frame is already ignored
                self.state = VoiceState.TRANSCRIBING
                self._detector.reset()
                self._turn = asyncio.create_task(self._run_turn(event))

    async def playback_done(self) -> None:
        """The client has finished playing everything it was sent."""
        if self.state == VoiceState.SPEAKING and (self._turn is None or self._turn.done()):
            self._cancel_playback_timer()
            await self._set_state(VoiceState.WAITING_FOR_NEXT_INPUT)

    async def _barge_in(self) -> None:
        self._cancel_playback_timer()
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
            await asyncio.gather(self._turn, return_exceptions=True)
        await self._send_json({"type": "stop_audio"})

    def _cancel_playback_timer(self) -> None:
        if self._playback_timer is not None:
            self._playback_timer.cancel()
            self._playback_timer = None

    # --- one turn -----------------------------------------------------------------------

    async def _progress(self, step: str) -> None:
        if step == "routed" and self._timer is not None:
            self._timer.mark("routed")
        elif step == "tool":
            await self._set_state(VoiceState.TOOL_EXECUTION)
        elif step == "generating":
            await self._set_state(VoiceState.GENERATING_RESPONSE)

    async def _fail(self, message: str) -> None:
        await self._send_json({"type": "error", "message": message})
        await self._set_state(VoiceState.ERROR)
        await self._set_state(VoiceState.WAITING_FOR_NEXT_INPUT)

    async def _run_turn(self, utterance: SpeechEvent) -> None:
        assert utterance.audio is not None  # noqa: S101 — "end" events carry the audio
        timer = self._timer = TurnTimer(utterance.last_speech_at, self._clock)
        timer.mark("vad_end")
        await self._set_state(VoiceState.TRANSCRIBING)
        try:
            transcript = await self._stt.transcribe(utterance.audio)
        except STTUnavailable as error:
            logger.warning("STT unavailable: %s", error)
            return await self._fail("Sorry, I could not hear that. Please try again.")
        timer.mark("transcript")
        if not transcript.text:
            return await self._set_state(VoiceState.WAITING_FOR_NEXT_INPUT)
        await self._send_json(
            {"type": "transcript", "text": transcript.text, "language": transcript.language}
        )

        await self._set_state(VoiceState.THINKING)
        try:
            result = await run_text_turn(
                self._user,
                transcript.text,
                self._session_id,
                self._llm,
                self._classifier,
                progress=self._progress,
            )
        except Exception:
            logger.exception("Voice turn failed")
            return await self._fail("Sorry, something went wrong. Please try again.")
        timer.mark("reply")
        await self._send_json(
            {
                "type": "reply",
                "text": result.reply,
                "intent": result.intent.value,
                "language": result.reply_language,
                "disclaimer": result.disclaimer,
                "blocked": result.blocked,
                "tools_called": result.tools_called,
            }
        )

        # From here the reply is committed, so being interrupted loses nothing
        await self._set_state(VoiceState.SPEAKING)
        seconds, voice, barged_in = 0.0, "", False
        try:
            seconds, voice = await self._speak(result, timer)
        except asyncio.CancelledError:
            barged_in = True
        await self._record(result, timer, voice=voice, barged_in=barged_in)
        if barged_in:
            return
        if seconds > 0:
            # Stay in SPEAKING until the client says playback finished, or clearly should have
            loop = asyncio.get_running_loop()
            self._playback_timer = loop.call_later(
                seconds + PLAYBACK_GRACE_S, lambda: loop.create_task(self._playback_timed_out())
            )
        else:
            await self._set_state(VoiceState.WAITING_FOR_NEXT_INPUT)

    async def _playback_timed_out(self) -> None:
        self._playback_timer = None
        if self.state == VoiceState.SPEAKING:
            await self._set_state(VoiceState.WAITING_FOR_NEXT_INPUT)

    async def _speak(self, result: TurnResult, timer: TurnTimer) -> tuple[float, str]:
        """Synthesise and send the reply. Returns the seconds sent and which voice spoke."""
        # A triage disclaimer is part of the answer, so it is spoken, not only shown
        spoken_text = f"{result.reply} {result.disclaimer}" if result.disclaimer else result.reply
        chunks = split_sentences(spoken_text) if self._chunking == "sentence" else [spoken_text]
        seconds, voice = 0.0, ""
        try:
            for chunk in chunks:
                speech = await self._tts.synthesize(chunk, result.reply_language)
                if seconds == 0:
                    timer.mark("first_audio")
                    await self._send_json(
                        {"type": "audio_start", "sample_rate": speech.sample_rate}
                    )
                await self._send_audio(speech.pcm16)
                seconds += speech.seconds
                voice = speech.provider or self._tts.name
        except TTSUnavailable as error:
            logger.warning("TTS unavailable: %s", error)
        if seconds > 0:
            timer.mark("last_audio")
            await self._send_json({"type": "audio_end"})
        else:
            await self._send_json({"type": "audio_unavailable"})
        return seconds, voice

    async def _record(
        self, result: TurnResult, timer: TurnTimer, *, voice: str, barged_in: bool
    ) -> None:
        stages = timer.stages()
        try:
            await TurnAnalytics(
                pipeline=PIPELINE,
                intent=result.intent.value,
                language=result.language,
                routed_by=result.routed_by,
                used_llm=result.used_llm,
                cache_hit=result.cached,
                answered_by=result.answered_by,
                blocked=result.blocked,
                tools_called=result.tools_called,
                stt_provider=self._stt.name,
                tts_provider=voice,
                spoken=bool(voice),
                barged_in=barged_in,
                stages=stages,
            ).insert()
            await self._send_json(
                {
                    "type": "timing",
                    "stages": stages,
                    # Labels for whoever is measuring: which path this turn took
                    "routed_by": result.routed_by,
                    "used_llm": result.used_llm,
                    "cache_hit": result.cached,
                    "answered_by": result.answered_by,
                    "stt_provider": self._stt.name,
                    "tts_provider": voice,
                }
            )
        except Exception:
            logger.exception("Could not record turn analytics")
