"""Shared pieces for the voice tests: synthetic audio and recording stand-ins."""

import asyncio
from typing import Any

import numpy as np

from services.stt import STTUnavailable, Transcript
from services.tts import Speech, TTSUnavailable
from voice.audio_utils import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, Audio, float_to_pcm16


def tone(milliseconds: int) -> bytes:
    """A loud tone: 'speech' as far as the energy VAD is concerned."""
    samples = SAMPLE_RATE * milliseconds // 1000
    wave = 0.3 * np.sin(2 * np.pi * 220 * np.arange(samples) / SAMPLE_RATE)
    return float_to_pcm16(wave.astype(np.float32))


def silence(milliseconds: int) -> bytes:
    return b"\x00\x00" * (SAMPLE_RATE * milliseconds // 1000)


def frames(data: bytes) -> list[Audio]:
    audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    count = len(audio) // FRAME_SAMPLES
    return [audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES] for i in range(count)]


# One spoken utterance followed by enough quiet for the test VAD to call it finished
UTTERANCE = tone(10 * FRAME_MS) + silence(12 * FRAME_MS)


class Wire:
    """Collects everything a session sends, in order."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.audio: list[bytes] = []

    async def send_json(self, message: dict[str, Any]) -> None:
        self.events.append(message)

    async def send_audio(self, pcm16: bytes) -> None:
        self.audio.append(pcm16)

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == kind]

    @property
    def states(self) -> list[str]:
        return [event["state"] for event in self.of("state")]


class FakeSTT:
    name = "fake-stt"

    def __init__(self, text: str = "What are the OPD timings?", fail: bool = False) -> None:
        self.text, self.fail = text, fail
        self.heard: list[Audio] = []
        self.gate: asyncio.Event | None = None

    async def transcribe(self, audio: Audio) -> Transcript:
        self.heard.append(audio)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise STTUnavailable("model exploded")
        return Transcript(self.text, "en")

    def warm_up(self) -> None:
        pass


class FakeTTS:
    name = "fake-tts"

    def __init__(self, fail: bool = False, seconds: float = 0.01) -> None:
        self.fail, self.seconds = fail, seconds
        self.spoken: list[str] = []
        self.languages: list[str] = []
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()

    async def synthesize(self, text: str, language: str = "en") -> Speech:
        self.started.set()
        self.languages.append(language)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise TTSUnavailable("quota")
        self.spoken.append(text)
        return Speech(b"\x01\x00" * int(24_000 * self.seconds), 24_000, self.name)

    def warm_up(self) -> None:
        pass
