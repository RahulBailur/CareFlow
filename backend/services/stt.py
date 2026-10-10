"""Speech to text. faster-whisper on CPU, or a mock."""

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from config import get_settings
from voice.audio_utils import SAMPLE_RATE, Audio

# CareBot's languages. Whisper is steered back to these if it hears something else.
LANGUAGES = ("en", "hi", "kn")


class STTUnavailable(Exception):
    """The utterance could not be transcribed."""


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str = "en"


class STTProvider(Protocol):
    name: str

    async def transcribe(self, audio: Audio) -> Transcript: ...

    def warm_up(self) -> None: ...


class MockSTT:
    """Returns a fixed transcript. For tests and for running without a speech model."""

    name = "mock"

    def __init__(self, text: str = "What are the OPD timings?", language: str = "en") -> None:
        self.text, self.language = text, language

    async def transcribe(self, audio: Audio) -> Transcript:
        return Transcript(self.text, self.language)

    def warm_up(self) -> None:
        pass


class FasterWhisperSTT:
    def __init__(self, model: str, compute_type: str, cpu_threads: int = 0) -> None:
        self.name = f"faster-whisper:{model}"
        self._model_name, self._compute_type = model, compute_type
        self._cpu_threads = cpu_threads
        self._model: Any = None

    def warm_up(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_name,
                device="cpu",
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
            )
            # One throwaway pass, so the first real utterance does not pay for a cold start
            self._run(np.zeros(SAMPLE_RATE, dtype=np.float32), "en")

    def _run(self, audio: Audio, language: str | None) -> tuple[str, Any]:
        segments, info = self._model.transcribe(
            audio, language=language, beam_size=1, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments).strip(), info

    def _transcribe(self, audio: Audio) -> Transcript:
        self.warm_up()
        text, info = self._run(audio, None)
        language = info.language
        if language not in LANGUAGES:
            # Heard as some other language: take the likeliest of ours and listen again
            likely = dict(info.all_language_probs or [])
            language = max(LANGUAGES, key=lambda code: likely.get(code, 0.0))
            text, _ = self._run(audio, language)
        return Transcript(text, language)

    async def transcribe(self, audio: Audio) -> Transcript:
        try:
            return await asyncio.to_thread(self._transcribe, audio)
        except Exception as error:  # noqa: BLE001 — any model or runtime failure
            raise STTUnavailable(f"{type(error).__name__}: {error}") from error


@lru_cache
def get_stt() -> STTProvider:
    settings = get_settings()
    if settings.stt_provider == "faster_whisper":
        return FasterWhisperSTT(
            settings.whisper_model, settings.whisper_compute_type, settings.whisper_cpu_threads
        )
    return MockSTT()
