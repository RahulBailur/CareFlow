"""Text to speech: Gemini TTS, with Piper (local, English) behind it, or a mock."""

import asyncio
import base64
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import httpx

from config import get_settings
from services.llm import GEMINI_URL
from voice.audio_utils import decode_speech

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 30.0
VOICE = "Kore"
COOLDOWN_S = 60.0  # how long a provider that just failed is skipped


class TTSUnavailable(Exception):
    """This text could not be spoken. The reply is still shown as text."""


@dataclass(frozen=True)
class Speech:
    pcm16: bytes  # mono, little-endian
    sample_rate: int
    provider: str = ""

    @property
    def seconds(self) -> float:
        return len(self.pcm16) / 2 / self.sample_rate


class TTSProvider(Protocol):
    name: str

    async def synthesize(self, text: str, language: str = "en") -> Speech: ...

    def warm_up(self) -> None: ...


class MockTTS:
    """Silence, 50 ms per character. For tests and for running without a TTS model."""

    name = "mock"
    sample_rate = 24_000

    async def synthesize(self, text: str, language: str = "en") -> Speech:
        samples = int(self.sample_rate * 0.05 * max(1, len(text)))
        return Speech(b"\x00\x00" * samples, self.sample_rate, self.name)

    def warm_up(self) -> None:
        pass


class GeminiTTS:
    name = "gemini"

    def __init__(
        self, api_key: str, model: str, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._api_key, self._model, self._transport = api_key, model, transport

    def warm_up(self) -> None:
        pass

    async def synthesize(self, text: str, language: str = "en") -> Speech:
        if not self._api_key or not self._model:
            raise TTSUnavailable("GEMINI_API_KEY and GEMINI_TTS_MODEL must both be set")
        body = {
            # The model reads its whole input aloud, so the text goes in bare
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}},
            },
        }
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=REQUEST_TIMEOUT_S
            ) as client:
                response = await client.post(
                    GEMINI_URL.format(model=self._model),
                    headers={"x-goog-api-key": self._api_key},
                    json=body,
                )
        except httpx.HTTPError as error:
            raise TTSUnavailable(f"Gemini TTS request failed: {type(error).__name__}") from error
        if response.status_code != httpx.codes.OK:
            raise TTSUnavailable(f"Gemini TTS returned HTTP {response.status_code}")
        try:
            inline = response.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
            pcm16, rate = decode_speech(base64.b64decode(inline["data"]), inline["mimeType"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise TTSUnavailable("Gemini TTS returned no audio") from error
        if not pcm16:
            raise TTSUnavailable("Gemini TTS returned empty audio")
        return Speech(pcm16, rate, self.name)


class PiperTTS:
    """Piper running locally on CPU. The voice is English, so other languages are refused."""

    name = "piper"

    def __init__(self, voice: str) -> None:
        self._voice_name = voice
        self._voice: Any = None

    def warm_up(self) -> None:
        if self._voice is not None:
            return
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        # e.g. "en_US-lessac-medium" lives at en/en_US/lessac/medium/ in rhasspy/piper-voices
        locale, speaker, quality = self._voice_name.split("-")
        folder = f"{locale.split('_')[0]}/{locale}/{speaker}/{quality}"
        model, config = (
            hf_hub_download("rhasspy/piper-voices", f"{folder}/{self._voice_name}.onnx{suffix}")
            for suffix in ("", ".json")
        )
        voice = PiperVoice.load(model, config_path=config)
        list(voice.synthesize("Ready."))  # a throwaway pass, so the first reply is not cold
        self._voice = voice

    def _synthesize(self, text: str) -> Speech:
        self.warm_up()
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            raise TTSUnavailable("Piper produced no audio")
        audio = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        return Speech(audio, chunks[0].sample_rate, self.name)

    async def synthesize(self, text: str, language: str = "en") -> Speech:
        if language != "en":
            raise TTSUnavailable(f"The Piper voice is English only, not {language}")
        try:
            return await asyncio.to_thread(self._synthesize, text)
        except TTSUnavailable:
            raise
        except Exception as error:  # noqa: BLE001 — a missing package, model or runtime failure
            raise TTSUnavailable(f"Piper failed: {type(error).__name__}: {error}") from error


class FallbackTTS:
    """Tries each provider in order. One that fails for a reason of its own (quota, timeout)
    is skipped for a minute, so every sentence does not wait on it again."""

    def __init__(
        self, providers: list[TTSProvider], clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._providers, self._clock = providers, clock
        self._skip_until: dict[str, float] = {}
        self.name = "+".join(provider.name for provider in providers)

    def warm_up(self) -> None:
        for provider in self._providers:
            try:
                provider.warm_up()
            except Exception as error:  # noqa: BLE001 — one unusable provider must not stop the rest
                logger.warning("TTS provider %s could not warm up: %s", provider.name, error)

    async def synthesize(self, text: str, language: str = "en") -> Speech:
        reasons = []
        for provider in self._providers:
            if self._clock() < self._skip_until.get(provider.name, 0.0):
                reasons.append(f"{provider.name}: cooling down")
                continue
            try:
                return await provider.synthesize(text, language)
            except TTSUnavailable as error:
                reasons.append(f"{provider.name}: {error}")
                if "English only" not in str(error):
                    self._skip_until[provider.name] = self._clock() + COOLDOWN_S
                    logger.warning("TTS provider %s unavailable: %s", provider.name, error)
        raise TTSUnavailable("; ".join(reasons) or "no TTS provider configured")


@lru_cache
def get_tts() -> TTSProvider:
    """`TTS_PROVIDER` names the first choice; the other real provider stands behind it."""
    settings = get_settings()
    if settings.tts_provider == "mock":
        return MockTTS()
    gemini = GeminiTTS(settings.gemini_api_key, settings.gemini_tts_model)
    piper = PiperTTS(settings.piper_voice)
    order: list[TTSProvider] = (
        [piper, gemini] if settings.tts_provider == "piper" else [gemini, piper]
    )
    return FallbackTTS(order)
