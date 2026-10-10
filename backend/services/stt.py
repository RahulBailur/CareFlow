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

WHISPER_WINDOW_S = 30.0  # what Whisper was trained on, and its upper limit
WINDOW_MARGIN_S = 5.0
FEATURE_FRAMES_PER_SECOND = 100
MAX_NEW_TOKENS = 128
NO_SPEECH_THRESHOLD = 0.6
# Average token log-probability. Measured: real speech -0.6 or better, noise -1.8 or worse
MIN_AVG_LOG_PROB = -1.0
START = "<|startoftranscript|>"
NO_TIMESTAMPS = "<|notimestamps|>"


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


def encode_window_seconds(utterance_seconds: float, minimum: float) -> float:
    """How much audio to give the Whisper encoder for one utterance.

    Whisper pads everything to 30 s, and the encoder cost follows the padded length, so a
    2 s question costs as much as a 30 s one. A shorter window is several times faster.
    It must not be tight, though: with barely any padding after the speech the decoder
    loops and repeats itself, so the speech always gets `WINDOW_MARGIN_S` of quiet after it.
    """
    return min(WHISPER_WINDOW_S, max(minimum, utterance_seconds + WINDOW_MARGIN_S))


class FasterWhisperSTT:
    def __init__(
        self, model: str, compute_type: str, cpu_threads: int = 0, min_window_s: float = 8.0
    ) -> None:
        self.name = f"faster-whisper:{model}"
        self._model_name, self._compute_type = model, compute_type
        self._cpu_threads = cpu_threads
        self._min_window_s = min_window_s  # 0 keeps the full 30 s window
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
            self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def _run(self, audio: Audio, language: str | None) -> tuple[str, Any]:
        segments, info = self._model.transcribe(
            audio, language=language, beam_size=1, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments).strip(), info

    def _full_window(self, audio: Audio) -> Transcript:
        text, info = self._run(audio, None)
        language = info.language
        if language not in LANGUAGES:
            # Heard as some other language: take the likeliest of ours and listen again
            likely = dict(info.all_language_probs or [])
            language = max(LANGUAGES, key=lambda code: likely.get(code, 0.0))
            text, _ = self._run(audio, language)
        return Transcript(text, language)

    def _short_window(self, audio: Audio, window_s: float) -> Transcript:
        """One encoder pass over `window_s` seconds, then a greedy decode."""
        import ctranslate2

        model, tokens = self._model.model, self._model.hf_tokenizer
        frames = int(window_s * FEATURE_FRAMES_PER_SECOND)
        features = self._model.feature_extractor(audio)[:, :frames]
        if features.shape[1] < frames:
            features = np.pad(features, ((0, 0), (0, frames - features.shape[1])))
        batch = np.ascontiguousarray(features[None].astype(np.float32))
        encoded = model.encode(ctranslate2.StorageView.from_array(batch))

        # The likeliest of our own languages, whatever else Whisper thinks it heard
        likely = dict(model.detect_language(encoded)[0])
        language = max(LANGUAGES, key=lambda code: likely.get(f"<|{code}|>", 0.0))
        prompt = [
            tokens.token_to_id(token)
            for token in (START, f"<|{language}|>", "<|transcribe|>", NO_TIMESTAMPS)
        ]
        result = model.generate(
            encoded,
            [prompt],
            beam_size=1,
            max_length=MAX_NEW_TOKENS,
            no_repeat_ngram_size=4,  # a second guard against the decoder looping
            return_no_speech_prob=True,
            return_scores=True,
        )[0]
        # Noise the VAD let through: Whisper would sooner invent a sentence than say nothing
        if result.no_speech_prob > NO_SPEECH_THRESHOLD or result.scores[0] < MIN_AVG_LOG_PROB:
            return Transcript("", language)
        end = tokens.token_to_id("<|endoftext|>")
        text = tokens.decode([token for token in result.sequences_ids[0] if token < end])
        return Transcript(text.strip(), language)

    def _transcribe(self, audio: Audio) -> Transcript:
        self.warm_up()
        window_s = encode_window_seconds(len(audio) / SAMPLE_RATE, self._min_window_s)
        if self._min_window_s <= 0 or window_s >= WHISPER_WINDOW_S:
            return self._full_window(audio)
        return self._short_window(audio, window_s)

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
            settings.whisper_model,
            settings.whisper_compute_type,
            settings.whisper_cpu_threads,
            settings.whisper_min_window_s,
        )
    return MockSTT()
