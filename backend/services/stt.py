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
    """faster-whisper on CPU, with an optional second model for Hindi and Kannada.

    Measured on the eval set: `base` is fast and accurate for English but writes Hindi in
    Urdu script, which nothing downstream can read; `small` writes Devanagari. So the main
    model listens first, and if it hears Hindi or Kannada the multilingual model transcribes.
    """

    def __init__(
        self,
        model: str,
        compute_type: str,
        cpu_threads: int = 0,
        min_window_s: float = 8.0,
        multilingual_model: str = "",
    ) -> None:
        second = multilingual_model if multilingual_model and multilingual_model != model else ""
        self.name = f"faster-whisper:{model}" + (f"+{second}" if second else "")
        self._model_name, self._multilingual_name = model, second
        self._compute_type, self._cpu_threads = compute_type, cpu_threads
        self._min_window_s = min_window_s  # 0 keeps the full 30 s window
        self._model: Any = None
        self._multilingual: Any = None

    def _load(self, name: str) -> Any:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            name, device="cpu", compute_type=self._compute_type, cpu_threads=self._cpu_threads
        )
        # One throwaway pass, so the first real utterance does not pay for a cold start
        self._short_window(model, np.zeros(SAMPLE_RATE, dtype=np.float32), 8.0, "en")
        return model

    def warm_up(self) -> None:
        if self._model is None:
            self._model = self._load(self._model_name)
        if self._multilingual_name and self._multilingual is None:
            self._multilingual = self._load(self._multilingual_name)

    def _run(self, model: Any, audio: Audio, language: str | None) -> tuple[str, Any]:
        segments, info = model.transcribe(
            audio, language=language, beam_size=1, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments).strip(), info

    def _full_window(self, model: Any, audio: Audio, language: str | None) -> Transcript:
        text, info = self._run(model, audio, language)
        heard = info.language
        if heard not in LANGUAGES:
            # Heard as some other language: take the likeliest of ours and listen again
            likely = dict(info.all_language_probs or [])
            heard = max(LANGUAGES, key=lambda code: likely.get(code, 0.0))
            text, _ = self._run(model, audio, heard)
        return Transcript(text, heard)

    def _short_window(
        self, model: Any, audio: Audio, window_s: float, language: str | None
    ) -> Transcript:
        """One encoder pass over `window_s` seconds, then a greedy decode."""
        import ctranslate2

        whisper, tokens = model.model, model.hf_tokenizer
        frames = int(window_s * FEATURE_FRAMES_PER_SECOND)
        features = model.feature_extractor(audio)[:, :frames]
        if features.shape[1] < frames:
            features = np.pad(features, ((0, 0), (0, frames - features.shape[1])))
        batch = np.ascontiguousarray(features[None].astype(np.float32))
        encoded = whisper.encode(ctranslate2.StorageView.from_array(batch))

        if language is None:
            # The likeliest of our own languages, whatever else Whisper thinks it heard
            likely = dict(whisper.detect_language(encoded)[0])
            language = max(LANGUAGES, key=lambda code: likely.get(f"<|{code}|>", 0.0))
        prompt = [
            tokens.token_to_id(token)
            for token in (START, f"<|{language}|>", "<|transcribe|>", NO_TIMESTAMPS)
        ]
        result = whisper.generate(
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

    def _with(self, model: Any, audio: Audio, language: str | None) -> Transcript:
        window_s = encode_window_seconds(len(audio) / SAMPLE_RATE, self._min_window_s)
        if self._min_window_s <= 0 or window_s >= WHISPER_WINDOW_S:
            return self._full_window(model, audio, language)
        return self._short_window(model, audio, window_s, language)

    def _transcribe(self, audio: Audio) -> Transcript:
        self.warm_up()
        first = self._with(self._model, audio, None)
        if first.language == "en" or self._multilingual is None:
            return first
        # Hindi or Kannada: the language is known now, so the second model only transcribes
        return self._with(self._multilingual, audio, first.language)

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
            settings.whisper_multilingual_model,
        )
    return MockSTT()
