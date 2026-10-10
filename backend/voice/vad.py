"""Server-side voice activity detection: where an utterance starts and ends, and barge-in."""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from voice.audio_utils import FRAME_MS, FRAME_SAMPLES, Audio

Probability = Callable[[Audio], float]

CONTEXT_SAMPLES = 64


class SileroVAD:
    """Silero VAD run frame by frame with onnxruntime, using the model faster-whisper ships."""

    def __init__(self) -> None:
        import faster_whisper
        import onnxruntime

        model = Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
        if not model.is_file():
            raise RuntimeError(f"Silero VAD model not found at {model}")
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            str(model), providers=["CPUExecutionProvider"], sess_options=options
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: Audio) -> float:
        window = np.concatenate([self._context, frame])[None, :]
        output, self._h, self._c = self._session.run(
            None, {"input": window, "h": self._h, "c": self._c}
        )
        self._context = frame[-CONTEXT_SAMPLES:]
        return float(np.asarray(output).reshape(-1)[0])


def energy_vad(threshold: float = 0.02) -> Probability:
    """Loudness as a stand-in for speech. For tests and for running without the VAD model."""

    def probability(frame: Audio) -> float:
        return 1.0 if float(np.sqrt(np.mean(np.square(frame)))) >= threshold else 0.0

    return probability


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.5
    min_speech_ms: int = 96  # this much continuous speech starts an utterance
    silence_ms: int = 500  # this much silence ends it
    pre_roll_ms: int = 320  # audio kept from just before speech was detected
    max_utterance_s: int = 30


@dataclass(frozen=True)
class SpeechEvent:
    kind: Literal["start", "end"]
    audio: Audio | None = None  # the whole utterance, on "end"
    last_speech_at: float = 0.0  # clock time of the last speech frame, on "end"


def _frames(milliseconds: int) -> int:
    return max(1, -(-milliseconds // FRAME_MS))


class SpeechDetector:
    """Turns per-frame speech probabilities into utterance start and end events."""

    def __init__(
        self,
        probability: Probability,
        config: VADConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probability = probability
        self._config = config or VADConfig()
        self._clock = clock
        self.min_speech_ms = self._config.min_speech_ms
        self._pre_roll: deque[Audio] = deque(maxlen=_frames(self._config.pre_roll_ms))
        self.reset()

    @property
    def in_speech(self) -> bool:
        return self._utterance is not None

    def reset(self) -> None:
        self._pre_roll.clear()
        self._utterance: list[Audio] | None = None
        self._speech_run = 0
        self._silence_run = 0
        self._last_speech_at = 0.0
        if hasattr(self._probability, "reset"):
            self._probability.reset()

    def feed(self, frame: Audio) -> SpeechEvent | None:
        assert len(frame) == FRAME_SAMPLES  # noqa: S101 — frames come from Framer
        is_speech = self._probability(frame) >= self._config.threshold

        if self._utterance is None:
            self._pre_roll.append(frame)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= _frames(self.min_speech_ms):
                self._utterance = list(self._pre_roll)
                self._silence_run = 0
                self._last_speech_at = self._clock()
                return SpeechEvent("start")
            return None

        self._utterance.append(frame)
        if is_speech:
            self._silence_run = 0
            self._last_speech_at = self._clock()
        else:
            self._silence_run += 1
        too_long = len(self._utterance) * FRAME_MS >= self._config.max_utterance_s * 1000
        if self._silence_run >= _frames(self._config.silence_ms) or too_long:
            event = SpeechEvent("end", np.concatenate(self._utterance), self._last_speech_at)
            self.reset()
            return event
        return None
