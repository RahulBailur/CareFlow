"""Per-stage timers for one voice turn. The numbers the whole project is measured by."""

import time
from collections.abc import Callable

# In the order they happen. Each is the clock time at which that point was reached.
MARKS = ("speech_end", "vad_end", "transcript", "routed", "reply", "first_audio", "last_audio")

# stage name -> (from mark, to mark)
STAGES: dict[str, tuple[str, str]] = {
    "end_of_speech_ms": ("speech_end", "vad_end"),  # the VAD's silence window
    "stt_ms": ("vad_end", "transcript"),
    "routing_ms": ("transcript", "routed"),
    "llm_ms": ("routed", "reply"),  # the agent's whole answer, tools included
    "tts_first_chunk_ms": ("reply", "first_audio"),
    # The headline number: the user stops speaking -> the first audio is sent
    "ttfa_ms": ("speech_end", "first_audio"),
    "total_ms": ("speech_end", "last_audio"),
}


class TurnTimer:
    def __init__(self, speech_end: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._marks: dict[str, float] = {"speech_end": speech_end}

    def mark(self, name: str) -> None:
        assert name in MARKS  # noqa: S101 — a typo here would silently lose a measurement
        self._marks.setdefault(name, self._clock())

    def stages(self) -> dict[str, float]:
        """Milliseconds for every stage whose start and end were both reached."""
        return {
            stage: round((self._marks[end] - self._marks[start]) * 1000, 1)
            for stage, (start, end) in STAGES.items()
            if start in self._marks and end in self._marks
        }
