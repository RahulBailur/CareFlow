"""PCM helpers. Audio lives in memory only: nothing here writes to disk."""

import io
import wave

import numpy as np
from numpy.typing import NDArray

Audio = NDArray[np.float32]  # mono, -1.0 to 1.0

SAMPLE_RATE = 16_000  # what the VAD and STT take
FRAME_SAMPLES = 512  # one Silero VAD frame: 32 ms at 16 kHz
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
FRAME_BYTES = FRAME_SAMPLES * 2


def pcm16_to_float(data: bytes) -> Audio:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: Audio) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample(audio: Audio, source_rate: int, target_rate: int) -> Audio:
    """Linear resampling. Plain, and good enough for speech in a POC."""
    if source_rate == target_rate or len(audio) == 0:
        return audio
    length = round(len(audio) * target_rate / source_rate)
    positions = np.linspace(0, len(audio) - 1, length)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def decode_speech(data: bytes, mime_type: str) -> tuple[bytes, int]:
    """(PCM16 mono bytes, sample rate) from a TTS payload: a WAV file or raw `audio/L16`."""
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data)) as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("expected 16-bit audio")
            frames = wav.readframes(wav.getnframes())
            rate, channels = wav.getframerate(), wav.getnchannels()
        if channels > 1:
            mixed = np.frombuffer(frames, dtype="<i2").reshape(-1, channels).mean(axis=1)
            frames = mixed.astype("<i2").tobytes()
        return frames, rate
    rate = 24_000
    for part in mime_type.split(";"):
        key, _, value = part.strip().partition("=")
        if key == "rate" and value.isdigit():
            rate = int(value)
    return data, rate


class Framer:
    """Cuts an arbitrary stream of PCM16 chunks into fixed VAD frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def push(self, data: bytes) -> list[Audio]:
        self._buffer.extend(data)
        frames = []
        while len(self._buffer) >= FRAME_BYTES:
            frames.append(pcm16_to_float(bytes(self._buffer[:FRAME_BYTES])))
            del self._buffer[:FRAME_BYTES]
        return frames
