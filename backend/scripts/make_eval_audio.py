"""Synthesise the labelled utterances as speech, for the latency and STT benchmarks.

No real voices: English is spoken by a local Piper voice (a different one from CareBot's
own), Hindi and Kannada by Gemini TTS. Output is 16 kHz mono WAV under eval/audio/, which
is gitignored and can be rebuilt at any time. Existing files are kept.

    python scripts/make_eval_audio.py                 # English only: no API calls
    python scripts/make_eval_audio.py --lang en,hi,kn # the rest use the Gemini TTS quota
"""

import argparse
import asyncio
import json
import wave
from pathlib import Path

from config import get_settings
from services.tts import GeminiTTS, PiperTTS, TTSProvider, TTSUnavailable
from voice.audio_utils import SAMPLE_RATE, float_to_pcm16, pcm16_to_float, resample

EVAL_DIR = Path(__file__).parent.parent / "eval"
AUDIO_DIR = EVAL_DIR / "audio"
SPEAKER_VOICE = "en_US-ryan-medium"  # not the voice CareBot answers in
PAUSE_BETWEEN_GEMINI_CALLS_S = 8.0  # the free tier allows only a few TTS requests a minute


def load_utterances(name: str = "utterances.jsonl") -> list[dict[str, str]]:
    lines = (EVAL_DIR / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def audio_path(utterance_id: str) -> Path:
    return AUDIO_DIR / f"{utterance_id}.wav"


def write_wav(path: Path, pcm16: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16)


def read_wav(path: Path) -> bytes:
    """PCM16 mono 16 kHz, as written by `write_wav`."""
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


async def synthesise(row: dict[str, str], piper: TTSProvider, gemini: TTSProvider) -> bytes:
    # Romanised Hindi and Kannada ("mixed") is still read best by a multilingual voice
    if row["lang"] == "en":
        speech = await piper.synthesize(row["text"], "en")
    else:
        speech = await gemini.synthesize(row["text"], "hi")
        await asyncio.sleep(PAUSE_BETWEEN_GEMINI_CALLS_S)
    audio = resample(pcm16_to_float(speech.pcm16), speech.sample_rate, SAMPLE_RATE)
    return float_to_pcm16(audio)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesise eval utterances as speech")
    parser.add_argument("--lang", default="en", help="comma-separated: en,hi,kn,mixed")
    parser.add_argument("--limit", type=int, default=0, help="at most this many new files")
    args = parser.parse_args()

    settings = get_settings()
    piper = PiperTTS(SPEAKER_VOICE)
    gemini = GeminiTTS(settings.gemini_api_key, settings.gemini_tts_model)
    wanted = [row for row in load_utterances() if row["lang"] in args.lang.split(",")]
    made = skipped = failed = 0
    for row in wanted:
        path = audio_path(row["id"])
        if path.exists():
            skipped += 1
            continue
        if args.limit and made >= args.limit:
            break
        try:
            write_wav(path, await synthesise(row, piper, gemini))
            made += 1
        except TTSUnavailable as error:
            failed += 1
            print(f"  {row['id']}: could not synthesise ({error})")
    print(f"{made} made, {skipped} already there, {failed} failed -> {AUDIO_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
