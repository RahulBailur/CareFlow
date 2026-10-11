"""Speech-to-text accuracy on phone-quality audio, measured rather than assumed.

Phone lines carry 8 kHz audio limited to roughly 300-3400 Hz, usually through a G.711
codec. Each labelled utterance is transcribed three ways and scored against its text:

    16 kHz      the audio as the browser sends it
    8 kHz       band-limited to 300-3400 Hz and sampled at 8 kHz
    8 kHz G.711 the same, then through 8-bit mu-law, as on a phone call

Whisper takes 16 kHz, so the phone versions are brought back up to 16 kHz first: the
information lost on the way down stays lost, which is the point.

    python scripts/make_eval_audio.py --lang en,hi,kn,mixed   # once
    python eval/stt_wer.py
    python eval/stt_wer.py --lang en --save

Scores are word error rate (WER) and character error rate (CER). For Hindi, Kannada and
code-mixed speech, Whisper may write a word in another script than the reference (Hindi
in Latin letters, English loanwords in Devanagari), which counts as an error here: read
those rows as an upper bound.
"""

import argparse
import json
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import get_settings
from scripts.make_eval_audio import audio_path, load_utterances, read_wav
from services.stt import FasterWhisperSTT
from voice.audio_utils import SAMPLE_RATE, Audio, pcm16_to_float, resample

RESULTS_DIR = Path(__file__).parent / "results"
PHONE_RATE = 8_000
PHONE_BAND_HZ = (300.0, 3_400.0)
FIR_TAPS = 101
MU = 255.0
CONDITIONS = ("16 kHz", "8 kHz", "8 kHz G.711")
JOINERS = dict.fromkeys(map(ord, "‌‍"))


# --- what a phone line does to audio ----------------------------------------------------


def band_pass(audio: Audio, rate: int, low: float, high: float, taps: int = FIR_TAPS) -> Audio:
    """A windowed-sinc band-pass FIR filter, applied without phase shift."""
    n = np.arange(taps) - (taps - 1) / 2

    def low_pass(cutoff: float) -> np.ndarray:
        return 2 * cutoff / rate * np.sinc(2 * cutoff / rate * n)

    kernel = (low_pass(high) - low_pass(low)) * np.hamming(taps)
    return np.convolve(audio, kernel, mode="same").astype(np.float32)


def mu_law(audio: Audio) -> Audio:
    """Round trip through 8-bit mu-law (G.711), as a phone codec would."""
    clipped = np.clip(audio, -1.0, 1.0)
    encoded = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    quantised = np.round((encoded + 1) / 2 * 255) / 255 * 2 - 1  # 256 levels
    return (np.sign(quantised) * np.expm1(np.abs(quantised) * np.log1p(MU)) / MU).astype(np.float32)


def phone_line(audio: Audio, codec: bool) -> Audio:
    """16 kHz audio as it would arrive over a phone call, back at 16 kHz for Whisper."""
    narrow = band_pass(audio, SAMPLE_RATE, *PHONE_BAND_HZ)
    phone = narrow[:: SAMPLE_RATE // PHONE_RATE]  # already below 4 kHz, so no aliasing
    if codec:
        phone = mu_law(phone)
    return resample(phone, PHONE_RATE, SAMPLE_RATE)


# --- scoring ----------------------------------------------------------------------------


def normalise(text: str) -> str:
    # Zero-width joiners only change how Kannada and Hindi are drawn, not what is said
    text = unicodedata.normalize("NFC", text).translate(JOINERS).lower().replace("-", " ")
    text = "".join(
        ch if ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M") else " "
        for ch in text
    )
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        diagonal, row[0] = row[0], i
        for j, hyp in enumerate(hypothesis, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (ref != hyp))
    return row[-1]


def errors(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """(word errors, reference words, character errors, reference characters)."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    return (
        edit_distance(ref.split(), hyp.split()),
        max(1, len(ref.split())),
        edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", ""))),
        max(1, len(ref.replace(" ", ""))),
    )


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """{language: {condition: {wer, cer, n}}}, plus an "all" language."""
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for row in rows:
        for language in (row["lang"], "all"):
            for condition, (words_wrong, words, chars_wrong, chars) in row["errors"].items():
                total = totals[(language, condition)]
                total[0] += words_wrong
                total[1] += words
                total[2] += chars_wrong
                total[3] += chars
                total[4] += 1
    summary: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (language, condition), (ww, w, cw, c, n) in totals.items():
        summary[language][condition] = {"wer": ww / w, "cer": cw / c, "n": n}
    return dict(summary)


def report(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = [f"{'language':9} {'n':>3}  " + "  ".join(f"{c:>17}" for c in CONDITIONS)]
    lines.append(f"{'':9} {'':>3}  " + "  ".join(f"{'WER':>8} {'CER':>8}" for _ in CONDITIONS))
    for language in ("en", "hi", "kn", "mixed", "all"):
        if language not in summary:
            continue
        by = summary[language]
        n = int(by[CONDITIONS[0]]["n"])
        cells = "  ".join(f"{by[c]['wer']:>8.1%} {by[c]['cer']:>8.1%}" for c in CONDITIONS)
        lines.append(f"{language:9} {n:>3}  {cells}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="STT accuracy at 16 kHz against phone audio")
    parser.add_argument("--lang", default="en,hi,kn,mixed")
    parser.add_argument("--set", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    name = "utterances.jsonl" if args.set == "dev" else "utterances_heldout.jsonl"
    wanted = [
        row
        for row in load_utterances(name)
        if row["lang"] in args.lang.split(",") and audio_path(row["id"]).exists()
    ]
    if not wanted:
        print("No audio found. Run: python scripts/make_eval_audio.py --lang en,hi,kn,mixed")
        return 1
    stt = FasterWhisperSTT(
        settings.whisper_model,
        settings.whisper_compute_type,
        settings.whisper_cpu_threads,
        settings.whisper_min_window_s,
        settings.whisper_multilingual_model,
    )
    stt.warm_up()
    started = time.monotonic()
    rows = []
    for row in wanted:
        clean = pcm16_to_float(read_wav(audio_path(row["id"])))
        versions = {
            "16 kHz": clean,
            "8 kHz": phone_line(clean, codec=False),
            "8 kHz G.711": phone_line(clean, codec=True),
        }
        heard = {condition: stt._transcribe(audio).text for condition, audio in versions.items()}
        rows.append(
            {
                "id": row["id"],
                "lang": row["lang"],
                "heard": heard,
                "errors": {c: errors(row["text"], text) for c, text in heard.items()},
            }
        )
        if args.verbose:
            print(f"  {row['id']:8} ref:  {row['text']}")
            for condition, text in heard.items():
                print(f"  {'':8} {condition:11}: {text}")

    summary = summarise(rows)
    seconds = time.monotonic() - started
    print(f"{stt.name}, {len(rows)} utterances, {seconds:.0f} s")
    print(report(summary))
    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"stt-wer-{args.set}.json"
        result = {
            "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "stt": stt.name,
            "conditions": list(CONDITIONS),
            "summary": summary,
            # The eval sentences are synthetic and public, so the transcripts can be kept
            "utterances": rows,
        }
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
