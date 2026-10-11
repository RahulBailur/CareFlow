"""Measure the voice pipeline turn by turn, against a running CareFlow server.

Each utterance is streamed over /ws/voice at the speed it would be spoken, exactly as the
browser does. The server reports its own per-stage timings for every turn; this script
collects them and prints P50 / P95 per stage.

    python scripts/make_eval_audio.py                      # once: synthesise the questions
    python eval/latency_bench.py --label baseline
    python eval/latency_bench.py --label "after X" --turns 10 --save

With few turns a P95 is close to the slowest turn: the turn count is printed with it.
`--save` writes eval/results/<label>.json (timings and labels only, nothing that was said).
"""

import argparse
import asyncio
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import httpx

from config import get_settings
from scripts.make_eval_audio import audio_path, load_utterances, read_wav

RESULTS_DIR = Path(__file__).parent / "results"
CHUNK_MS = 100
CHUNK_BYTES = 16_000 * 2 * CHUNK_MS // 1000
TURN_TIMEOUT_S = 180.0

# Printed in this order; the names are the server's (voice/timing.py)
STAGES = [
    ("end_of_speech_ms", "End-of-speech detection"),
    ("stt_ms", "Speech to text"),
    ("routing_ms", "Routing"),
    ("llm_ms", "LLM (agent, tools included)"),
    ("tts_first_chunk_ms", "TTS first chunk"),
    ("ttfa_ms", "Time to first audio"),
    ("total_ms", "Whole turn"),
]


def percentile(values: list[float], percent: float) -> float:
    """Nearest-rank percentile: always one of the measured values."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def summarise(turns: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """{stage: {n, p50, p95, min, max}} over the turns that reached that stage."""
    summary = {}
    for stage, _ in STAGES:
        values = [turn["stages"][stage] for turn in turns if stage in turn["stages"]]
        if values:
            summary[stage] = {
                "n": len(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "min": min(values),
                "max": max(values),
            }
    return summary


def count_by(turns: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for turn in turns:
        label = str(turn.get(key) or "none")
        counts[label] = counts.get(label, 0) + 1
    return counts


def report(label: str, turns: list[dict[str, Any]]) -> str:
    lines = [
        f"{label}: {len(turns)} turns",
        f"{'stage':30} {'n':>3} {'P50':>9} {'P95':>9} {'max':>9}",
    ]
    for stage, name in STAGES:
        stats = summarise(turns).get(stage)
        if stats:
            p50, p95, worst = (stats[key] / 1000 for key in ("p50", "p95", "max"))
            lines.append(f"{name:30} {stats['n']:>3} {p50:>7.2f} s {p95:>7.2f} s {worst:>7.2f} s")
    lines.append(f"spoken by: {count_by(turns, 'tts_provider')}")
    lines.append(f"routed by: {count_by(turns, 'routed_by')}")
    lines.append(f"answered by: {count_by(turns, 'answered_by')}")
    for label, key in (("with the LLM", "used_llm"), ("from the cache", "cache_hit")):
        lines.append(f"answered {label}: {sum(1 for t in turns if t.get(key))}/{len(turns)}")
    return "\n".join(lines)


async def run_turn(ws: aiohttp.ClientWebSocketResponse, pcm: bytes) -> dict[str, Any] | None:
    """Speak one utterance and wait for the server's timing report."""
    for start in range(0, len(pcm), CHUNK_BYTES):
        await ws.send_bytes(pcm[start : start + CHUNK_BYTES])
        await asyncio.sleep(CHUNK_MS / 1000)
    spoken_until = time.monotonic()

    quiet, stop = b"\x00\x00" * (CHUNK_BYTES // 2), asyncio.Event()

    async def keep_silent() -> None:  # a real microphone never stops sending
        while not stop.is_set():
            await ws.send_bytes(quiet)
            await asyncio.sleep(CHUNK_MS / 1000)

    silence = asyncio.create_task(keep_silent())
    turn: dict[str, Any] = {"intent": None, "heard": False}
    transcribing = False
    try:
        async with asyncio.timeout(TURN_TIMEOUT_S):
            while True:
                message = await ws.receive()
                if message.type == aiohttp.WSMsgType.BINARY:
                    turn.setdefault(
                        "client_ttfa_ms", round((time.monotonic() - spoken_until) * 1000, 1)
                    )
                elif message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    if event["type"] == "transcript":
                        turn["heard"] = True
                    elif event["type"] == "reply":
                        turn["intent"] = event["intent"]
                    elif event["type"] == "timing":
                        turn.update({k: v for k, v in event.items() if k != "type"})
                        return turn
                    elif event["type"] == "state" and event["state"] == "TRANSCRIBING":
                        transcribing = True
                    elif event["type"] == "state" and event["state"] == "WAITING_FOR_NEXT_INPUT":
                        # After transcribing with nothing heard, the server just waits again.
                        # (A WAITING left over from the previous turn's playback is not that.)
                        if transcribing and not turn["heard"]:
                            return None
                else:
                    raise ConnectionError(f"connection closed ({ws.close_code})")
    finally:
        stop.set()
        await silence
        await ws.send_json({"type": "playback_done"})


async def main() -> int:
    parser = argparse.ArgumentParser(description="Voice pipeline latency benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--email", default="divya@careflow.example", help="a seeded patient")
    parser.add_argument("--label", default="run")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--pipeline", choices=["cascade", "live"], default="cascade")
    parser.add_argument("--turns", type=int, default=0, help="0 = every utterance with audio")
    parser.add_argument("--gap", type=float, default=4.0, help="seconds between turns")
    parser.add_argument("--hardware", default="", help="a note stored with the results")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    rows = [
        row
        for row in load_utterances()
        if row["lang"] in args.lang.split(",") and audio_path(row["id"]).exists()
    ]
    rows = rows[: args.turns] if args.turns else rows
    if not rows:
        print("No audio found. Run: python scripts/make_eval_audio.py")
        return 1

    async with httpx.AsyncClient() as client:
        login = await client.post(
            f"{args.url}/api/auth/login",
            json={"email": args.email, "password": settings.seed_default_password},
        )
    login.raise_for_status()
    turns: list[dict[str, Any]] = []
    async with (
        aiohttp.ClientSession() as http,
        http.ws_connect(
            f"{args.url.replace('http', 'ws', 1)}/ws/voice?pipeline={args.pipeline}", max_msg_size=0
        ) as ws,
    ):
        await ws.send_json({"type": "auth", "token": login.json()["access_token"]})
        for index, row in enumerate(rows, 1):
            turn = await run_turn(ws, read_wav(audio_path(row["id"])))
            if turn is None:
                print(f"  {index:>2}/{len(rows)} {row['id']}: not heard")
                continue
            turn.pop("heard")
            turn["utterance"] = row["id"]
            turns.append(turn)
            ttfa = turn["stages"].get("ttfa_ms")
            shown = f"{ttfa / 1000:.1f} s to first audio" if ttfa else "not spoken"
            print(f"  {index:>2}/{len(rows)} {row['id']}: {shown}")
            await asyncio.sleep(args.gap)
        await ws.send_json({"type": "end"})

    if not turns:
        print("No turn completed.")
        return 1
    print()
    print(report(args.label, turns))
    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{args.label.replace(' ', '-').lower()}.json"
        result = {
            "label": args.label,
            "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "pipeline": args.pipeline,
            "config": {
                "whisper_model": settings.whisper_model,
                "whisper_cpu_threads": settings.whisper_cpu_threads,
                "vad_silence_ms": settings.vad_silence_ms,
                "llm_provider": settings.llm_provider,
                "llm_model": settings.gemini_text_model,
                "tts_provider": settings.tts_provider,
                "tts_chunking": settings.tts_chunking,
            },
            "hardware": args.hardware or platform.processor(),
            "summary": summarise(turns),
            "turns": turns,
        }
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
