"""The benchmark's arithmetic and file handling. The benchmark itself needs a live server."""

from pathlib import Path

import pytest
from httpx import AsyncClient

from eval.latency_bench import STAGES, count_by, percentile, report, summarise
from scripts.make_eval_audio import audio_path, load_utterances, read_wav, write_wav
from services.llm import MockLLM
from tests.helpers import make_user
from tests.test_chat import seed_hospital
from tests.voice_helpers import UTTERANCE, FakeSTT, FakeTTS, Wire
from voice import timing
from voice.cascade_pipeline import VoiceSession
from voice.vad import VADConfig, energy_vad


@pytest.mark.parametrize(
    ("values", "percent", "expected"),
    [
        ([5.0], 50, 5.0),
        ([5.0], 95, 5.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.0),
        ([4.0, 1.0, 3.0, 2.0], 50, 2.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 50, 3.0),
        (list(map(float, range(1, 21))), 95, 19.0),
        (list(map(float, range(1, 11))), 95, 10.0),
        (list(map(float, range(1, 101))), 95, 95.0),
    ],
)
def test_percentile_is_nearest_rank(values: list[float], percent: float, expected: float) -> None:
    assert percentile(values, percent) == expected


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


def test_a_percentile_is_always_a_value_that_was_measured() -> None:
    values = [120.0, 4000.0, 900.0]

    assert percentile(values, 50) in values and percentile(values, 95) in values


TURNS = [
    {"stages": {"stt_ms": 1000.0, "ttfa_ms": 5000.0}, "tts_provider": "piper", "used_llm": True},
    {"stages": {"stt_ms": 3000.0, "ttfa_ms": 9000.0}, "tts_provider": "piper", "used_llm": False},
    {"stages": {"stt_ms": 2000.0}, "tts_provider": "", "used_llm": True},  # could not be spoken
]


def test_summary_counts_each_stage_over_the_turns_that_reached_it() -> None:
    summary = summarise(TURNS)

    assert summary["stt_ms"] == {"n": 3, "p50": 2000.0, "p95": 3000.0, "min": 1000.0, "max": 3000.0}
    assert summary["ttfa_ms"]["n"] == 2 and summary["ttfa_ms"]["p50"] == 5000.0
    assert "llm_ms" not in summary


def test_counts_group_missing_values_as_none() -> None:
    assert count_by(TURNS, "tts_provider") == {"piper": 2, "none": 1}


def test_the_report_states_the_turn_count_next_to_the_percentiles() -> None:
    text = report("baseline", TURNS)

    assert "baseline: 3 turns" in text
    assert "Speech to text" in text and "2.00 s" in text and "3.00 s" in text
    assert "answered with the LLM: 2/3" in text


def test_the_bench_knows_every_stage_the_server_measures() -> None:
    assert {stage for stage, _ in STAGES} == set(timing.STAGES)


def test_wav_files_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "clip.wav"

    write_wav(path, UTTERANCE)

    assert read_wav(path) == UTTERANCE


def test_eval_audio_lives_in_the_gitignored_folder() -> None:
    row = load_utterances()[0]

    assert audio_path(row["id"]).parts[-3:] == ("eval", "audio", f"{row['id']}.wav")


async def test_the_timing_event_says_which_path_the_turn_took(client: AsyncClient) -> None:
    await seed_hospital()
    wire = Wire()
    session = VoiceSession(
        await make_user(),
        "bench-session-1",
        wire.send_json,
        wire.send_audio,
        stt=FakeSTT(),
        tts=FakeTTS(),
        probability=energy_vad(),
        vad_config=VADConfig(min_speech_ms=96, silence_ms=200, pre_roll_ms=96),
        llm=MockLLM(),
    )

    await session.receive_audio(UTTERANCE)
    assert session._turn is not None
    await session._turn

    event = wire.of("timing")[0]
    assert event["routed_by"] == "keywords" and event["used_llm"] is False
    assert event["stt_provider"] == "fake-stt" and event["tts_provider"] == "fake-tts"
    assert "text" not in event and "transcript" not in event
