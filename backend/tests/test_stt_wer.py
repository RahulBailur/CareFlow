"""The phone-line simulation and the scoring used by eval/stt_wer.py."""

from typing import Any

import numpy as np
import pytest

from eval.stt_wer import (
    CONDITIONS,
    band_pass,
    edit_distance,
    errors,
    mu_law,
    normalise,
    phone_line,
    report,
    summarise,
)
from services.stt import FasterWhisperSTT, Transcript
from voice.audio_utils import SAMPLE_RATE


def tone(hz: float, seconds: float = 1.0, rate: int = SAMPLE_RATE) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * hz * np.arange(int(rate * seconds)) / rate)).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    middle = audio[len(audio) // 4 : -len(audio) // 4]  # away from the filter's edges
    return float(np.sqrt(np.mean(np.square(middle))))


@pytest.mark.parametrize("hz", [500, 1000, 3000])
def test_the_phone_band_lets_speech_frequencies_through(hz: float) -> None:
    assert rms(band_pass(tone(hz), SAMPLE_RATE, 300, 3400)) > 0.8 * rms(tone(hz))


@pytest.mark.parametrize("hz", [60, 5000, 7000])
def test_the_phone_band_removes_what_a_phone_line_cannot_carry(hz: float) -> None:
    assert rms(band_pass(tone(hz), SAMPLE_RATE, 300, 3400)) < 0.1 * rms(tone(hz))


def test_mu_law_is_close_for_speech_levels_but_not_lossless() -> None:
    audio = np.linspace(-0.9, 0.9, 1001, dtype=np.float32)

    back = mu_law(audio)

    assert np.max(np.abs(back - audio)) < 0.03
    assert not np.array_equal(back, audio)
    assert len(np.unique(np.round(back, 6))) <= 256  # eight bits


def test_phone_audio_comes_back_at_16_khz_and_the_same_length() -> None:
    audio = tone(1000, seconds=2.0)

    for codec in (False, True):
        assert abs(len(phone_line(audio, codec)) - len(audio)) <= 2


def test_phone_audio_has_lost_its_high_frequencies() -> None:
    high = tone(6000)

    assert rms(phone_line(high, codec=False)) < 0.1 * rms(high)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("What are the OPD timings?", "what are the opd timings"),
        ("Dr. Asha Rao's slot", "dr asha rao s slot"),
        ("ओपीडी का समय क्या है?", "ओपीडी का समय क्या है"),  # vowel signs survive
        ("ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ರದ್ದು", "ಅಪಾಯಿಂಟ್ಮೆಂಟ್ ರದ್ದು"),  # joiner or not, one word
    ],
)
def test_normalise(raw: str, expected: str) -> None:
    assert normalise(raw) == normalise(expected)


def test_edit_distance() -> None:
    assert edit_distance([], []) == 0
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1  # substitution
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1  # deletion
    assert edit_distance(["a", "c"], ["a", "b", "c"]) == 1  # insertion
    assert edit_distance(["a", "b"], []) == 2


def test_errors_counts_words_and_characters() -> None:
    words_wrong, words, chars_wrong, chars = errors(
        "What are the OPD timings?", "what are the OPD timing"
    )

    assert (words_wrong, words) == (1, 5)
    assert (chars_wrong, chars) == (1, len("whataretheopdtimings"))


def test_a_perfect_transcript_scores_zero_whatever_the_punctuation() -> None:
    assert errors("Is the hospital open on Sunday?", "is the hospital open on sunday")[0] == 0


def test_summary_and_report_cover_every_condition_and_language() -> None:
    rows = [
        {"lang": "en", "errors": {c: (1, 10, 2, 40) for c in CONDITIONS}},
        {"lang": "hi", "errors": {c: (0, 5, 0, 20) for c in CONDITIONS}},
    ]

    summary = summarise(rows)

    assert summary["en"]["16 kHz"] == {"wer": 0.1, "cer": 0.05, "n": 1}
    assert summary["all"]["8 kHz G.711"]["wer"] == pytest.approx(1 / 15)
    text = report(summary)
    assert all(condition in text for condition in CONDITIONS)
    assert "en" in text and "hi" in text and "all" in text


class Recorder:
    """Stands in for a Whisper model inside FasterWhisperSTT: says which one was used."""

    def __init__(self, name: str, heard: str) -> None:
        self.name, self.heard = name, heard


def stt_with(main: str, second: str | None, monkeypatch: pytest.MonkeyPatch) -> Any:
    stt = FasterWhisperSTT("base", "int8", multilingual_model=second or "")
    stt._model = Recorder("base", main)
    stt._multilingual = Recorder("small", "") if second else None

    def fake(model: Recorder, audio: Any, language: str | None) -> Transcript:
        heard = language or model.heard
        return Transcript(f"{model.name}:{heard}", heard)

    monkeypatch.setattr(stt, "_with", fake)
    return stt


@pytest.mark.parametrize(
    ("heard", "second", "expected"),
    [
        ("en", "small", "base:en"),  # English stays on the fast model
        ("hi", "small", "small:hi"),  # Hindi goes to the model that writes Devanagari
        ("kn", "small", "small:kn"),
        ("hi", None, "base:hi"),  # with no second model, the main one does everything
    ],
)
def test_hindi_and_kannada_go_to_the_multilingual_model(
    heard: str, second: str | None, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stt = stt_with(heard, second, monkeypatch)

    assert stt._transcribe(np.zeros(16_000, np.float32)).text == expected


def test_the_provider_name_says_which_models_are_in_use() -> None:
    assert FasterWhisperSTT("base", "int8", multilingual_model="small").name == (
        "faster-whisper:base+small"
    )
    assert FasterWhisperSTT("small", "int8", multilingual_model="small").name == (
        "faster-whisper:small"
    )
    assert FasterWhisperSTT("base", "int8").name == "faster-whisper:base"
