"""Pipeline B with stand-in speech models: detection, turn flow, barge-in, timings."""

import asyncio
import base64
import io
import wave
from typing import Any

import httpx
import numpy as np
import pytest
from httpx import AsyncClient

from agents.guardrails import DISCLAIMERS, SAFE_FALLBACKS
from agents.intent_classifier import IntentClassifier
from models.turn_analytics import TurnAnalytics
from services.llm import MockLLM
from services.tts import GeminiTTS, MockTTS, TTSUnavailable
from tests.helpers import make_user
from tests.test_chat import ScriptedLLM, say, seed_hospital
from tests.voice_helpers import UTTERANCE, FakeSTT, FakeTTS, Wire, frames, silence, tone
from voice import cascade_pipeline
from voice.audio_utils import (
    FRAME_MS,
    FRAME_SAMPLES,
    Framer,
    decode_speech,
    float_to_pcm16,
    pcm16_to_float,
    resample,
)
from voice.cascade_pipeline import VoiceSession, VoiceState, split_sentences
from voice.timing import TurnTimer
from voice.vad import SpeechDetector, VADConfig, energy_vad

CONFIG = VADConfig(min_speech_ms=96, silence_ms=200, pre_roll_ms=96, max_utterance_s=2)

# --- audio helpers ----------------------------------------------------------------------


def test_pcm_round_trip_is_lossless_to_one_step() -> None:
    audio = np.array([0.0, 0.5, -0.5, 0.999, -1.0], dtype=np.float32)

    assert np.allclose(pcm16_to_float(float_to_pcm16(audio)), audio, atol=1 / 32000)


def test_out_of_range_samples_are_clipped_not_wrapped() -> None:
    assert pcm16_to_float(float_to_pcm16(np.array([2.0, -2.0], dtype=np.float32))).tolist() == [
        pytest.approx(1.0, abs=1e-3),
        pytest.approx(-1.0, abs=1e-3),
    ]


@pytest.mark.parametrize(
    ("source", "target"), [(24_000, 16_000), (16_000, 8_000), (16_000, 16_000)]
)
def test_resampling_changes_the_length_in_proportion(source: int, target: int) -> None:
    audio = np.zeros(source, dtype=np.float32)

    assert len(resample(audio, source, target)) == target


def test_framer_cuts_any_chunking_into_whole_frames() -> None:
    data = tone(5 * FRAME_MS)
    framer = Framer()

    got = [
        frame
        for start in range(0, len(data), 333)
        for frame in framer.push(data[start : start + 333])
    ]

    assert len(got) == 5 and all(len(frame) == FRAME_SAMPLES for frame in got)
    assert np.allclose(np.concatenate(got), pcm16_to_float(data))


def test_a_wav_payload_is_decoded_with_its_own_sample_rate() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22_050)
        wav.writeframes(b"\x01\x00" * 10)

    assert decode_speech(buffer.getvalue(), "audio/wav") == (b"\x01\x00" * 10, 22_050)


def test_raw_l16_takes_the_rate_from_the_mime_type() -> None:
    assert decode_speech(b"\x01\x00", "audio/L16;codec=pcm;rate=16000") == (b"\x01\x00", 16_000)
    assert decode_speech(b"\x01\x00", "audio/L16")[1] == 24_000


# --- sentences --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("One short reply.", ["One short reply."]),
        (
            "Your appointment is booked for Monday. Please arrive ten minutes early.",
            ["Your appointment is booked for Monday.", "Please arrive ten minutes early."],
        ),
        (
            "You are booked with Dr. Asha Rao on Monday. Please arrive ten minutes early.",
            ["You are booked with Dr. Asha Rao on Monday.", "Please arrive ten minutes early."],
        ),
        (
            "Yes. It is on the first floor of Block A.",
            ["Yes. It is on the first floor of Block A."],
        ),
        (
            "The department is on the first floor of Block A. OK?",
            ["The department is on the first floor of Block A. OK?"],
        ),
        (
            "आपको जनरल मेडिसिन विभाग में जाना चाहिए। क्या मैं अपॉइंटमेंट बुक कर दूँ?",
            ["आपको जनरल मेडिसिन विभाग में जाना चाहिए।", "क्या मैं अपॉइंटमेंट बुक कर दूँ?"],
        ),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


# --- detector ---------------------------------------------------------------------------


def run_detector(data: bytes, config: VADConfig = CONFIG) -> list[Any]:
    detector = SpeechDetector(energy_vad(), config)
    return [event for frame in frames(data) if (event := detector.feed(frame))]


def test_silence_never_starts_an_utterance() -> None:
    assert run_detector(silence(2000)) == []


def test_a_blip_shorter_than_the_minimum_is_ignored() -> None:
    assert run_detector(silence(6 * FRAME_MS) + tone(2 * FRAME_MS) + silence(640)) == []


def test_speech_then_silence_gives_start_and_end() -> None:
    events = run_detector(silence(300) + tone(15 * FRAME_MS) + silence(400))

    assert [event.kind for event in events] == ["start", "end"]


def test_the_utterance_keeps_the_pre_roll_and_all_the_speech() -> None:
    events = run_detector(silence(10 * FRAME_MS) + tone(15 * FRAME_MS) + silence(400))
    audio = events[1].audio

    spoken = 15 * FRAME_SAMPLES
    assert audio is not None and len(audio) >= spoken
    loud = int(np.sum(np.abs(audio) > 0.01))  # a sine dips under this near its zero crossings
    assert 0.9 * spoken <= loud <= spoken


def test_a_pause_shorter_than_the_silence_window_does_not_end_it() -> None:
    data = tone(10 * FRAME_MS) + silence(3 * FRAME_MS) + tone(10 * FRAME_MS) + silence(400)

    assert [event.kind for event in run_detector(data)] == ["start", "end"]


def test_an_utterance_is_cut_off_at_the_maximum_length() -> None:
    events = run_detector(tone(3000))

    assert [event.kind for event in events][:2] == ["start", "end"]
    assert events[1].audio is not None and len(events[1].audio) <= 2.1 * 16_000


def test_the_end_event_carries_when_speech_actually_stopped() -> None:
    now = [0.0]
    detector = SpeechDetector(energy_vad(), CONFIG, lambda: now[0])
    events = []
    for frame in frames(tone(10 * FRAME_MS) + silence(400)):
        now[0] += FRAME_MS / 1000
        if event := detector.feed(frame):
            events.append(event)

    # Ten frames of speech, then the clock kept running through the silence window
    assert events[-1].last_speech_at == pytest.approx(10 * FRAME_MS / 1000)
    assert now[0] - events[-1].last_speech_at >= CONFIG.silence_ms / 1000


def test_two_utterances_in_a_row_are_both_detected() -> None:
    data = (tone(10 * FRAME_MS) + silence(400)) * 2

    assert [event.kind for event in run_detector(data)] == ["start", "end", "start", "end"]


# --- timer ------------------------------------------------------------------------------


def test_stage_timings_are_differences_between_marks() -> None:
    now = [10.0]
    timer = TurnTimer(10.0, lambda: now[0])
    for name, at in [
        ("vad_end", 10.5),
        ("transcript", 11.5),
        ("routed", 11.6),
        ("reply", 12.6),
        ("first_audio", 13.1),
        ("last_audio", 14.0),
    ]:
        now[0] = at
        timer.mark(name)

    assert timer.stages() == {
        "end_of_speech_ms": 500.0,
        "stt_ms": 1000.0,
        "routing_ms": 100.0,
        "llm_ms": 1000.0,
        "tts_first_chunk_ms": 500.0,
        "ttfa_ms": 3100.0,
        "total_ms": 4000.0,
    }


def test_stages_that_never_finished_are_left_out() -> None:
    timer = TurnTimer(0.0, lambda: 1.0)
    timer.mark("vad_end")

    assert timer.stages() == {"end_of_speech_ms": 1000.0}


# --- a session --------------------------------------------------------------------------


class Rig:
    """A VoiceSession wired to stand-ins, with everything it sends collected."""

    def __init__(self, user: Any, llm: Any = None, chunking: str = "sentence") -> None:
        self.wire, self.stt, self.tts = Wire(), FakeSTT(), FakeTTS()
        self.session = VoiceSession(
            user,
            "voice-session-1",
            self.wire.send_json,
            self.wire.send_audio,
            stt=self.stt,
            tts=self.tts,
            probability=energy_vad(),
            vad_config=CONFIG,
            chunking=chunking,
            llm=llm or MockLLM(),
            classifier=IntentClassifier(),
        )

    async def speak(self, wait: bool = True) -> None:
        await self.session.receive_audio(UTTERANCE)
        if wait and self.session._turn is not None:
            await self.session._turn


@pytest.fixture
async def rig(client: AsyncClient) -> Rig:
    await seed_hospital()
    return Rig(await make_user())


async def test_a_full_turn_goes_from_speech_to_spoken_reply(rig: Rig) -> None:
    await rig.session.start()
    await rig.speak()

    assert (
        rig.wire.states
        == (
            "IDLE LISTENING TRANSCRIBING THINKING TOOL_EXECUTION GENERATING_RESPONSE SPEAKING"
        ).split()
    )
    assert rig.wire.of("transcript")[0]["text"] == "What are the OPD timings?"
    reply = rig.wire.of("reply")[0]
    assert reply["intent"] == "support" and "9 am to 1 pm" in reply["text"]
    assert rig.wire.of("audio_start") == [{"type": "audio_start", "sample_rate": 24_000}]
    assert rig.wire.audio and rig.wire.of("audio_end")
    assert rig.tts.spoken == [reply["text"]]
    assert len(rig.stt.heard) == 1 and len(rig.stt.heard[0]) >= 10 * FRAME_SAMPLES


async def test_events_arrive_in_the_order_the_client_needs(rig: Rig) -> None:
    await rig.speak()

    kinds = [event["type"] for event in rig.wire.events if event["type"] != "state"]
    assert kinds == ["transcript", "reply", "audio_start", "audio_end", "timing"]


async def test_playback_done_moves_on_to_waiting(rig: Rig) -> None:
    await rig.speak()
    assert rig.session.state == VoiceState.SPEAKING

    await rig.session.playback_done()

    assert rig.session.state == VoiceState.WAITING_FOR_NEXT_INPUT


async def test_a_client_that_never_reports_playback_is_timed_out(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cascade_pipeline, "PLAYBACK_GRACE_S", 0.05)

    await rig.speak()
    await asyncio.sleep(0.2)

    assert rig.session.state == VoiceState.WAITING_FOR_NEXT_INPUT


async def test_a_second_turn_works_after_the_first(rig: Rig) -> None:
    await rig.speak()
    await rig.session.playback_done()
    await rig.speak()

    assert len(rig.wire.of("reply")) == 2


async def test_tool_use_is_shown_as_its_own_states(client: AsyncClient) -> None:
    await seed_hospital()
    llm = ScriptedLLM(say("We open at nine."))  # the lookup itself is the tool step
    rig = Rig(await make_user(), llm)

    await rig.speak()

    expected = "LISTENING TRANSCRIBING THINKING TOOL_EXECUTION GENERATING_RESPONSE SPEAKING"
    assert rig.wire.states == expected.split()


async def test_audio_arriving_while_a_turn_is_worked_out_is_ignored(rig: Rig) -> None:
    rig.stt.gate = asyncio.Event()
    await rig.speak(wait=False)

    await rig.session.receive_audio(UTTERANCE * 2)
    rig.stt.gate.set()
    assert rig.session._turn is not None
    await rig.session._turn

    assert len(rig.stt.heard) == 1 and len(rig.wire.of("reply")) == 1


async def test_an_empty_transcript_goes_back_to_waiting_without_a_reply(rig: Rig) -> None:
    rig.stt.text = ""

    await rig.speak()

    assert rig.wire.of("reply") == [] and rig.wire.audio == []
    assert rig.session.state == VoiceState.WAITING_FOR_NEXT_INPUT


async def test_an_stt_failure_is_reported_and_the_session_carries_on(rig: Rig) -> None:
    rig.stt.fail = True
    await rig.speak()

    assert rig.wire.of("error") and rig.wire.states[-2:] == ["ERROR", "WAITING_FOR_NEXT_INPUT"]

    rig.stt.fail = False
    await rig.speak()
    assert len(rig.wire.of("reply")) == 1


async def test_when_tts_fails_the_reply_is_still_delivered_as_text(rig: Rig) -> None:
    rig.tts.fail = True

    await rig.speak()

    assert rig.wire.of("reply") and rig.wire.of("audio_unavailable")
    assert rig.wire.audio == [] and rig.wire.of("audio_end") == []
    assert rig.session.state == VoiceState.WAITING_FOR_NEXT_INPUT


async def test_each_sentence_is_synthesised_and_sent_separately(client: AsyncClient) -> None:
    text = "Your appointment is booked for Monday. Please arrive ten minutes early."
    rig = Rig(await make_user(), ScriptedLLM(say(text)))

    await rig.speak()

    assert rig.tts.spoken == [
        "Your appointment is booked for Monday.",
        "Please arrive ten minutes early.",
    ]
    assert len(rig.wire.audio) == 2 and len(rig.wire.of("audio_start")) == 1


async def test_reply_chunking_sends_the_whole_reply_in_one_request(client: AsyncClient) -> None:
    text = "Your appointment is booked for Monday. Please arrive ten minutes early."
    rig = Rig(await make_user(), ScriptedLLM(say(text)), chunking="reply")

    await rig.speak()

    assert rig.tts.spoken == [text]


# --- barge-in ---------------------------------------------------------------------------


async def test_speaking_over_the_reply_stops_it(rig: Rig) -> None:
    rig.tts.gate = asyncio.Event()  # the reply is still being synthesised
    await rig.speak(wait=False)
    await rig.tts.started.wait()
    assert rig.session.state == VoiceState.SPEAKING

    await rig.session.receive_audio(tone(12 * FRAME_MS))

    assert rig.wire.of("stop_audio") == [{"type": "stop_audio"}]
    assert rig.wire.audio == [] and rig.wire.of("audio_end") == []
    assert rig.session.state == VoiceState.LISTENING
    assert rig.wire.of("reply"), "the reply text was already delivered"


async def test_speaking_during_client_playback_also_stops_it(rig: Rig) -> None:
    await rig.speak()
    assert rig.wire.of("audio_end") and rig.session.state == VoiceState.SPEAKING

    await rig.session.receive_audio(tone(12 * FRAME_MS))

    assert rig.wire.of("stop_audio") and rig.session.state == VoiceState.LISTENING


async def test_the_interrupting_utterance_becomes_the_next_turn(rig: Rig) -> None:
    rig.tts.gate = asyncio.Event()
    await rig.speak(wait=False)
    await rig.tts.started.wait()
    rig.tts.gate = None

    await rig.session.receive_audio(UTTERANCE)
    assert rig.session._turn is not None
    await rig.session._turn

    assert len(rig.wire.of("reply")) == 2 and len(rig.stt.heard) == 2


async def test_a_short_noise_while_speaking_does_not_interrupt(rig: Rig) -> None:
    await rig.speak()

    # Long enough to start a normal utterance, too short to count as barge-in
    await rig.session.receive_audio(tone(4 * FRAME_MS) + silence(400))

    assert rig.wire.of("stop_audio") == [] and rig.session.state == VoiceState.SPEAKING


async def test_barge_in_does_not_undo_what_the_turn_already_did(rig: Rig) -> None:
    rig.tts.gate = asyncio.Event()
    await rig.speak(wait=False)
    await rig.tts.started.wait()

    await rig.session.receive_audio(tone(12 * FRAME_MS))

    analytics = await TurnAnalytics.find_all().to_list()
    assert [a.barged_in for a in analytics] == [True]
    assert analytics[0].spoken is False


# --- guardrails -------------------------------------------------------------------------


@pytest.mark.guardrail
async def test_the_triage_disclaimer_is_spoken_not_only_shown(client: AsyncClient) -> None:
    rig = Rig(await make_user())
    rig.stt.text = "I have a fever and cough"

    await rig.speak()

    assert rig.wire.of("reply")[0]["disclaimer"] == DISCLAIMERS["en"]
    assert DISCLAIMERS["en"] in " ".join(rig.tts.spoken)


@pytest.mark.guardrail
async def test_a_diagnosis_from_the_model_is_never_spoken(client: AsyncClient) -> None:
    rig = Rig(await make_user(), ScriptedLLM(say("You have pneumonia. Take amoxicillin.")))
    rig.stt.text = "I have a bad cough"

    await rig.speak()

    spoken = " ".join(rig.tts.spoken)
    assert "pneumonia" not in spoken and "amoxicillin" not in spoken
    assert SAFE_FALLBACKS["en"] in spoken
    assert rig.wire.of("reply")[0]["blocked"] is True


@pytest.mark.guardrail
async def test_analytics_keep_timings_and_labels_but_nothing_that_was_said(rig: Rig) -> None:
    await rig.speak()

    record = (await TurnAnalytics.find_all().to_list())[0]
    stored = str(record.model_dump())
    assert "OPD timings" not in stored and "9 am to 1 pm" not in stored
    assert "user_id" not in record.model_dump() and "session" not in stored
    assert record.pipeline == "cascade" and record.intent == "support"
    assert record.stt_provider == "fake-stt" and record.tts_provider == "fake-tts"
    assert set(record.stages) == {
        "end_of_speech_ms", "stt_ms", "routing_ms", "llm_ms", "tts_first_chunk_ms", "ttfa_ms",
        "total_ms",
    }  # fmt: skip
    assert rig.wire.of("timing")[0]["stages"] == record.stages


async def test_time_to_first_audio_includes_the_silence_the_vad_waited_for(rig: Rig) -> None:
    await rig.speak()

    stages = rig.wire.of("timing")[0]["stages"]
    assert stages["ttfa_ms"] >= stages["end_of_speech_ms"] >= 0
    assert stages["total_ms"] >= stages["ttfa_ms"]


# --- providers --------------------------------------------------------------------------


def wav_bytes(rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x01\x00" * rate)
    return buffer.getvalue()


def audio_response(data: bytes, mime: str) -> httpx.Response:
    inline = {"mimeType": mime, "data": base64.b64encode(data).decode()}
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"inlineData": inline}]}}]}
    )


async def test_gemini_tts_sends_the_text_bare_and_decodes_wav() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        seen["key_in_url"] = "k-secret" in str(request.url)
        return audio_response(wav_bytes(), "audio/wav")

    speech = await GeminiTTS("k-secret", "tts-model", httpx.MockTransport(handler)).synthesize(
        "Hi."
    )

    assert seen["body"]["contents"] == [{"parts": [{"text": "Hi."}]}]
    assert seen["body"]["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert seen["key_in_url"] is False
    assert speech.sample_rate == 24_000 and speech.seconds == pytest.approx(1.0)


async def test_gemini_tts_accepts_raw_l16() -> None:
    transport = httpx.MockTransport(
        lambda r: audio_response(b"\x01\x00" * 16_000, "audio/L16;rate=16000")
    )

    speech = await GeminiTTS("k", "m", transport).synthesize("Hi.")

    assert speech.sample_rate == 16_000 and speech.seconds == pytest.approx(1.0)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, text="quota k-secret"),
        httpx.Response(200, json={"candidates": []}),
        httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "no audio"}]}}]}),
        audio_response(b"", "audio/L16"),
    ],
)
async def test_gemini_tts_failures_become_tts_unavailable(response: httpx.Response) -> None:
    tts = GeminiTTS("k-secret", "m", httpx.MockTransport(lambda request: response))

    with pytest.raises(TTSUnavailable) as raised:
        await tts.synthesize("Hi.")

    assert "k-secret" not in str(raised.value)


async def test_gemini_tts_without_a_key_or_model_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    for key, model in (("", "m"), ("k", "")):
        with pytest.raises(TTSUnavailable):
            await GeminiTTS(key, model, httpx.MockTransport(handler)).synthesize("Hi.")


async def test_mock_tts_returns_silence_of_a_sensible_length() -> None:
    speech = await MockTTS().synthesize("Hello there")

    assert speech.sample_rate == 24_000 and 0.4 < speech.seconds < 0.7
    assert set(speech.pcm16) == {0}
