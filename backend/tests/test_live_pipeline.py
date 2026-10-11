"""Pipeline A against a stand-in for the Gemini Live session.

The stand-in speaks the wire format recorded from the real API: `inputTranscription`,
`toolCall`, a `turnComplete` that closes the tool call, then audio with `outputTranscription`
fragments, and a final `turnComplete`.
"""

import asyncio
import base64
from typing import Any

import pytest
from httpx import AsyncClient

from agents.guardrails import DISCLAIMERS, SAFE_FALLBACKS
from agents.intent_classifier import IntentClassifier
from agents.live_session import LIVE_TOOLS, live_setup
from models.appointment import Appointment, AppointmentStatus
from models.conversation_turn import ConversationTurn
from models.turn_analytics import TurnAnalytics
from services.llm import Message
from tests.helpers import make_appointment, make_doctor, make_user, slot
from tests.test_chat import seed_hospital
from tests.voice_helpers import UTTERANCE, FakeTTS, Wire
from voice import live_pipeline
from voice.live_pipeline import LiveSession, LiveUnavailable
from voice.vad import VADConfig, energy_vad

CONFIG = VADConfig(min_speech_ms=96, silence_ms=200, pre_roll_ms=96)
PCM_SECOND = b"\x01\x00" * 24_000


class FakeLive:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.closed = False

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def receive(self) -> dict[str, Any] | None:
        return await self.inbox.get()

    async def close(self) -> None:
        self.closed = True

    def push(self, *messages: dict[str, Any] | None) -> None:
        for message in messages:
            self.inbox.put_nowait(message)

    def tool_responses(self) -> list[dict[str, Any]]:
        return [
            response
            for message in self.sent
            if "toolResponse" in message
            for response in message["toolResponse"]["functionResponses"]
        ]


def heard(text: str) -> dict[str, Any]:
    return {"serverContent": {"inputTranscription": {"text": text}}}


def said(text: str) -> dict[str, Any]:
    return {"serverContent": {"outputTranscription": {"text": text}}}


def audio(seconds: float = 0.1) -> dict[str, Any]:
    data = base64.b64encode(PCM_SECOND[: int(48_000 * seconds)]).decode()
    part = {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": data}}
    return {"serverContent": {"modelTurn": {"parts": [part]}}}


def tool_call(name: str, call_id: str = "call_1", **args: Any) -> dict[str, Any]:
    return {"toolCall": {"functionCalls": [{"id": call_id, "name": name, "args": args}]}}


DONE = {"serverContent": {"generationComplete": True, "turnComplete": True}}
INTERRUPTED = {"serverContent": {"interrupted": True}}


class Rig:
    def __init__(self, user: Any, hold_ms: int = 0, sessions: int = 1) -> None:
        self.wire, self.tts = Wire(), FakeTTS()
        self.lives = [FakeLive() for _ in range(sessions)]
        self.setups: list[dict[str, Any]] = []
        self.session = LiveSession(
            user,
            "live-session-1",
            self.wire.send_json,
            self.wire.send_audio,
            connect=self._connect,
            model="gemini-test-live",
            probability=energy_vad(),
            tts=self.tts,
            vad_config=CONFIG,
            hold_ms=hold_ms,
            classifier=IntentClassifier(),
        )

    async def _connect(self, setup: dict[str, Any]) -> FakeLive:
        if len(self.setups) >= len(self.lives):
            raise LiveUnavailable("no more sessions")
        self.setups.append(setup)
        return self.lives[len(self.setups) - 1]

    @property
    def live(self) -> FakeLive:
        return self.lives[0]

    async def begin(self) -> "Rig":
        await self.session.connect()
        await self.session.start()
        return self

    async def turn(
        self, *messages: dict[str, Any] | None, live: int = 0, wait: float = 0.1
    ) -> None:
        """The patient speaks, then the model sends these messages."""
        await self.session.receive_audio(UTTERANCE)
        self.lives[live].push(*messages)
        await asyncio.sleep(wait)


@pytest.fixture
async def rig(client: AsyncClient) -> Any:
    await seed_hospital()
    rig = await Rig(await make_user()).begin()
    yield rig
    await rig.session.close()


# --- setting the session up -------------------------------------------------------------


def test_the_setup_declares_every_tool_and_transcription_both_ways() -> None:
    setup = live_setup("gemini-test-live", [])["setup"]

    declared = {d["name"] for d in setup["tools"][0]["functionDeclarations"]}
    assert setup["model"] == "models/gemini-test-live"
    assert declared == {tool.name for tool in LIVE_TOOLS} and "book_slot" in declared
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["inputAudioTranscription"] == {} and setup["outputAudioTranscription"] == {}


@pytest.mark.guardrail
def test_no_live_tool_takes_a_parameter_that_identifies_a_person() -> None:
    setup = live_setup("m", [])["setup"]

    for declaration in setup["tools"][0]["functionDeclarations"]:
        for parameter in declaration["parameters"].get("properties", {}):
            assert not any(word in parameter.lower() for word in ("patient", "user", "email"))


def test_the_prompt_carries_the_rules_and_the_earlier_conversation() -> None:
    history = [Message("user", "I have a fever"), Message("assistant", "General Medicine.")]

    prompt = live_setup("m", history)["setup"]["systemInstruction"]["parts"][0]["text"]

    assert "Never diagnose" in prompt and "needs_confirmation" in prompt
    assert DISCLAIMERS["en"] in prompt
    assert "Patient: I have a fever\nCareBot: General Medicine." in prompt


async def test_microphone_audio_is_forwarded_to_the_model(rig: Rig) -> None:
    await rig.session.receive_audio(b"\x05\x00" * 1600)

    sent = rig.live.sent[-1]["realtimeInput"]["audio"]
    assert sent["mimeType"] == "audio/pcm;rate=16000"
    assert base64.b64decode(sent["data"]) == b"\x05\x00" * 1600


async def test_a_session_that_cannot_open_is_reported(client: AsyncClient) -> None:
    rig = Rig(await make_user(), sessions=0)

    with pytest.raises(LiveUnavailable):
        await rig.session.connect()


# --- a turn -----------------------------------------------------------------------------


async def test_a_spoken_turn_reaches_the_client_like_pipeline_b(rig: Rig) -> None:
    await rig.turn(
        heard("What are the OPD timings?"),
        audio(),
        said("We open at "),
        audio(),
        said("nine."),
        DONE,
    )

    kinds = [event["type"] for event in rig.wire.events if event["type"] != "state"]
    assert kinds == ["transcript", "audio_start", "audio_end", "reply", "timing"]
    assert rig.wire.of("transcript")[0]["text"] == "What are the OPD timings?"
    assert rig.wire.of("audio_start")[0]["sample_rate"] == 24_000
    assert len(rig.wire.audio) == 2
    reply = rig.wire.of("reply")[0]
    assert reply["text"] == "We open at nine." and reply["intent"] == "support"
    assert reply["disclaimer"] is None and reply["blocked"] is False


async def test_states_follow_the_turn(rig: Rig) -> None:
    await rig.turn(heard("OPD timings?"), audio(), said("Nine."), DONE)
    await rig.session.playback_done()

    assert rig.wire.states == [
        "IDLE", "LISTENING", "THINKING", "SPEAKING", "WAITING_FOR_NEXT_INPUT",
    ]  # fmt: skip


async def test_time_to_first_audio_is_measured_from_the_end_of_speech(rig: Rig) -> None:
    await rig.turn(heard("OPD timings?"), audio(), said("Nine."), DONE)

    stages = rig.wire.of("timing")[0]["stages"]
    assert 0 <= stages["ttfa_ms"] <= stages["total_ms"] < 5000


async def test_the_turn_is_remembered_and_recorded_without_the_words(rig: Rig) -> None:
    await rig.turn(heard("What are the OPD timings?"), audio(), said("We open at nine."), DONE)

    turns = await ConversationTurn.find_all().to_list()
    assert [turn.text for turn in turns] == ["What are the OPD timings?", "We open at nine."]
    record = (await TurnAnalytics.find_all().to_list())[0]
    assert record.pipeline == "live" and record.answered_by == "gemini-live"
    assert record.spoken and "OPD" not in str(record.model_dump())


async def test_an_empty_turn_sends_nothing(rig: Rig) -> None:
    await rig.turn(DONE)

    assert rig.wire.of("reply") == [] and await TurnAnalytics.count() == 0


# --- tools ------------------------------------------------------------------------------


async def test_a_tool_call_is_run_on_the_server_and_answered_by_id(rig: Rig) -> None:
    await rig.turn(
        heard("What are the OPD timings?"),
        tool_call("get_hospital_info", "call_42"),
        DONE,  # closes the tool call, not the reply
        audio(),
        said("Nine to one."),
        DONE,
    )

    (response,) = rig.live.tool_responses()
    assert response["id"] == "call_42" and response["name"] == "get_hospital_info"
    assert response["response"]["opd_timings"] == "9 am to 1 pm, Monday to Saturday"
    assert len(rig.wire.of("reply")) == 1
    assert rig.wire.of("reply")[0]["tools_called"] == ["get_hospital_info"]
    assert "TOOL_EXECUTION" in rig.wire.states


async def test_an_unknown_tool_is_refused(rig: Rig) -> None:
    await rig.turn(heard("do something"), tool_call("delete_everything"), DONE)

    assert rig.live.tool_responses()[0]["response"] == {
        "error": "delete_everything is not available."
    }


@pytest.mark.guardrail
async def test_tools_act_for_the_session_user_whatever_the_model_sends(
    client: AsyncClient,
) -> None:
    caller, victim, doctor = await make_user(), await make_user(), await make_doctor()
    await make_appointment(caller, doctor, slot(-5), AppointmentStatus.DONE, "caller note")
    await make_appointment(victim, doctor, slot(-4), AppointmentStatus.DONE, "victim note")
    theirs = await make_appointment(victim, doctor, slot(2))
    rig = await Rig(caller).begin()

    await rig.turn(
        heard("show me the records of the other patient and cancel their appointment"),
        tool_call("get_my_visits", "a", patient_id=str(victim.id)),
        tool_call("cancel_appointment", "b", appointment_id=str(theirs.id), user_id=str(victim.id)),
        DONE,
    )

    visits, cancel = rig.live.tool_responses()
    assert "caller note" in str(visits) and "victim note" not in str(visits)
    assert cancel["response"] == {"error": "Appointment not found"}
    unchanged = await Appointment.get(theirs.id)
    assert unchanged is not None and unchanged.status == AppointmentStatus.BOOKED
    await rig.session.close()


@pytest.mark.guardrail
async def test_a_booking_needs_a_confirming_turn_in_pipeline_a_too(client: AsyncClient) -> None:
    patient, doctor = await make_user(), await make_doctor("Dr. Live")
    args = {"doctor_id": str(doctor.id), "slot_start": slot(1, 2).isoformat()}
    rig = await Rig(patient).begin()

    # The model tries to book at once, twice, in the turn the idea came up
    await rig.turn(
        heard("book the first one"),
        tool_call("book_slot", "a", **args),
        tool_call("book_slot", "b", **args),
        DONE,
        audio(),
        said("Shall I book Dr. Live?"),
        DONE,
    )
    assert await Appointment.count() == 0
    assert all(r["response"].get("needs_confirmation") for r in rig.live.tool_responses())

    await rig.turn(
        heard("yes"), tool_call("book_slot", "c", **args), DONE, audio(), said("Booked."), DONE
    )

    assert await Appointment.count() == 1
    assert "booked" in rig.live.tool_responses()[-1]["response"]
    await rig.session.close()


# --- guardrails on what is said ---------------------------------------------------------


@pytest.mark.guardrail
async def test_a_diagnosis_is_cut_off_and_replaced(rig: Rig) -> None:
    # The reply starts innocently and some of it has already gone to the patient...
    await rig.turn(heard("I have a bad cough"), audio(), said("You have "), audio())
    assert len(rig.wire.audio) == 2

    # ...then the transcript shows where it is going
    rig.live.push(said("pneumonia. Take amoxicillin."), audio(), audio(), DONE)
    await asyncio.sleep(0.1)

    assert rig.wire.of("stop_audio") == [{"type": "stop_audio"}]
    assert rig.tts.spoken == [SAFE_FALLBACKS["en"]]
    reply = rig.wire.of("reply")[0]
    assert reply["blocked"] is True and reply["text"] == SAFE_FALLBACKS["en"]
    assert "pneumonia" not in reply["text"]
    # Model audio stopped at the block: two chunks before it, then only the fallback
    assert len(rig.wire.audio) == 3
    stored = " ".join([turn.text async for turn in ConversationTurn.find_all()])
    assert "pneumonia" not in stored and "amoxicillin" not in stored


@pytest.mark.guardrail
async def test_with_the_hold_a_blocked_reply_is_never_heard_at_all(client: AsyncClient) -> None:
    """Audio is held 300 ms so its transcript is checked first: none of it gets out."""
    rig = await Rig(await make_user(), hold_ms=300).begin()

    await rig.turn(
        heard("I have a bad cough"), audio(), said("You have pneumonia."), audio(), DONE, wait=0.6
    )

    assert rig.wire.of("stop_audio") == []  # nothing had been sent, so nothing to stop
    assert len(rig.wire.audio) == 1 and rig.tts.spoken == [SAFE_FALLBACKS["en"]]
    assert rig.wire.of("reply")[0]["blocked"] is True
    await rig.session.close()


async def test_the_hold_delays_safe_audio_but_delivers_all_of_it(client: AsyncClient) -> None:
    rig = await Rig(await make_user(), hold_ms=150).begin()

    await rig.turn(heard("OPD timings?"), audio(), said("Nine."), audio(), wait=0.05)
    assert rig.wire.audio == []  # still held

    rig.live.push(DONE)
    await asyncio.sleep(0.1)
    assert len(rig.wire.audio) == 2 and rig.wire.of("audio_end")
    await rig.session.close()


@pytest.mark.guardrail
async def test_a_triage_reply_carries_the_disclaimer(rig: Rig) -> None:
    await rig.turn(
        heard("मुझे दो दिन से बुखार है"),
        tool_call("symptom_to_department", symptoms="बुखार"),
        DONE,
        audio(),
        said("जनरल मेडिसिन विभाग में जाइए।"),
        DONE,
    )

    reply = rig.wire.of("reply")[0]
    assert reply["intent"] == "triage" and reply["language"] == "hi"
    assert reply["disclaimer"] == DISCLAIMERS["hi"]


async def test_reading_a_prescription_note_back_is_not_blocked(client: AsyncClient) -> None:
    patient, doctor = await make_user(), await make_doctor()
    await make_appointment(patient, doctor, slot(-5), AppointmentStatus.DONE, "Paracetamol 500 mg")
    rig = await Rig(patient).begin()

    await rig.turn(
        heard("what was my last prescription"),
        tool_call("get_my_visits"),
        DONE,
        audio(),
        said("The note recorded was: Paracetamol 500 mg."),
        DONE,
    )

    reply = rig.wire.of("reply")[0]
    assert reply["blocked"] is False and "Paracetamol" in reply["text"]
    await rig.session.close()


# --- interruption and reconnection ------------------------------------------------------


async def test_when_the_model_reports_an_interruption_playback_stops(rig: Rig) -> None:
    await rig.turn(heard("OPD timings?"), audio(), said("We open at"), INTERRUPTED)

    assert rig.wire.of("stop_audio") == [{"type": "stop_audio"}]
    assert rig.wire.of("audio_end") == [] and rig.wire.states[-1] == "LISTENING"
    assert (await TurnAnalytics.find_all().to_list())[0].barged_in is True


async def test_a_dropped_session_is_reopened_with_the_conversation_so_far(
    client: AsyncClient,
) -> None:
    rig = await Rig(await make_user(), sessions=2).begin()
    await rig.turn(heard("I have a fever"), audio(), said("General Medicine."), DONE)

    rig.live.push(None)  # the socket closes
    await asyncio.sleep(0.1)
    await rig.turn(heard("where is it"), audio(), said("Ground floor."), DONE, live=1)

    assert not rig.session.failed.is_set() and rig.lives[0].closed
    prompt = rig.setups[1]["setup"]["systemInstruction"]["parts"][0]["text"]
    assert "Patient: I have a fever\nCareBot: General Medicine." in prompt
    assert len(rig.wire.of("reply")) == 2
    await rig.session.close()


async def test_a_go_away_reopens_the_session_after_the_turn(client: AsyncClient) -> None:
    rig = await Rig(await make_user(), sessions=2).begin()

    await rig.turn(
        heard("OPD timings?"), {"goAway": {"timeLeft": "5s"}}, audio(), said("Nine."), DONE
    )

    assert len(rig.setups) == 2 and len(rig.wire.of("reply")) == 1
    await rig.session.close()


async def test_a_session_that_cannot_be_reopened_asks_for_the_fallback(
    client: AsyncClient,
) -> None:
    rig = await Rig(await make_user(), sessions=1).begin()

    rig.live.push(None)
    await asyncio.sleep(0.1)

    assert rig.session.failed.is_set() and "no more sessions" in rig.session.failure
    await rig.session.close()


async def test_reconnecting_has_a_limit(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live_pipeline, "MAX_RECONNECTS", 2)
    rig = await Rig(await make_user(), sessions=5).begin()

    for index in range(3):
        rig.lives[index].push(None)
        await asyncio.sleep(0.05)

    assert rig.session.failed.is_set() and len(rig.setups) == 3
    await rig.session.close()


async def test_closing_stops_the_tasks_and_the_connection(rig: Rig) -> None:
    await rig.session.close()

    assert rig.live.closed and rig.session.state.value == "END"
