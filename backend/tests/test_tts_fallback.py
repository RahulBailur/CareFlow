"""Which voice speaks: the fallback order, the cool-down, and the language each can handle."""

import asyncio

import pytest
from httpx import AsyncClient

from agents.intent_classifier import IntentClassifier
from models.turn_analytics import TurnAnalytics
from services.llm import MockLLM
from services.tts import COOLDOWN_S, FallbackTTS, PiperTTS, Speech, TTSUnavailable
from tests.helpers import make_user
from tests.test_chat import ScriptedLLM, say, seed_hospital
from tests.voice_helpers import UTTERANCE, FakeSTT, FakeTTS, Wire
from voice.cascade_pipeline import VoiceSession
from voice.vad import VADConfig, energy_vad


class Voice:
    def __init__(self, name: str, error: str | None = None) -> None:
        self.name, self.error, self.calls = name, error, 0

    async def synthesize(self, text: str, language: str = "en") -> Speech:
        self.calls += 1
        if self.error:
            raise TTSUnavailable(self.error)
        return Speech(b"\x01\x00" * 240, 24_000, self.name)

    def warm_up(self) -> None:
        if self.error == "broken":
            raise RuntimeError("cannot load")


async def test_the_first_provider_speaks_when_it_can() -> None:
    first, second = Voice("first"), Voice("second")

    speech = await FallbackTTS([first, second]).synthesize("Hello.")

    assert speech.provider == "first" and second.calls == 0


async def test_the_next_provider_takes_over_when_the_first_fails() -> None:
    first, second = Voice("first", "HTTP 429"), Voice("second")

    speech = await FallbackTTS([first, second]).synthesize("Hello.")

    assert speech.provider == "second"


async def test_a_provider_that_failed_is_skipped_until_it_has_cooled_down() -> None:
    now = [0.0]
    first, second = Voice("first", "HTTP 429"), Voice("second")
    tts = FallbackTTS([first, second], clock=lambda: now[0])

    for _ in range(3):
        await tts.synthesize("Hello.")
    assert first.calls == 1  # not asked again for every sentence

    now[0] = COOLDOWN_S + 1
    first.error = None
    assert (await tts.synthesize("Hello.")).provider == "first"


async def test_a_voice_refusing_a_language_is_not_put_on_cool_down() -> None:
    piper_like, other = Voice("piper", "The Piper voice is English only, not hi"), Voice("other")
    tts = FallbackTTS([piper_like, other])

    await tts.synthesize("नमस्ते", "hi")
    await tts.synthesize("नमस्ते", "hi")

    assert piper_like.calls == 2  # still asked each time: English may come next


async def test_when_every_provider_fails_the_reasons_are_reported() -> None:
    tts = FallbackTTS([Voice("a", "HTTP 429"), Voice("b", "no model")])

    with pytest.raises(TTSUnavailable) as raised:
        await tts.synthesize("Hello.")

    assert "a: HTTP 429" in str(raised.value) and "b: no model" in str(raised.value)


def test_one_provider_failing_to_warm_up_does_not_stop_the_others() -> None:
    good = Voice("good")
    warmed: list[str] = []
    good.warm_up = lambda: warmed.append("good")  # type: ignore[method-assign]

    FallbackTTS([Voice("bad", "broken"), good]).warm_up()

    assert warmed == ["good"]


@pytest.mark.parametrize("language", ["hi", "kn"])
async def test_the_english_piper_voice_refuses_other_languages(language: str) -> None:
    """Checked before the model is loaded, so this needs no Piper install."""
    with pytest.raises(TTSUnavailable, match="English only"):
        await PiperTTS("en_US-lessac-medium").synthesize("नमस्ते", language)


def session(user: object, tts: object, llm: object, stt_text: str) -> tuple[VoiceSession, Wire]:
    wire = Wire()
    return (
        VoiceSession(
            user,  # type: ignore[arg-type]
            "voice-session-1",
            wire.send_json,
            wire.send_audio,
            stt=FakeSTT(stt_text),
            tts=tts,  # type: ignore[arg-type]
            probability=energy_vad(),
            vad_config=VADConfig(min_speech_ms=96, silence_ms=200, pre_roll_ms=96),
            llm=llm,  # type: ignore[arg-type]
            classifier=IntentClassifier(),
        ),
        wire,
    )


async def speak(voice_session: VoiceSession) -> None:
    await voice_session.receive_audio(UTTERANCE)
    assert voice_session._turn is not None
    await voice_session._turn


async def test_the_voice_is_told_the_language_of_the_reply(client: AsyncClient) -> None:
    tts = FakeTTS()
    voice_session, wire = session(
        await make_user(), tts, ScriptedLLM(say("ओपीडी सुबह नौ बजे खुलता है।")), "ओपीडी का समय क्या है?"
    )

    await speak(voice_session)

    assert tts.languages == ["hi"] and wire.of("reply")[0]["language"] == "hi"


async def test_a_rule_based_reply_to_a_hindi_question_is_spoken_as_english(
    client: AsyncClient,
) -> None:
    await seed_hospital()
    tts = FakeTTS()
    voice_session, wire = session(await make_user(), tts, MockLLM(), "ओपीडी का समय क्या है?")

    await speak(voice_session)

    assert tts.languages == ["en"] and wire.of("reply")[0]["language"] == "en"


async def test_analytics_record_the_voice_that_actually_spoke(client: AsyncClient) -> None:
    await seed_hospital()
    tts = FallbackTTS([Voice("gemini", "HTTP 429"), Voice("piper")])
    voice_session, _ = session(await make_user(), tts, MockLLM(), "What are the OPD timings?")

    await speak(voice_session)

    record = (await TurnAnalytics.find_all().to_list())[0]
    assert record.tts_provider == "piper" and record.spoken is True


async def test_analytics_show_when_nothing_could_speak(client: AsyncClient) -> None:
    await seed_hospital()
    tts = FallbackTTS([Voice("gemini", "HTTP 429")])
    voice_session, wire = session(await make_user(), tts, MockLLM(), "What are the OPD timings?")

    await speak(voice_session)

    record = (await TurnAnalytics.find_all().to_list())[0]
    assert record.tts_provider == "" and record.spoken is False
    assert wire.of("audio_unavailable") and wire.of("reply")


async def test_fallback_happens_inside_one_reply_without_losing_a_sentence(
    client: AsyncClient,
) -> None:
    first = Voice("gemini")
    second = Voice("piper")
    tts = FallbackTTS([first, second])
    text = "Your appointment is booked for Monday. Please arrive ten minutes early."
    voice_session, wire = session(await make_user(), tts, ScriptedLLM(say(text)), "book it")

    original = first.synthesize

    async def fail_on_second_call(chunk: str, language: str = "en") -> Speech:
        if first.calls >= 1:
            first.calls += 1
            raise TTSUnavailable("HTTP 429")
        return await original(chunk, language)

    first.synthesize = fail_on_second_call  # type: ignore[method-assign, assignment]
    await speak(voice_session)
    await asyncio.sleep(0)

    assert len(wire.audio) == 2 and wire.of("audio_end")
