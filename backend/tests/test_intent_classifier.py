import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agents.embeddings import Vector, load_embedder
from agents.intent_classifier import (
    IntentClassifier,
    detect_language,
    keyword_scores,
    normalise,
)
from agents.intent_data import EXEMPLARS, KEYWORDS, Intent

EVAL_DIR = Path(__file__).parent.parent / "eval"


def _rows(name: str) -> list[dict[str, str]]:
    lines = (EVAL_DIR / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class FakeEmbedder:
    """Puts each text on the axis of the first intent whose name appears in it."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls += 1
        axes = list(Intent)
        vectors = []
        for text in texts:
            hit = next((i for i, intent in enumerate(axes) if intent.value in text.lower()), None)
            vectors.append([1.0 if i == hit else 0.0 for i in range(len(axes))])
        return vectors


@pytest.fixture
def exemplars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agents.intent_classifier.EXEMPLARS", {intent: [intent.value] for intent in Intent}
    )


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("I want to book an appointment for tomorrow", Intent.BOOKING),
        ("kal ka appointment cancel karo", Intent.BOOKING),
        ("मेरा अपॉइंटमेंट रद्द कर दीजिए", Intent.BOOKING),
        ("ನನ್ನ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ರದ್ದು ಮಾಡಿ", Intent.BOOKING),
        ("I have chest pain, which department should I go to?", Intent.TRIAGE),
        ("ನನಗೆ ಜ್ವರ ಮತ್ತು ಕೆಮ್ಮು ಇದೆ", Intent.TRIAGE),
        ("Show me my last prescription", Intent.RECORDS),
        ("pichli baar doctor ne kya dawai di thi?", Intent.RECORDS),
        ("What are the OPD timings?", Intent.SUPPORT),
        ("बच्चों का विभाग कहाँ है?", Intent.SUPPORT),
        ("Thank you", Intent.GENERAL),
    ],
)
def test_clear_requests_are_settled_by_keywords_without_the_llm(text: str, intent: Intent) -> None:
    result = IntentClassifier().classify(text)

    assert result.intent == intent
    assert result.source == "keywords"
    assert not result.needs_llm


def test_keywords_alone_never_call_the_embedder() -> None:
    embedder = FakeEmbedder()

    IntentClassifier(embedder).classify("Please cancel my appointment")

    assert embedder.calls == 0


@pytest.mark.parametrize("text", ["", "   ", "xyzzy plugh", "the weather is nice today"])
def test_an_unrecognised_turn_is_handed_to_the_llm(text: str) -> None:
    result = IntentClassifier().classify(text)

    assert result.needs_llm
    assert result.source == "none"
    assert result.intent == Intent.GENERAL


def test_a_single_weak_hint_is_a_guess_not_a_decision() -> None:
    result = IntentClassifier().classify("Is Dr. Asha Rao free this Thursday?")

    assert result.intent == Intent.BOOKING
    assert result.needs_llm


def test_evenly_split_signals_go_to_the_llm() -> None:
    # "appointment" (booking) against "history" (records): the rules cannot choose
    result = IntentClassifier().classify("appointment history")

    assert result.needs_llm


def test_embeddings_settle_a_turn_the_keywords_missed(exemplars: None) -> None:
    result = IntentClassifier(FakeEmbedder()).classify("this is a triage sort of sentence")

    assert result.intent == Intent.TRIAGE
    assert result.source == "embedding"
    assert not result.needs_llm


def test_embeddings_with_no_clear_winner_go_to_the_llm(exemplars: None) -> None:
    result = IntentClassifier(FakeEmbedder()).classify("nothing recognisable here")

    assert result.needs_llm


def test_embeddings_that_contradict_a_keyword_hint_go_to_the_llm(exemplars: None) -> None:
    # "free" hints at booking; the fake embedder says triage
    result = IntentClassifier(FakeEmbedder()).classify("is the triage room free")

    assert result.needs_llm
    assert result.intent == Intent.BOOKING


def test_exemplars_are_embedded_once_and_reused(exemplars: None) -> None:
    embedder = FakeEmbedder()
    classifier = IntentClassifier(embedder)

    classifier.classify("a triage sentence")
    classifier.classify("a support sentence")

    assert embedder.calls == 3  # exemplars once, then one call per query


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("I need an appointment", "en"),
        ("मुझे अपॉइंटमेंट चाहिए", "hi"),
        ("ನನಗೆ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಬೇಕು", "kn"),
        ("kal ka appointment cancel karo", "hi"),
        ("nanna appointment cancel maadi", "kn"),
        ("OPD timing kya hai?", "hi"),
        ("emergency number enu?", "kn"),
    ],
)
def test_language_detection(text: str, language: str) -> None:
    assert detect_language(text) == language
    assert IntentClassifier().classify(text).language == language


def test_normalise_strips_joiners_case_and_punctuation() -> None:
    assert normalise("  Hello,   WORLD!? ") == "hello world"
    assert normalise("ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್") == normalise("ಅಪಾಯಿಂಟ್ಮೆಂಟ್")


def test_word_boundaries_stop_short_keywords_matching_inside_words() -> None:
    # "hi" must not fire inside "this", nor "band" inside "husband"
    assert keyword_scores("this is my husband")[Intent.GENERAL] == 0
    assert keyword_scores("this is my husband")[Intent.SUPPORT] == 0


def test_every_intent_has_keywords_and_exemplars() -> None:
    assert set(KEYWORDS) == set(EXEMPLARS) == set(Intent)


def test_exemplars_never_repeat_an_eval_utterance() -> None:
    """Otherwise the embedding stage would be graded on sentences it was given."""
    evaluated = {
        normalise(row["text"])
        for name in ("utterances.jsonl", "utterances_heldout.jsonl")
        for row in _rows(name)
    }
    exemplars = {normalise(phrase) for phrases in EXEMPLARS.values() for phrase in phrases}

    assert not evaluated & exemplars


@pytest.mark.parametrize("name", ["utterances.jsonl", "utterances_heldout.jsonl"])
def test_eval_sets_are_well_formed(name: str) -> None:
    rows = _rows(name)

    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["intent"] for row in rows} == {intent.value for intent in Intent}
    assert {row["lang"] for row in rows} == {"en", "hi", "kn", "mixed"}
    assert all(row["text"].strip() for row in rows)


def test_the_main_eval_set_is_the_size_the_prd_asks_for() -> None:
    assert 60 <= len(_rows("utterances.jsonl")) <= 80


def test_keyword_rules_hold_on_the_set_they_were_written_with() -> None:
    """A regression guard, not a quality claim: this set and the rules share an author."""
    classifier = IntentClassifier()
    rows = _rows("utterances.jsonl")

    wrong = [row["id"] for row in rows if classifier.classify(row["text"]).intent != row["intent"]]

    assert len(wrong) / len(rows) <= 0.05, wrong


def test_a_confident_local_answer_is_never_wrong_on_the_heldout_set() -> None:
    classifier = IntentClassifier()

    confident_and_wrong = [
        row["id"]
        for row in _rows("utterances_heldout.jsonl")
        if not (result := classifier.classify(row["text"])).needs_llm
        and result.intent != row["intent"]
    ]

    assert confident_and_wrong == []


def test_the_embedder_is_off_when_no_model_is_configured() -> None:
    assert load_embedder("") is None
