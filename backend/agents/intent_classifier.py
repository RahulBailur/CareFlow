"""Local intent classifier: keyword rules first, sentence embeddings second.

Most turns are settled here on CPU with no model call. A turn that neither stage is sure
about comes back with `needs_llm=True` and a best guess, and the orchestrator asks the LLM.
"""

import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from agents.embeddings import Embedder, Vector, load_embedder
from agents.intent_data import EXEMPLARS, KEYWORDS, Intent
from config import get_settings

Language = Literal["en", "hi", "kn"]
Source = Literal["keywords", "embedding", "none"]

# Keyword confidence is top / (top + runner_up + 1): one clear signal with no rival passes
KEYWORD_CONFIDENT = 0.6
# Cosine similarity to the closest exemplar, and its lead over the next intent
EMBEDDING_CONFIDENT = 0.6
EMBEDDING_MARGIN = 0.05

_JOINERS = dict.fromkeys(map(ord, "‌‍"))  # zero-width (non-)joiners
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)

_ROMAN_HINDI = frozenset(
    "hai hain kya karo kar mujhe mera meri mere mein kahan batao dikhao karna nahi aur "
    "kal aap raha rahi tha thi chahiye kaunse".split()
)
_ROMAN_KANNADA = frozenset(
    "maadi madi beku nanna nanage ide enu ellide elli yaava yaavaga hogbeku thumba nale "
    "illa aagittu torisi kodi".split()
)


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    source: Source
    needs_llm: bool
    language: Language


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_JOINERS).lower()
    # Keep combining marks (Indic vowel signs are category M, which \w does not cover)
    kept = "".join(
        ch if ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M") else " "
        for ch in text
    )
    return " ".join(kept.split())


def detect_language(text: str) -> Language:
    """Script first; for Latin script, a few very common romanised Hindi/Kannada words."""
    if any("ಀ" <= ch <= "೿" for ch in text):
        return "kn"
    if any("ऀ" <= ch <= "ॿ" for ch in text):
        return "hi"
    words = set(normalise(text).split())
    hindi, kannada = len(words & _ROMAN_HINDI), len(words & _ROMAN_KANNADA)
    if kannada > hindi:
        return "kn"
    if hindi:
        return "hi"
    return "en"


def _compile(pattern: str) -> re.Pattern[str]:
    if pattern.isascii():
        return re.compile(rf"\b(?:{pattern})\b")
    return re.compile(re.escape(normalise(pattern)))


_RULES: dict[Intent, list[tuple[re.Pattern[str], int]]] = {
    intent: [(_compile(pattern), weight) for pattern, weight in rules]
    for intent, rules in KEYWORDS.items()
}


def keyword_scores(text: str) -> dict[Intent, int]:
    clean = normalise(text)
    return {
        intent: sum(weight for pattern, weight in rules if pattern.search(clean))
        for intent, rules in _RULES.items()
    }


def _cosine(a: Vector, b: Vector) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


class IntentClassifier:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder
        self._exemplars: list[tuple[Intent, Vector]] | None = None

    @property
    def has_embeddings(self) -> bool:
        return self._embedder is not None

    def _similarities(self, text: str) -> dict[Intent, float]:
        """Closest-exemplar cosine similarity for each intent."""
        assert self._embedder is not None  # noqa: S101 — callers check has_embeddings
        if self._exemplars is None:
            labelled = [
                (intent, phrase) for intent, phrases in EXEMPLARS.items() for phrase in phrases
            ]
            vectors = self._embedder.embed([phrase for _, phrase in labelled])
            self._exemplars = [
                (intent, v) for (intent, _), v in zip(labelled, vectors, strict=True)
            ]
        query = self._embedder.embed([text])[0]
        best = dict.fromkeys(Intent, 0.0)
        for intent, vector in self._exemplars:
            best[intent] = max(best[intent], _cosine(query, vector))
        return best

    def classify(self, text: str) -> IntentResult:
        language = detect_language(text)
        scores = keyword_scores(text)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        (kw_intent, top), (_, runner_up) = ranked[0], ranked[1]
        kw_confidence = top / (top + runner_up + 1)

        if top and top > runner_up and kw_confidence >= KEYWORD_CONFIDENT:
            return IntentResult(kw_intent, kw_confidence, "keywords", False, language)

        kw_guess = kw_intent if top > runner_up else None
        if self._embedder is None or not text.strip():
            return IntentResult(kw_guess or Intent.GENERAL, kw_confidence, "none", True, language)

        similar = sorted(self._similarities(text).items(), key=lambda item: item[1], reverse=True)
        (emb_intent, best), (_, second) = similar[0], similar[1]
        agrees = kw_guess is None or kw_guess == emb_intent
        if best >= EMBEDDING_CONFIDENT and best - second >= EMBEDDING_MARGIN and agrees:
            return IntentResult(emb_intent, best, "embedding", False, language)
        return IntentResult(kw_guess or emb_intent, best, "none", True, language)


@lru_cache
def get_classifier() -> IntentClassifier:
    return IntentClassifier(load_embedder(get_settings().intent_embedding_model))
