"""Optional sentence-embedding backend for the intent classifier.

Installed with the `ml` extra (`pip install -e ".[ml]"`). Without it the classifier still
works on keywords alone and marks more turns as needing the LLM.
"""

from collections.abc import Sequence
from typing import Protocol

Vector = list[float]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


class FastEmbedEmbedder:
    """ONNX model run on CPU through fastembed: no PyTorch, no API call."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [[float(x) for x in vector] for vector in self._model.embed(list(texts))]


def load_embedder(model_name: str) -> Embedder | None:
    """The embedder, or None when it is switched off or fastembed is not installed."""
    if not model_name:
        return None
    try:
        return FastEmbedEmbedder(model_name)
    except ImportError:
        return None
