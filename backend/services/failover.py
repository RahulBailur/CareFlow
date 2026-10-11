"""The LLM fallback chain and its circuit breakers.

    Redis cache  ->  Gemini  ->  Ollama (local)  ->  rule-based replies

The cache is checked by the orchestrator before any of this, and the rule-based replies are
what an agent falls back to when this chain raises `LLMUnavailable`. This module is the
middle: try each model in order, and stop asking one that keeps failing.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from services.llm import LLMProvider, LLMResponse, LLMUnavailable, Message, ToolSpec

logger = logging.getLogger(__name__)

BreakerState = Literal["closed", "open", "half_open"]

# Every provider failure during the current turn, as "provider:kind"
_turn_errors: ContextVar[list[str] | None] = ContextVar("llm_turn_errors", default=None)


@contextmanager
def collecting_llm_errors() -> Iterator[list[str]]:
    """Collect the LLM failures of one turn, e.g. ["gemini:429"], for the analytics."""
    errors: list[str] = []
    token = _turn_errors.set(errors)
    try:
        yield errors
    finally:
        _turn_errors.reset(token)


def error_kind(message: str) -> str:
    if "429" in message:
        return "429"
    if "circuit open" in message:
        return "circuit_open"
    if "Timeout" in message or "timeout" in message:
        return "timeout"
    if "HTTP 5" in message:
        return "5xx"
    return "error"


def _note(provider: str, kind: str) -> None:
    errors = _turn_errors.get()
    if errors is not None:
        errors.append(f"{provider}:{kind}")


class CircuitBreaker:
    """Stops calling a provider that keeps failing.

    closed     normal: every call goes through.
    open       after `threshold` failures in a row: nothing goes through for `cooldown_s`,
               so a provider that is out of quota does not cost every turn a wasted wait.
    half_open  after the cool-down: calls go through again; one success closes the breaker,
               one failure opens it for another full cool-down.
    """

    def __init__(
        self, threshold: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._threshold, self._cooldown_s, self._clock = max(1, threshold), cooldown_s, clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return "closed"
        return "open" if self._clock() - self._opened_at < self._cooldown_s else "half_open"

    def allows(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures, self._opened_at = 0, None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold or self._opened_at is not None:
            self._opened_at = self._clock()


class FailoverLLM:
    """Asks each provider in turn until one answers. The answer says which one it was."""

    def __init__(self, tiers: list[tuple[LLMProvider, CircuitBreaker]]) -> None:
        self._tiers = tiers
        self.name = "+".join(provider.name for provider, _ in tiers)

    def breaker_states(self) -> dict[str, BreakerState]:
        return {provider.name: breaker.state for provider, breaker in self._tiers}

    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        reasons = []
        for provider, breaker in self._tiers:
            if not breaker.allows():
                reasons.append(f"{provider.name}: circuit open")
                _note(provider.name, "circuit_open")
                continue
            try:
                response = await provider.generate(system, messages, tools)
            except LLMUnavailable as error:
                was = breaker.state
                breaker.record_failure()
                if breaker.state == "open" and was != "open":
                    logger.warning("Circuit opened for %s: %s", provider.name, error)
                reasons.append(f"{provider.name}: {error}")
                _note(provider.name, error_kind(str(error)))
                continue
            breaker.record_success()
            response.provider = provider.name
            return response
        raise LLMUnavailable("; ".join(reasons) or "no LLM provider configured")
