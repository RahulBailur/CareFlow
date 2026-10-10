"""Provider-agnostic LLM interface with tool calling.

Agents talk to `LLMProvider` only. When a provider cannot answer (no key, quota, timeout,
bad response) it raises `LLMUnavailable`, and the caller falls back to rule-based replies.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Protocol

import httpx

from config import get_settings

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REQUEST_TIMEOUT_S = 20.0


class LLMUnavailable(Exception):
    """The LLM could not produce an answer for this turn."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant turns
    tool_name: str = ""  # tool turns
    tool_result: dict[str, Any] | None = None  # tool turns
    # The provider's own form of an assistant turn, replayed verbatim on the next request
    # to that same provider (a turn can move to another one halfway through)
    raw: Any = None
    provider: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    provider: str = ""  # which model answered


class LLMProvider(Protocol):
    name: str

    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse: ...


class MockLLM:
    """No model at all (`LLM_PROVIDER=mock`): every turn takes the rule-based path."""

    name = "mock"

    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        raise LLMUnavailable("LLM_PROVIDER is mock")


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini takes an OpenAPI-style subset: upper-case types, no additionalProperties."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if key == "type" and isinstance(value, str):
            out[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            out[key] = {name: _gemini_schema(prop) for name, prop in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _gemini_schema(value)
        else:
            out[key] = value
    return out


def _gemini_contents(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            contents.append({"role": "user", "parts": [{"text": message.text}]})
        elif message.role == "assistant":
            if message.raw is not None and message.provider == GeminiLLM.name:
                contents.append(message.raw)
                continue
            parts: list[dict[str, Any]] = [{"text": message.text}] if message.text else []
            parts += [
                {"functionCall": {"name": c.name, "args": c.args}} for c in message.tool_calls
            ]
            contents.append({"role": "model", "parts": parts})
        else:
            part = {
                "functionResponse": {
                    "name": message.tool_name,
                    "response": message.tool_result or {},
                }
            }
            # Results of one assistant turn's calls travel together in a single user turn
            if contents and contents[-1]["role"] == "user" and _is_tool_turn(contents[-1]):
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
    return contents


def _is_tool_turn(content: dict[str, Any]) -> bool:
    return all("functionResponse" in part for part in content["parts"])


class GeminiLLM:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._transport = transport
        self._timeout_s = timeout_s

    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        if not self._api_key or not self._model:
            raise LLMUnavailable("GEMINI_API_KEY and GEMINI_TEXT_MODEL must both be set")
        body: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": _gemini_contents(messages),
            "generationConfig": {"temperature": 0.2},
        }
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": _gemini_schema(tool.parameters),
                        }
                        for tool in tools
                    ]
                }
            ]
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout_s
            ) as client:
                response = await client.post(
                    GEMINI_URL.format(model=self._model),
                    headers={"x-goog-api-key": self._api_key},
                    json=body,
                )
        except httpx.HTTPError as error:
            raise LLMUnavailable(f"Gemini request failed: {type(error).__name__}") from error
        if response.status_code != httpx.codes.OK:
            # The body is not included: it can echo the request
            raise LLMUnavailable(f"Gemini returned HTTP {response.status_code}")

        try:
            content = response.json()["candidates"][0]["content"]
            parts = content.get("parts", [])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise LLMUnavailable("Gemini returned no usable candidate") from error
        text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
        calls = [
            ToolCall(p["functionCall"]["name"], dict(p["functionCall"].get("args") or {}))
            for p in parts
            if "functionCall" in p
        ]
        return LLMResponse(text=text, tool_calls=calls, raw=content, provider=self.name)


def _ollama_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if message.role == "user":
            out.append({"role": "user", "content": message.text})
        elif message.role == "assistant":
            turn: dict[str, Any] = {"role": "assistant", "content": message.text}
            if message.tool_calls:
                turn["tool_calls"] = [
                    {"function": {"name": call.name, "arguments": call.args}}
                    for call in message.tool_calls
                ]
            out.append(turn)
        else:
            out.append(
                {
                    "role": "tool",
                    "tool_name": message.tool_name,
                    "content": json.dumps(message.tool_result or {}, ensure_ascii=False),
                }
            )
    return out


class OllamaLLM:
    """A small model running locally through Ollama: slower and weaker, but always there."""

    name = "ollama"

    def __init__(
        self,
        url: str,
        model: str,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url, self._model = url.rstrip("/"), model
        self._timeout_s, self._transport = timeout_s, transport

    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        if not self._url or not self._model:
            raise LLMUnavailable("OLLAMA_URL and OLLAMA_MODEL must both be set")
        body: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "messages": _ollama_messages(system, messages),
            "options": {"temperature": 0.2},
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout_s
            ) as client:
                response = await client.post(f"{self._url}/api/chat", json=body)
        except httpx.HTTPError as error:
            raise LLMUnavailable(f"Ollama request failed: {type(error).__name__}") from error
        if response.status_code != httpx.codes.OK:
            raise LLMUnavailable(f"Ollama returned HTTP {response.status_code}")
        try:
            message = response.json()["message"]
            calls = []
            for call in message.get("tool_calls") or []:
                arguments = call["function"].get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append(ToolCall(call["function"]["name"], dict(arguments)))
            text = message.get("content") or ""
        except (KeyError, TypeError, ValueError) as error:
            raise LLMUnavailable("Ollama returned no usable message") from error
        return LLMResponse(text=text, tool_calls=calls, provider=self.name)


@lru_cache
def get_llm() -> LLMProvider:
    """The fallback chain: `LLM_PROVIDER` first, the other real provider behind it."""
    settings = get_settings()
    if settings.llm_provider == "mock":
        return MockLLM()
    from services.failover import CircuitBreaker, FailoverLLM

    gemini: LLMProvider = GeminiLLM(
        settings.gemini_api_key, settings.gemini_text_model, timeout_s=settings.llm_timeout_s
    )
    chain = [gemini]
    if settings.ollama_model:  # the local tier only exists once a model is named
        ollama = OllamaLLM(settings.ollama_url, settings.ollama_model, settings.ollama_timeout_s)
        chain = [ollama, gemini] if settings.llm_provider == "ollama" else [gemini, ollama]
    return FailoverLLM(
        [
            (
                provider,
                CircuitBreaker(
                    settings.gemini_breaker_threshold, settings.gemini_breaker_cooldown_s
                ),
            )
            for provider in chain
        ]
    )
