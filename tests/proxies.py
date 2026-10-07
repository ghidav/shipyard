"""The two ends of the proxy, faked: a sampling client answering a fixed text and a renderer
of one character per token, so the reply to a request says which client served it. The
server between them is real: aiohttp, the cookbook's app, a port the kernel chose."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import tinker

from shipyard.proxy.cookbook import ParseTermination, ToolCall
from shipyard.proxy.endpoint import Endpoint
from shipyard.proxy.wire import CONTROL_PATH, RECORDS_PATH

MODEL = "Qwen/Qwen3-8B"


@dataclass
class FakeSequence:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "stop"


@dataclass
class FakeResponse:
    sequences: list[FakeSequence]
    prompt_cache_hit_tokens: int = 0


@dataclass
class FakeSampler:
    """`tinker.SamplingClient` as the proxy uses it: `answer` comes back as its characters'
    code points, one sequence per sample asked."""

    answer: str
    asked: list[Any] = field(default_factory=list)
    prompts: list[Any] = field(default_factory=list)
    cached: int = 0

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.asked.append(sampling_params)
        self.prompts.append(prompt)
        tokens = [ord(character) for character in self.answer]
        return FakeResponse(
            [FakeSequence(tokens, [-0.5] * len(tokens)) for _ in range(num_samples)],
            prompt_cache_hit_tokens=self.cached,
        )


class FakeRenderer:
    """The renderer methods the cookbook's proxy calls: a message renders to its text's
    code points and a completion decodes back the same way."""

    def get_stop_sequences(self) -> list[int]:
        return []

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> tinker.ModelInput:
        text = "".join(_text(message.get("content")) for message in messages)
        return tinker.ModelInput.from_ints([ord(character) for character in text])

    def parse_response(self, response: list[int]) -> tuple[dict[str, Any], Any]:
        text = "".join(chr(token) for token in response)
        return {"role": "assistant", "content": text}, ParseTermination.STOP_SEQUENCE


def _text(content: Any) -> str:
    """A message's text: the string itself, or its text parts joined."""
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def endpoint(
    sampler: Any = None, *, by_path: Callable[[str | None], Any] | None = None, **overrides: Any
) -> Endpoint:
    """An endpoint on loopback whose clients are fakes: `sampler` for every path, or
    `by_path(path)` so a swap is observable. Loopback, so nothing reaches in from outside
    and macOS asks no firewall question."""
    fixed = sampler if sampler is not None else FakeSampler("hi")

    async def factory(path: str | None) -> Any:
        return by_path(path) if by_path is not None else fixed

    fields: dict[str, Any] = {
        "base_model": MODEL,
        "renderer": FakeRenderer(),
        "host": "127.0.0.1",
        "bind": "127.0.0.1",
        "client_factory": factory,
    }
    return Endpoint(**{**fields, **overrides})


def ids(text: str) -> tuple[int, ...]:
    """What the fake renderer renders this text to."""
    return tuple(ord(character) for character in text)


def root(started: Endpoint) -> str:
    return f"http://127.0.0.1:{started.port}"


async def ask(
    started: Endpoint,
    trial: str,
    *,
    key: str | None = None,
    headers: dict[str, str] | None = None,
    **body: Any,
) -> httpx.Response:
    """One chat completion, as a harness on the OpenAI wire sends it."""
    sent = {"Authorization": f"Bearer {started.token if key is None else key}", **(headers or {})}
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{started.address_for(trial)}/chat/completions",
            headers=sent,
            json={"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "go"}], **body},
        )


async def ask_anthropic(started: Endpoint, trial: str, **body: Any) -> httpx.Response:
    """One messages request, as Claude Code sends it: `x-api-key`, `/v1/messages`."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{started.address_for(trial)}/messages",
            headers={"x-api-key": started.token},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "go"}],
                **body,
            },
        )


async def fetch(
    started: Endpoint, trial: str, headers: dict[str, str] | None = None, **params: str
) -> httpx.Response:
    """The records fetch a run makes; the endpoint's own token unless told otherwise."""
    if headers is None:
        headers = {"Authorization": f"Bearer {started.token}"}
    async with httpx.AsyncClient() as client:
        return await client.get(
            f"{root(started)}{RECORDS_PATH}/{trial}", headers=headers, params=params
        )


async def control(
    started: Endpoint, model_path: Any, *, token: str | None = None, raw: Any = None
) -> httpx.Response:
    """A swap over the control route, with the control token unless told otherwise."""
    bearer = started.control_token if token is None else token
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{root(started)}{CONTROL_PATH}",
            headers={"Authorization": f"Bearer {bearer}"},
            **({"content": raw} if raw is not None else {"json": {"model_path": model_path}}),
        )


# ------------------------------------------------------------- tools, thinking, bridging

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {"name": "ls", "description": "list", "parameters": {"type": "object"}},
    }
]
TOOLS_ANTHROPIC = [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}]


@dataclass
class ScriptedSampler(FakeSampler):
    """One answer per request, in order; the last one repeats."""

    answers: list[str] = field(default_factory=list)

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.asked.append(sampling_params)
        self.prompts.append(prompt)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        tokens = [ord(character) for character in answer]
        return FakeResponse([FakeSequence(tokens, [-0.5] * len(tokens))])


class BridgeRenderer(FakeRenderer):
    """Thinking as `{...}` ahead of the text, a tool call as `CALL name {json}`, a tool
    result as `tool[<id>]:text`; an assistant message re-renders UPPER-CASED behind `>` and
    closed by the stop token, so a fresh render never equals the sampled tokens."""

    STOP = 0

    def get_stop_sequences(self) -> list[int]:
        return [self.STOP]

    def create_conversation_prefix_with_tools(
        self, tool_specs: list[Any], system_prompt: str = ""
    ) -> list[dict[str, Any]]:
        names = ",".join(spec["name"] for spec in tool_specs)
        return [{"role": "system", "content": f"{system_prompt}<tools:{names}>"}]

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> tinker.ModelInput:
        out: list[int] = []
        for message in messages:
            role, content = message.get("role"), message.get("content")
            text = _text(content)
            if role == "assistant":
                thought = "".join(
                    "{" + str(part.get("thinking") or "") + "}"
                    for part in (content if isinstance(content, list) else [])
                    if isinstance(part, dict) and part.get("type") == "thinking"
                )
                calls = "".join(
                    f"CALL {call.function.name} {call.function.arguments}"
                    for call in message.get("tool_calls") or []
                )
                out += [ord(">"), *ids((thought + text + calls).upper()), self.STOP, ord("\n")]
            elif role == "tool":
                out += [*ids(f"tool[{message.get('tool_call_id', '')}]:{text}"), ord("\n")]
            else:
                out += [*ids(f"{role}:{text}"), ord("\n")]
        return tinker.ModelInput.from_ints([*out, ord(">")])

    def parse_response(self, response: list[int]) -> tuple[dict[str, Any], Any]:
        text = "".join(chr(token) for token in response if token != self.STOP)
        parts: list[dict[str, Any]] = []
        if text.startswith("{"):
            thought, _, text = text[1:].partition("}")
            parts.append({"type": "thinking", "thinking": thought})
        text, _, call = text.partition("CALL ")
        if text:
            parts.append({"type": "text", "text": text})
        message: dict[str, Any] = {"role": "assistant", "content": parts}
        if call:
            name, _, arguments = call.partition(" ")
            function = ToolCall.FunctionBody(name=name, arguments=arguments or "{}")
            message["tool_calls"] = [ToolCall(function=function)]
        return message, ParseTermination.STOP_SEQUENCE


def chars(tokens: Any) -> str:
    """What the fake renderer's ids read as; the stop token shows as a NUL."""
    return "".join(chr(int(token)) for token in tokens)
