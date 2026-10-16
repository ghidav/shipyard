"""Fake sampling clients and renderers for the proxy server, plus HTTP helpers for tests that
run a real server. The fake renderers map one character to one token."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import tinker
from tinker_cookbook.renderers import ToolCall

from shipyard.proxy.cookbook import ParseTermination
from shipyard.proxy.server import Server
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
    """Fake `tinker.SamplingClient` that returns scripted replies. `answer` is either one reply
    for every request or a list of replies used in turn, the last one repeating; a string
    reply is returned as its code points and a sequence of ids as is."""

    answer: str | Sequence[str | Sequence[int]]
    asked: list[Any] = field(default_factory=list)
    prompts: list[Any] = field(default_factory=list)
    cached: int = 0

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.asked.append(sampling_params)
        self.prompts.append(prompt)
        replies = [self.answer] if isinstance(self.answer, str) else self.answer
        reply = replies[min(len(self.asked), len(replies)) - 1]
        tokens = [ord(character) for character in reply] if isinstance(reply, str) else list(reply)
        return FakeResponse(
            [FakeSequence(tokens, [-0.5] * len(tokens)) for _ in range(num_samples)],
            prompt_cache_hit_tokens=self.cached,
        )


@dataclass
class Echo:
    """Fake sampling client that replies with the prompt tokens, truncated to `max_tokens`."""

    path: str | None = None

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        tokens = list(prompt.to_ints())[: int(sampling_params.max_tokens or 0) or None]
        return FakeResponse([FakeSequence(tokens, [0.0] * len(tokens)) for _ in range(num_samples)])

    @staticmethod
    async def make(path: str | None) -> Echo:
        return Echo(path)


class FakeRenderer:
    """Fake renderer implementing the methods the cookbook proxy calls. A message renders to
    the code points of its text, and a completion decodes back the same way."""

    def get_stop_sequences(self) -> list[int]:
        return []

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> tinker.ModelInput:
        text = "".join(text_of(message.get("content")) for message in messages)
        return tinker.ModelInput.from_ints([ord(character) for character in text])

    def parse_response(self, response: list[int]) -> tuple[dict[str, Any], Any]:
        text = "".join(chr(token) for token in response)
        return {"role": "assistant", "content": text}, ParseTermination.STOP_SEQUENCE


def text_of(content: Any) -> str:
    """Return the text of a message's content, joining the text parts of a list."""
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def server(
    sampler: Any = None, *, by_path: Callable[[str | None], Any] | None = None, **overrides: Any
) -> Server:
    """Build a proxy `Server` on loopback with fake sampling clients. `sampler` serves every
    weights path unless `by_path(path)` picks a client per path, which makes swaps
    observable. Loopback keeps outside hosts away and avoids the macOS firewall prompt."""
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
    return Server(**{**fields, **overrides})


def by_path(path: str | None) -> FakeSampler:
    """Return a fake sampler that answers with the last part of `path`, or "base" if None."""
    return FakeSampler("base" if path is None else path.rsplit("/", 1)[-1])


def ids(text: str) -> tuple[int, ...]:
    """Return the token ids the fake renderer produces for `text`."""
    return tuple(ord(character) for character in text)


def root(started: Server) -> str:
    return f"http://127.0.0.1:{started.port}"


def address(started: Server, trial: str) -> str:
    """Return the base URL for `trial` on a started server, matching `Proxy.address_for`."""
    return f"{root(started)}/r/trial/{trial}/v1"


async def ask(
    started: Server,
    trial: str,
    *,
    key: str | None = None,
    headers: dict[str, str] | None = None,
    **body: Any,
) -> httpx.Response:
    """Send one OpenAI chat completion request for `trial`, as a harness would."""
    sent = {"Authorization": f"Bearer {started.token if key is None else key}", **(headers or {})}
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{address(started, trial)}/chat/completions",
            headers=sent,
            json={"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "go"}], **body},
        )


async def ask_anthropic(started: Server, trial: str, **body: Any) -> httpx.Response:
    """Send one Anthropic messages request for `trial`, authenticated with `x-api-key` as
    Claude Code does."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{address(started, trial)}/messages",
            headers={"x-api-key": started.token},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "go"}],
                **body,
            },
        )


async def fetch(
    started: Server, trial: str, headers: dict[str, str] | None = None, **params: str
) -> httpx.Response:
    """Fetch the records for `trial`, authenticated with the server token unless `headers`
    is given."""
    if headers is None:
        headers = {"Authorization": f"Bearer {started.token}"}
    async with httpx.AsyncClient() as client:
        return await client.get(
            f"{root(started)}{RECORDS_PATH}/{trial}", headers=headers, params=params
        )


async def control(
    started: Server, model_path: Any, *, token: str | None = None, raw: Any = None
) -> httpx.Response:
    """Request a weights swap on the control route, authenticated with the control token
    unless `token` is given."""
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


class BridgeRenderer(FakeRenderer):
    """Fake renderer that encodes thinking and tool calls as plain text. Thinking renders as
    `{...}` before the text, a tool call as `CALL name {json}`, and a tool result as
    `tool[<id>]:text`. An assistant message re-renders upper-cased after `>` and ends with the
    stop token, so a fresh render never equals the sampled tokens."""

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
            text = text_of(content)
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
    """Decode fake renderer ids back to text; the stop token decodes as NUL."""
    return "".join(chr(int(token)) for token in tokens)
