"""Fake sampling clients and renderers for the proxy server, plus HTTP helpers for tests that
run a real server. The fake renderers map one character to one token."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import httpx
import tinker
from tinker_cookbook.renderers import ToolCall

from harness_proxy.cookbook import ParseTermination
from harness_proxy.server import Lane, Server
from harness_proxy.wire import TRIALS_PATH, Opening, Picture, Record, closing_path

MODEL = "Qwen/Qwen3-8B"
#: The name of the lane `server` serves, as `harness-proxy --model` names it.
LANE = "agent"
#: What `server` gives the server rather than its lane.
SERVER_FIELDS = frozenset(one.name for one in fields(Server)) - {"lanes"}


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
    sampler: Any = None,
    *,
    by_path: Callable[[str | None], Any] | None = None,
    others: Sequence[Lane] = (),
    **overrides: Any,
) -> Server:
    """Build a proxy `Server` on loopback with one lane, `LANE`, on fake sampling clients, and
    `others` after it. `sampler` serves every weights path unless `by_path(path)` picks a client
    per path, which makes swaps observable. Each override goes to the server or to the lane,
    whichever takes it. Loopback keeps outside hosts away and avoids the macOS firewall prompt."""
    fixed = sampler if sampler is not None else FakeSampler("hi")

    async def factory(path: str | None) -> Any:
        return by_path(path) if by_path is not None else fixed

    lane: dict[str, Any] = {
        "name": LANE,
        "model": MODEL,
        "renderer": FakeRenderer(),
        "client_factory": factory,
    }
    served: dict[str, Any] = {"host": "127.0.0.1", "bind": "127.0.0.1", "control_token": "ctl"}
    for key, value in overrides.items():
        (served if key in SERVER_FIELDS else lane)[key] = value
    return Server(lanes=[Lane(**lane), *others], **served)


def only(started: Server) -> Lane:
    """Return the server's one lane, which a trial opened on none is opened on."""
    return started.lane("")


def recorded(started: Server, trial: str) -> list[Record]:
    """Return what the open trial `trial` has recorded so far; [] for one that is not open."""
    opened = started.trial(trial)
    return list(opened.records) if opened is not None else []


def by_path(path: str | None) -> FakeSampler:
    """Return a fake sampler that answers with the last part of `path`, or "base" if None."""
    return FakeSampler("base" if path is None else path.rsplit("/", 1)[-1])


def ids(text: str) -> tuple[int, ...]:
    """Return the token ids the fake renderer produces for `text`."""
    return tuple(ord(character) for character in text)


def root(started: Server) -> str:
    return f"http://127.0.0.1:{started.port}"


async def opened(
    started: Server,
    trial: str,
    *,
    lane: str | None = None,
    renderer: str | None = None,
    weights: str | None = None,
) -> str:
    """Open `trial` on the control route, as the run does, and return its token; opening it
    again the same way returns the same token."""
    opening = Opening(trial, lane, renderer, weights)
    async with httpx.AsyncClient() as client:
        answered = await client.post(
            f"{root(started)}{TRIALS_PATH}",
            headers={"Authorization": f"Bearer {started.control_token}"},
            json=opening.body,
        )
    assert answered.status_code == 200, answered.text
    return str(answered.json()["token"])


async def ask(
    started: Server,
    trial: str,
    *,
    key: str | None = None,
    headers: dict[str, str] | None = None,
    **body: Any,
) -> httpx.Response:
    """Send one OpenAI chat completion request for `trial`, opened on first use, as a harness
    would; with the token `key` instead of the trial's when given."""
    token = await opened(started, trial) if key is None else key
    sent = {"Authorization": f"Bearer {token}", **(headers or {})}
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{root(started)}/v1/chat/completions",
            headers=sent,
            json={"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "go"}], **body},
        )


async def ask_anthropic(started: Server, trial: str, **body: Any) -> httpx.Response:
    """Send one Anthropic messages request for `trial`, opened on first use, authenticated with
    `x-api-key` as Claude Code does."""
    token = await opened(started, trial)
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{root(started)}/v1/messages",
            headers={"x-api-key": token},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "go"}],
                **body,
            },
        )


async def close(
    started: Server, trial: str, headers: dict[str, str] | None = None
) -> httpx.Response:
    """Close `trial` on the control route, authenticated with the control token unless
    `headers` is given, as the run closes it."""
    if headers is None:
        headers = {"Authorization": f"Bearer {started.control_token}"}
    async with httpx.AsyncClient() as client:
        return await client.post(f"{root(started)}{closing_path(trial)}", headers=headers)


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


def made(
    prompt: str | tuple[int, ...] = "go",
    completion: str | tuple[int, ...] = "ok",
    *,
    seq: int = 1,
    error: str | None = None,
    weights: str | None = None,
    cached: int = 0,
    bridged: bool = False,
    request_id: str | None = None,
    images: tuple[Picture, ...] = (),
) -> Record:
    """Build a `Record` as the proxy would store it, rendering text one character per token."""
    prompt_ids = ids(prompt) if isinstance(prompt, str) else tuple(prompt)
    completion_ids = (
        () if error else (ids(completion) if isinstance(completion, str) else tuple(completion))
    )
    return Record(
        seq=seq,
        at=0.0,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        inference_logprobs=(-0.5,) * len(completion_ids),
        stop_reason="" if error else "stop",
        sampling_params={"temperature": 1.0},
        requested_model="m",
        sample_ms=0.0 if error else 12.0,
        weights=weights,
        cached_tokens=cached,
        request_id=request_id,
        bridged=bridged,
        prompt_digests=(),
        reply_digest=None,
        error=error,
        images=images,
    )
