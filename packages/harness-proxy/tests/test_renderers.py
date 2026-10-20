"""Tests for a proxy given several renderers, such as reasoning efforts: each trial's address
names the one its calls are rendered with, and `/healthz` names them all."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
import tinker

from harness_proxy import cookbook, serve
from harness_proxy.client import Proxy, serve_command
from harness_proxy.server import Server
from proxies import (
    MODEL,
    TOOLS_OPENAI,
    BridgeRenderer,
    FakeResponse,
    FakeSampler,
    FakeSequence,
    chars,
    ids,
    root,
    server,
)


class Effort(BridgeRenderer):
    """`BridgeRenderer` that opens every prompt with its effort, as Qwen3.8's renderers write
    theirs into the system message."""

    def __init__(self, effort: str) -> None:
        self.effort = effort

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> tinker.ModelInput:
        rendered = super().build_generation_prompt(messages)
        return tinker.ModelInput.from_ints([*ids(f"[{self.effort}]"), *rendered.to_ints()])


@dataclass
class Caller:
    """Fake sampler that calls `ls` until a tool result is in the prompt, then says `done`."""

    prompts: list[Any] = field(default_factory=list)

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.prompts.append(prompt)
        tokens = list(ids("done" if "tool[" in chars(prompt.to_ints()) else "CALL ls {}"))
        return FakeResponse([FakeSequence(tokens, [-0.5] * len(tokens))])


def efforts() -> dict[str, Effort]:
    return {"low": Effort("low"), "high": Effort("high")}


async def ask_at(started: Server, trial: str, renderer: str | None, **body: Any) -> httpx.Response:
    """Send one OpenAI chat completion for `trial` to the address the run's client gives it."""
    base = Proxy(origin=root(started), token=started.token).address_for(trial, renderer)
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {started.token}"},
            json={"model": "m", "messages": [{"role": "user", "content": "go"}], **body},
        )


# ------------------------------------------------------------------ the address


def test_a_trials_address_names_its_renderer_when_it_has_one() -> None:
    proxy = Proxy(origin="http://proxy:8000/", token="t")
    assert proxy.address_for("x__1") == "http://proxy:8000/r/trial/x__1/v1"
    assert proxy.address_for("x__1", "low") == "http://proxy:8000/r/trial/x__1/renderer/low/v1"
    for bad in ("", "a/b"):
        with pytest.raises(ValueError, match="a renderer goes in the URL path"):
            proxy.address_for("x__1", bad)


async def test_each_trial_is_rendered_with_the_renderer_its_address_names() -> None:
    sampler = FakeSampler("hi")
    async with server(sampler, renderer=efforts()) as started:
        for trial, renderer in (("a", "low"), ("b", "high"), ("c", None)):
            assert (await ask_at(started, trial, renderer)).status_code == 200
        opened = {t: chars(started.records_for(t)[0].prompt_token_ids) for t in "abc"}
    assert opened == {"a": "[low]user:go\n>", "b": "[high]user:go\n>", "c": "[low]user:go\n>"}, (
        "an address that names none takes the first"
    )


async def test_a_renderer_the_proxy_was_not_given_is_refused_before_it_samples() -> None:
    sampler = FakeSampler("hi")
    async with server(sampler, renderer={"low": Effort("low")}) as started:
        refused = await ask_at(started, "t", "medium")
        assert refused.status_code == 400
        assert "does not render with 'medium': it renders with low" in refused.text
        assert sampler.asked == [] and started.records_for("t") == []


async def test_each_renderer_bridges_its_own_trials_tool_loop() -> None:
    async with server(Caller(), renderer=efforts()) as started:
        for trial, renderer in (("a", "low"), ("b", "high")):
            history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
            first = await ask_at(started, trial, renderer, messages=history, tools=TOOLS_OPENAI)
            said = first.json()["choices"][0]["message"]
            history += [
                {"role": "assistant", "content": None, "tool_calls": said["tool_calls"]},
                {"role": "tool", "tool_call_id": said["tool_calls"][0]["id"], "content": "a b"},
            ]
            second = await ask_at(started, trial, renderer, messages=history, tools=TOOLS_OPENAI)
            assert second.status_code == 200, second.text
        for trial, renderer in (("a", "low"), ("b", "high")):
            one, two = started.records_for(trial)
            assert (one.bridged, two.bridged) == (False, True)
            assert chars(one.prompt_token_ids).startswith(f"[{renderer}]system:")
            assert two.prompt_token_ids[: len(one.prompt_token_ids)] == one.prompt_token_ids


# ------------------------------------------------------------------ what it was given


async def test_named_renderers_are_built_from_the_cookbook_in_the_order_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str | None] = []

    def renderer_for(model: str, name: str | None = None) -> Effort:
        built.append(name)
        return Effort(name or "default")

    monkeypatch.setattr(cookbook, "renderer_for", renderer_for)
    async with server(renderer=["low", "high"]) as started:
        assert started.renderers == ["low", "high"] and built == ["low", "high"]
    built.clear()
    async with server(renderer=None) as started:
        assert started.renderers == [] and built == [None], "the model's default, named by none"


async def test_healthz_names_the_renderers_and_the_client_reads_them() -> None:
    async with server(renderer=efforts()) as started:
        proxy = Proxy(origin=root(started), token=started.token)
        assert await proxy.ready(seconds=5) == MODEL
        assert proxy.renderers == ["low", "high"]
    async with server() as plain:
        proxy = Proxy(origin=root(plain), token=plain.token)
        await proxy.ready(seconds=5)
        assert proxy.renderers == [], "a proxy on its model's default names none"


async def test_a_proxy_that_says_nothing_of_renderers_reads_as_unknown() -> None:
    answer = {"status": "ok", "serving": "m"}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=answer))
    proxy = Proxy(origin="http://proxy:1", token="t", transport=transport)
    await proxy.ready(seconds=1)
    assert proxy.renderers is None


def test_the_serve_command_gives_each_renderer_its_own_flag() -> None:
    args = serve.parser().parse_args(["--model", "m", "--renderer", "low", "--renderer", "high"])
    assert args.renderer == ["low", "high"]
    made = serve.server_for(
        "m",
        weights=None,
        bind="127.0.0.1",
        port=0,
        advertise=None,
        renderer=args.renderer,
        settings=None,
        token="k",
        control_token="c",
    )
    assert made.renderer == ["low", "high"]
    command = serve_command({"model": "m", "renderer": ["low", "high"]}, host="h")
    flagged = [at for at, one in enumerate(command) if one == "--renderer"]
    assert [command[at + 1] for at in flagged] == ["low", "high"]
    assert "--renderer" not in serve_command({"model": "m", "renderer": None}, host="h")
