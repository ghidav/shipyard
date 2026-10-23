"""Tests for a proxy given several renderers, such as reasoning efforts: each trial is opened
with the one its calls are rendered with, and `/healthz` names them all."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
import tinker

from harness_proxy import cookbook, serve
from harness_proxy.client import Proxy
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
    only,
    recorded,
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


def client(started: Server) -> Proxy:
    """Return the run's client for a started server."""
    return Proxy(origin=root(started), control_token=started.control_token)


async def ask_at(started: Server, trial: str, renderer: str | None, **body: Any) -> httpx.Response:
    """Open `trial` with `renderer` through the run's client, once, and send one OpenAI chat
    completion with its token to the address every trial dials."""
    proxy = client(started)
    token = await proxy.open(trial, renderer=renderer)
    async with httpx.AsyncClient() as http:
        return await http.post(
            f"{proxy.address}/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "m", "messages": [{"role": "user", "content": "go"}], **body},
        )


# ------------------------------------------------------------------ the opening


def test_every_trial_dials_one_address_and_its_token_says_which_trial_it_is() -> None:
    proxy = Proxy(origin="http://proxy:8000/", control_token="c")
    assert proxy.address == proxy.local_address == "http://proxy:8000/v1"


async def test_each_trial_is_rendered_with_the_renderer_it_was_opened_with() -> None:
    sampler = FakeSampler("hi")
    async with server(sampler, renderer=efforts()) as started:
        for trial, renderer in (("a", "low"), ("b", "high"), ("c", None)):
            assert (await ask_at(started, trial, renderer)).status_code == 200
        opened = {t: chars(recorded(started, t)[0].prompt_token_ids) for t in "abc"}
    assert opened == {"a": "[low]user:go\n>", "b": "[high]user:go\n>", "c": "[low]user:go\n>"}, (
        "a trial opened with none takes the first"
    )


async def test_a_renderer_the_proxy_was_not_given_is_refused_when_the_trial_opens() -> None:
    sampler = FakeSampler("hi")
    async with server(sampler, renderer={"low": Effort("low")}) as started:
        with pytest.raises(
            RuntimeError, match="does not render with 'medium': it renders with low"
        ):
            await client(started).open("t", renderer="medium")
        assert sampler.asked == [] and started.trial("t") is None


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
            one, two = recorded(started, trial)
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
        assert only(started).renderers == ["low", "high"] and built == ["low", "high"]
    built.clear()
    async with server(renderer=None) as started:
        assert only(started).renderers == [] and built == [None], (
            "the model's default, named by none"
        )


async def test_healthz_names_each_lanes_renderers_and_the_client_reads_them() -> None:
    async with server(renderer=efforts()) as started:
        proxy = client(started)
        (lane,) = (await proxy.ready(seconds=5)).values()
        assert (lane.model, lane.renderers) == (MODEL, ("low", "high"))
    async with server() as plain:
        proxy = client(plain)
        await proxy.ready(seconds=5)
        assert proxy.lanes["agent"].renderers == (), "a lane on its model's default names none"


def test_the_renderer_flag_names_each_renderer_of_the_one_lane() -> None:
    args = serve.parser().parse_args(["--model", "m", "--renderer", "low", "--renderer", "high"])
    assert args.renderer == ["low", "high"]
    given = serve.document(None, lane=args.lane, model=args.model, renderer=args.renderer)
    assert given == {"lanes": [{"name": "agent", "model": "m", "renderer": ["low", "high"]}]}
