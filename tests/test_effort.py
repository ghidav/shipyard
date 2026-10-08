"""`[rollout] effort`: the thinking effort a renderer that takes one at prompt time builds
every prompt at, Inkling's tml_v0 among them. The key loads within tml_v0's range, reaches
the proxy, a scripted serve included, wraps the renderer beneath the bridge, and `check`
prints the effort in use or blocks it on a renderer that takes none."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Any

import pytest
import tinker
import tinker_cookbook.renderers as renderers

from shipyard import resolved
from shipyard import serve as serving
from shipyard.config import ConfigError, Finding, check, load
from shipyard.proxy.rendering import AtEffort, at_effort, takes_effort
from tests.proxies import (
    TOOLS_OPENAI,
    BridgeRenderer,
    FakeRenderer,
    FakeSampler,
    ScriptedSampler,
    ask,
    endpoint,
)
from tests.trials import fixture_tasks, write_blueprint

INKLING = "thinkingmachines/Inkling-Small"
BLUEPRINT = """
[model]
name = "{model}"
[data]
dataset = "fixture"
batch_size = 1
[rollout]
harness = "pi@0.85.1"
{rollout}
[recipe]
kind = "evaluate"
"""


def blueprint(tmp_path: Path, model: str = INKLING, rollout: str = "") -> Path:
    return write_blueprint(tmp_path, BLUEPRINT.format(model=model, rollout=rollout))


class EffortRenderer(FakeRenderer):
    """The fake renderer with tml_v0's parameter, noting the effort of every build."""

    def __init__(self) -> None:
        self.efforts: list[float] = []

    def build_generation_prompt(
        self, messages: list[dict[str, Any]], effort: float = 0.9
    ) -> tinker.ModelInput:
        self.efforts.append(effort)
        return super().build_generation_prompt(messages)


# ------------------------------------------------------------------ the key


def test_effort_loads_within_tml_v0s_range(tmp_path: Path) -> None:
    assert load(blueprint(tmp_path)).rollout.effort is None, "the renderer's own"
    assert load(blueprint(tmp_path / "s", rollout="effort = 0.3")).rollout.effort == 0.3
    for bad in ("1.0", "-0.1"):
        with pytest.raises(ConfigError) as refused:
            load(blueprint(tmp_path / bad, rollout=f"effort = {bad}"))
        assert [p.split(":")[0] for p in refused.value.problems] == ["[rollout] effort"]


def test_serve_takes_the_effort_from_the_runs_settings() -> None:
    made = serving.endpoint_for(
        INKLING,
        weights=None,
        bind="127.0.0.1",
        port=0,
        advertise=None,
        renderer=None,
        settings={"effort": 0.3},
        token="k",
        control_token="c",
    )
    assert made.effort == 0.3


async def test_a_scripted_serve_starts_and_answers_at_the_runs_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fake renderer takes an effort and ignores it, so a scripted serve handed the
    run's `[rollout] effort` starts like one handed none."""
    monkeypatch.setenv(serving.FAKE_SAMPLER_ENV, "1")
    made = serving.endpoint_for(
        INKLING,
        weights=None,
        bind="127.0.0.1",
        port=0,
        advertise=None,
        renderer=None,
        settings={"effort": 0.3},
        token="k",
        control_token="c",
    )
    async with made as started:
        said = await ask(started, "t", messages=[{"role": "user", "content": "echo"}])
    assert said.status_code == 200, said.text
    assert said.json()["choices"][0]["message"]["content"] == "echo"


# ------------------------------------------------------------- the renderer wrap


def test_the_wrap_hands_the_effort_and_refuses_a_renderer_that_takes_none() -> None:
    inner = EffortRenderer()
    wrapped = at_effort(inner, 0.3)
    assert isinstance(wrapped, AtEffort) and wrapped.get_stop_sequences() == []
    wrapped.build_generation_prompt([{"role": "user", "content": "go"}])
    wrapped.build_generation_prompt([{"role": "user", "content": "go"}], effort=0.1)
    assert inner.efforts == [0.3, 0.1], "an effort the caller names wins"
    assert at_effort(inner, None) is inner
    plain = FakeRenderer()
    assert not takes_effort(plain) and takes_effort(inner)
    with pytest.raises(ValueError, match="takes no thinking effort"):
        at_effort(plain, 0.3)


class BridgeAtEffort(BridgeRenderer):
    """The bridge's fake renderer with tml_v0's parameter, noting every build's effort."""

    def __init__(self) -> None:
        self.efforts: list[float] = []

    def build_generation_prompt(
        self, messages: list[dict[str, Any]], effort: float = 0.9
    ) -> tinker.ModelInput:
        self.efforts.append(effort)
        return super().build_generation_prompt(messages)


async def test_the_proxy_and_its_bridge_render_every_prompt_at_the_runs_effort() -> None:
    inner = BridgeAtEffort()
    sampler = ScriptedSampler("", answers=["CALL ls {}", "done"])
    async with endpoint(sampler, renderer=inner, effort=0.3) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        first = (await ask(started, "t", messages=history, tools=TOOLS_OPENAI)).json()
        said = first["choices"][0]["message"]
        history.append({"role": "assistant", "content": None, "tool_calls": said["tool_calls"]})
        history.append(
            {"role": "tool", "tool_call_id": said["tool_calls"][0]["id"], "content": "a"}
        )
        assert (await ask(started, "t", messages=history, tools=TOOLS_OPENAI)).status_code == 200
        assert [r.bridged for r in started.records_for("t")] == [False, True]
    assert len(inner.efforts) >= 3, "the first prompt, then the bridge's head and full renders"
    assert set(inner.efforts) == {0.3}


async def test_the_proxy_refuses_an_effort_before_any_session_opens() -> None:
    made: list[str | None] = []

    async def factory(path: str | None) -> Any:
        made.append(path)
        return FakeSampler("ok")

    with pytest.raises(ValueError, match="takes no thinking effort"):
        await endpoint(renderer=FakeRenderer(), effort=0.3, client_factory=factory).start()
    assert made == [], "no sampling client was made"


# ------------------------------------------------------------- what check says


def test_check_prints_the_effort_tml_v0_renders_at_and_the_one_named(tmp_path: Path) -> None:
    found = resolved.resolved(load(blueprint(tmp_path)))["rollout"]
    assert (found["renderer"], found["effort"]) == ("tml_v0", 0.9)
    assert "\neffort = 0.9\n" in resolved.report(blueprint(tmp_path / "r")).text()
    named = resolved.resolved(load(blueprint(tmp_path / "n", rollout="effort = 0.3")))
    assert named["rollout"]["effort"] == 0.3
    qwen = resolved.resolved(load(blueprint(tmp_path / "q", model="Qwen/Qwen3-8B")))
    assert qwen["rollout"]["effort"] is None, "qwen3 takes no effort"
    assert "effort" not in resolved.report(blueprint(tmp_path / "qt", "Qwen/Qwen3-8B")).text()


def test_check_blocks_an_effort_on_a_renderer_that_takes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    found = check(blueprint(tmp_path / "q", model="Qwen/Qwen3-8B", rollout="effort = 0.3"))
    assert (
        Finding(
            "blocked",
            "[rollout] effort: the renderer qwen3 takes no thinking effort when it builds a "
            "prompt; where a model's effort levels are renderers, [rollout] renderer picks one",
        )
        in found
    )
    named = blueprint(
        tmp_path / "n", model="Qwen/Qwen3-8B", rollout='renderer = "tml_v0"\neffort = 0.3'
    )
    assert not [f for f in check(named) if "effort" in f.text]
    assert not [
        f for f in check(blueprint(tmp_path / "i", rollout="effort = 0.3")) if "effort" in f.text
    ]


def test_the_table_names_every_cookbook_renderer_that_takes_an_effort() -> None:
    """A renderer the cookbook adds with an `effort` parameter fails here, not in a run."""
    taking = set()
    for module in pkgutil.iter_modules(renderers.__path__, f"{renderers.__name__}."):
        loaded = importlib.import_module(module.name)
        for _, cls in inspect.getmembers(loaded, inspect.isclass):
            if cls.__module__ == loaded.__name__ and takes_effort(cls):
                taking.add((cls.__module__, cls.__name__))
    assert taking == set(resolved.PROMPT_EFFORT.values())
    source = inspect.getsource(renderers.get_renderer)
    for name, (_, cls) in resolved.PROMPT_EFFORT.items():
        assert re.search(rf'name == "{name}":\s+renderer = {cls}\(', source), name
