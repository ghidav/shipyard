"""Tests for a model's renderers, such as reasoning efforts: its `[models.<role>]` table names them,
its lane in the proxy holds them, and each rollout of it picks one, which its trials are opened
with."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness_proxy.wire import CONTROL_TOKEN_ENV, Served
from shipyard.config import Config, ConfigError, load
from shipyard.models.endpoints import ProxyEndpoint
from shipyard.run import files
from tests.fakes import Alternating, FakeProxy, FakeTrials, Harbor
from tests.runs import GRPO, ONE_STEP, rows, run_toml, train
from tests.service import FakeService

LOW, HIGH = "qwen3_8_low_reasoning", "qwen3_8_xhigh_reasoning"
#: The line of `[models.agent]` that its renderers are written after.
AGENT = "lora_rank = 32"


def rendered(given: str) -> dict[str, str]:
    """The config's `[models.agent]` with `renderers = <given>`."""
    return {AGENT: f"{AGENT}\nrenderers = {given}"}


EFFORTS = rendered(f'["{LOW}", "{HIGH}"]')


def loaded(tmp_path: Path, **replacements: str) -> Config:
    """Load the GRPO config over the four-task dataset with `replacements` made in it."""
    return load(run_toml(tmp_path, **replacements), GRPO.Recipe)


@pytest.fixture(autouse=True)
def _no_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harbor's own check of each harness's config, which is not what these tests are about."""
    monkeypatch.setattr("shipyard.jobs.engine.preflight", lambda *_: None)


# ------------------------------------------------------------------ the config


def test_a_model_takes_a_renderer_or_several_each_named_once(tmp_path: Path) -> None:
    assert loaded(tmp_path).models["agent"].renderers == [], "none: the model's own"
    assert loaded(tmp_path, **rendered('""')).models["agent"].renderers == []
    assert loaded(tmp_path, **rendered(f'"{LOW}"')).models["agent"].renderers == [LOW]
    assert loaded(tmp_path, **EFFORTS).models["agent"].renderers == [LOW, HIGH]
    for given, problem in (
        (f'["{LOW}", "{LOW}"]', "each renderer is named once"),
        ("[1]", "a renderer or a list of them"),
    ):
        with pytest.raises(ConfigError) as refused:
            loaded(tmp_path, **rendered(given))
        assert refused.value.problems == [f"[models.agent] renderers: {problem}"]


def test_the_rollouts_renderer_is_refused_where_it_moved(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as refused:
        loaded(tmp_path, **{"concurrency = 2": f'concurrency = 2\nrenderer = "{LOW}"'})
    assert refused.value.problems == [
        "[rollout] renderer: unknown key: it moved, to [models.<role>] renderers"
    ]


def test_a_providers_model_takes_no_renderer(tmp_path: Path) -> None:
    claude = {'model = "Qwen/Qwen3-8B"': 'model = "claude-sonnet-5-5"\nprovider = "anthropic"'}
    config = loaded(tmp_path, **claude, **EFFORTS)
    with pytest.raises(ValueError, match="renders the calls of a model on Tinker or Together"):
        GRPO(config, tmp_path / "run")
    assert not (tmp_path / "run").exists(), "refused before the run directory is made"


# ------------------------------------------------------------------ a rollout's renderer


async def test_a_rollout_names_a_renderer_its_model_holds_and_must_when_it_holds_several(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, _ = fakes
    trials = harbor(FakeTrials())

    class Unnamed(GRPO):
        async def step(self) -> None:
            tasks, agent = self.draw(), {"model": self.agent, "harness": "pi", "modules": None}
            with pytest.raises(ValueError, match=f"name its renderer, one of {LOW}, {HIGH}"):
                await self.rollout(tasks, attempts=1, **agent)
            with pytest.raises(
                ValueError,
                match=rf"rollout\(renderer='medium'\): \[models.agent\] renderers holds {LOW}",
            ):
                await self.rollout(tasks, attempts=1, renderer="medium", **agent)
            assert trials.configs == [], "refused before any trial runs"

    several = await train(run_toml(tmp_path, **EFFORTS, **ONE_STEP), tmp_path / "runs", Unnamed)
    assert list(files.read(several / files.JOBS)) == [], "no job was named"
    home = run_toml(tmp_path, **rendered(f'"{LOW}"'), **ONE_STEP)
    single = await train(home, tmp_path / "runs")
    assert [job["renderer"] for job in files.read(single / files.JOBS)] == [LOW], "its only one"
    assert [renderer for _, _, renderer, _ in proxy.opened] == [LOW] * 4


async def test_a_models_lane_holds_its_renderers_and_each_trial_opens_on_the_one_its_rollout_named(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, _ = fakes
    trials = harbor(Alternating())

    class Efforts(GRPO):
        async def step(self) -> None:
            tasks = self.draw()
            both = await asyncio.gather(
                *(
                    self.rollout(
                        tasks,
                        model=self.agent,
                        harness=self.recipe.harness,
                        attempts=self.recipe.group_size,
                        modules=None,
                        renderer=one,
                    )
                    for one in (LOW, HIGH)
                )
            )
            for groups, renderer in zip(both, (LOW, HIGH), strict=True):
                assert {group.renderer for group in groups} == {renderer}

    home = run_toml(tmp_path, **EFFORTS, **ONE_STEP)
    directory = await train(home, tmp_path / "runs", Efforts)
    (lane,) = proxy.placed[0][1]["lanes"]
    assert (lane["name"], lane["renderer"]) == ("agent", [LOW, HIGH]), "one lane, with both"
    jobs = {job["job"]: job["renderer"] for job in files.read(directory / files.JOBS)}
    assert sorted(jobs.values()) == sorted([LOW, HIGH]), "a job at each renderer"
    opened = {name: renderer for name, _, renderer, _ in proxy.opened}
    for config in trials.configs:
        assert opened[config.trial_name] == jobs[Path(config.trials_dir).name]
    (row,) = rows(directory)
    assert (row["groups"], row["trials"], row["reward_mean"]) == (4, 8, 0.5)
    assert not any(LOW in key or HIGH in key for key in row), "no breakdown by renderer"


# ------------------------------------------------------------------ a remote proxy


async def test_a_remote_proxy_that_lacks_a_named_renderer_is_refused_before_any_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    remote = {"concurrency = 2": 'concurrency = 2\nendpoint_url = "https://proxy.example"'}
    config = loaded(tmp_path, **EFFORTS, **remote)
    table = config.models["agent"]
    endpoint = ProxyEndpoint(config, tmp_path, environ={CONTROL_TOKEN_ENV: "c"})
    lane = endpoint.lane("agent", table.model, renderers=table.renderers)

    def renders(*names: str) -> dict[str, Served]:
        return {"agent": Served(lane.model, None, names)}

    proxy.lanes = renders(LOW)
    match = rf"renders lane agent with {LOW}, and \[models.agent\] renderers names {HIGH}"
    with pytest.raises(RuntimeError, match=match):
        await endpoint.open()
    proxy.lanes = renders()
    with pytest.raises(RuntimeError, match="with its model's default alone"):
        await endpoint.open()
    proxy.lanes = renders(HIGH, LOW, "other")
    await endpoint.open()
    assert endpoint.proxy is proxy, "a proxy that renders with each of them is taken"
