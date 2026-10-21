"""Tests for training at several renderers, such as reasoning efforts: how a rollout splits its
tasks between them, how each trial reaches its own, and what the rows say of each."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, Served
from shipyard import Trainer, load
from shipyard.config import ConfigError
from shipyard.jobs.engine import split
from shipyard.run import files
from shipyard.run.logs import AGENT
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.runs import ONE_STEP, blueprint, rows, train
from tests.service import FakeService
from tests.test_endpoints import agent

LOW, HIGH = "qwen3_8_low_reasoning", "qwen3_8_xhigh_reasoning"
ROLLOUT = "concurrency = 2"


def rendered(given: str) -> dict[str, str]:
    """The blueprint's `[rollout]` with `renderer = <given>`."""
    return {ROLLOUT: f"{ROLLOUT}\nrenderer = {given}"}


EFFORTS = rendered(f'["{LOW}", "{HIGH}"]')


# ------------------------------------------------------------------ the blueprint


def test_the_rollout_takes_a_renderer_or_several_each_named_once(tmp_path: Path) -> None:
    assert load(blueprint(tmp_path)).rollout.renderer == [], "none: the model's own"
    assert load(blueprint(tmp_path, **rendered('""'))).rollout.renderer == []
    assert load(blueprint(tmp_path, **rendered(f'"{LOW}"'))).rollout.renderer == [LOW]
    assert load(blueprint(tmp_path, **EFFORTS)).rollout.renderer == [LOW, HIGH]
    for given, problem in (
        (f'["{LOW}", "{LOW}"]', "each renderer is named once"),
        ('["a/b"]', "a renderer is named in its trials' addresses, so it holds no '/'"),
        ("[1]", "a renderer or a list of them"),
    ):
        with pytest.raises(ConfigError) as refused:
            load(blueprint(tmp_path, **rendered(given)))
        assert refused.value.problems == [f"[rollout] renderer: {problem}"]


def test_a_providers_agent_takes_no_renderer(tmp_path: Path) -> None:
    claude = {'model = "Qwen/Qwen3-8B"': 'model = "claude-sonnet-5-5"\nprovider = "anthropic"'}
    config = load(blueprint(tmp_path, **claude, **EFFORTS))
    with pytest.raises(ValueError, match="renders the calls of a model on Tinker or Together"):
        Trainer(config, tmp_path / "run")
    assert not (tmp_path / "run").exists(), "refused before the run directory is made"


# ------------------------------------------------------------------ the jobs a rollout runs


def test_a_rollout_runs_every_task_at_every_harness_and_renderer() -> None:
    a, b = Path("a"), Path("b")
    assert split([a, b], 4, ["pi", "cc"], [LOW, HIGH], mix=False) == [
        ("pi", LOW, [a, b], 4),
        ("pi", HIGH, [a, b], 4),
        ("cc", LOW, [a, b], 4),
        ("cc", HIGH, [a, b], 4),
    ], "a group per task, renderer and harness, of four"


def test_mixed_harnesses_share_a_groups_attempts_at_each_renderer() -> None:
    a, b = Path("a"), Path("b")
    assert split([a, b], 4, ["pi", "cc"], [LOW, HIGH], mix=True) == [
        ("pi", LOW, [a, b], 2),
        ("pi", HIGH, [a, b], 2),
        ("cc", LOW, [a, b], 2),
        ("cc", HIGH, [a, b], 2),
    ], "a group per task and renderer, of four, two from each harness"


# ------------------------------------------------------------------ a run at two renderers


async def test_a_run_trains_and_evaluates_at_each_renderer_and_says_which(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, _ = fakes
    trials = harbor(Alternating())
    with_test = {"group_size = 2": 'group_size = 2\ntest = "four"\ntest_group_size = 1'}
    home = blueprint(tmp_path, **EFFORTS, **ONE_STEP, **with_test)
    directory = await train(home, tmp_path / "runs")
    (lane,) = proxy.placed[0][1]["lanes"]
    assert lane["renderer"] == [LOW, HIGH], "one proxy, the agent's lane started with both"
    jobs = {job["job"]: job for job in files.read(directory / files.JOBS)}
    trained = [job["renderer"] for job in jobs.values() if job["purpose"] == "training"]
    assert trained == [LOW, HIGH], "the batch's two tasks, one at each"
    tested = [job["renderer"] for job in jobs.values() if job["purpose"] == "evaluation"]
    assert sorted(tested) == sorted([LOW, HIGH] * 2), "each test batch at each renderer"
    for config in trials.configs:
        renderer = jobs[Path(config.trials_dir).name]["renderer"]
        address = config.agent.env["OPENAI_BASE_URL"]
        assert address.endswith(f"/renderer/{renderer}/v1"), "each trial dials its own"
    (row,) = rows(directory)
    for one in (LOW, HIGH):
        assert (row[f"reward_mean_{one}"], row[f"sampled_mean_{one}"]) == (0.5, 2.0)
    assert (row["eval_tasks"], row["eval_trials"], row["eval_graded"]) == (4, 8, 8)
    assert {f"eval_reward_mean_{LOW}", f"eval_reward_mean_{HIGH}"} <= set(row)


async def test_a_recipe_may_roll_out_at_one_renderer_and_no_other(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(Alternating())

    class Pinned(Trainer):
        async def step(self) -> None:
            groups = await self.rollout(self.draw(), renderer=HIGH)
            assert {one.renderer for group in groups for one in group.trials} == {HIGH}
            with pytest.raises(ValueError, match=r"rollout\(renderer='medium'\): the proxy"):
                await self.rollout(self.draw(), renderer="medium")

    directory = await train(blueprint(tmp_path, **EFFORTS, **ONE_STEP), tmp_path / "runs", Pinned)
    assert [job["renderer"] for job in files.read(directory / files.JOBS)] == [HIGH]
    (row,) = rows(directory)
    assert not any(key.startswith("reward_mean_") for key in row), "one renderer: no breakdown"


async def test_a_remote_proxy_that_lacks_a_named_renderer_is_refused_before_any_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    remote = rendered(f'["{LOW}", "{HIGH}"]\nendpoint_url = "https://proxy.example"')
    lane = agent(
        load(blueprint(tmp_path, **remote)),
        tmp_path,
        environ={PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"},
    )
    endpoint = lane.proxy

    def renders(*names: str) -> dict[str, Served]:
        return {AGENT: Served(lane.model, lane.model, None, names)}

    proxy.lanes = renders(LOW)
    match = rf"renders lane agent with {LOW}, and \[rollout\] renderer names {HIGH}"
    with pytest.raises(RuntimeError, match=match):
        await endpoint.open()
    proxy.lanes = renders()
    with pytest.raises(RuntimeError, match="with its model's default alone"):
        await endpoint.open()
    proxy.lanes = renders(HIGH, LOW, "other")
    await endpoint.open()
    assert endpoint.proxy is proxy, "a proxy that renders with each of them is taken"
