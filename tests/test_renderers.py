"""Tests for training at several renderers, such as reasoning efforts: how a rollout splits its
tasks between them, how each trial reaches its own, and what the rows say of each."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV
from shipyard import Trainer, load
from shipyard.config import ConfigError
from shipyard.models.endpoints import ProxyEndpoint
from shipyard.run import record
from shipyard.trainer import split
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.test_trainer import ONE_STEP, _rows, _train, blueprint
from tests.trainers import FakeService

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
    with pytest.raises(ValueError, match="the proxy renders the calls of a model the run trains"):
        Trainer(config, tmp_path / "run")
    assert not (tmp_path / "run").exists(), "refused before the run directory is made"


# ------------------------------------------------------------------ the jobs a rollout runs


def test_the_tasks_take_the_renderers_in_turn_so_each_group_is_one_renderers() -> None:
    a, b, c = Path("a"), Path("b"), Path("c")
    efforts = [LOW, HIGH]
    assert split([a, b, c], 4, ["pi"], efforts, mix=False, turn=0) == [
        ("pi", LOW, [a, c], 4),
        ("pi", HIGH, [b], 4),
    ]
    assert split([a, b, c], 4, ["pi"], efforts, mix=False, turn=1) == [
        ("pi", LOW, [b], 4),
        ("pi", HIGH, [a, c], 4),
    ], "turns go on by step"


def test_unmixed_harnesses_and_renderers_take_one_turn_over_every_pair() -> None:
    met = set()
    for turn in range(4):
        [(one, renderer, _, attempts)] = split(
            [Path("a")], 2, ["pi", "cc"], [LOW, HIGH], mix=False, turn=turn
        )
        met.add((one, renderer, attempts))
    assert met == {("pi", LOW, 2), ("pi", HIGH, 2), ("cc", LOW, 2), ("cc", HIGH, 2)}, (
        "across steps, a task meets every harness at every renderer"
    )


def test_mixed_harnesses_share_a_tasks_attempts_at_its_renderer() -> None:
    a, b = Path("a"), Path("b")
    assert split([a, b], 4, ["pi", "cc"], [LOW, HIGH], mix=True, turn=0) == [
        ("pi", LOW, [a], 2),
        ("cc", LOW, [a], 2),
        ("pi", HIGH, [b], 2),
        ("cc", HIGH, [b], 2),
    ]


# ------------------------------------------------------------------ a run at two renderers


async def test_a_run_trains_and_evaluates_at_each_renderer_and_says_which(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, _ = fakes
    trials = harbor(Alternating())
    with_test = {"group_size = 2": 'group_size = 2\ntest = "four"\ntest_group_size = 1'}
    home = blueprint(tmp_path, **EFFORTS, **ONE_STEP, **with_test)
    directory = await _train(home, tmp_path / "runs")
    assert proxy.placed[0][1]["renderer"] == [LOW, HIGH], "one proxy, started with both"
    jobs = {job["job"]: job for job in record.read(directory / record.JOBS)}
    trained = [job["renderer"] for job in jobs.values() if job["purpose"] == "training"]
    assert trained == [LOW, HIGH], "the batch's two tasks, one at each"
    tested = [job["renderer"] for job in jobs.values() if job["purpose"] == "evaluation"]
    assert sorted(tested) == sorted([LOW, HIGH] * 2), "each test batch at each renderer"
    for config in trials.configs:
        renderer = jobs[Path(config.trials_dir).name]["renderer"]
        address = config.agent.env["OPENAI_BASE_URL"]
        assert address.endswith(f"/renderer/{renderer}/v1"), "each trial dials its own"
    (row,) = _rows(directory)
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

    directory = await _train(blueprint(tmp_path, **EFFORTS, **ONE_STEP), tmp_path / "runs", Pinned)
    assert [job["renderer"] for job in record.read(directory / record.JOBS)] == [HIGH]
    (row,) = _rows(directory)
    assert not any(key.startswith("reward_mean_") for key in row), "one renderer: no breakdown"


async def test_a_remote_proxy_that_lacks_a_named_renderer_is_refused_before_any_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = FakeProxy(renderers=[LOW])
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    remote = rendered(f'["{LOW}", "{HIGH}"]\nendpoint_url = "https://proxy.example"')
    endpoint = ProxyEndpoint(
        load(blueprint(tmp_path, **remote)),
        tmp_path,
        environ={PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"},
    )
    with pytest.raises(RuntimeError, match=rf"renders with {LOW}, and \[rollout\] renderer names"):
        await endpoint.open()
    proxy.renderers = None
    with pytest.raises(RuntimeError, match="does not say which renderers it renders with"):
        await endpoint.open()
    proxy.renderers = [HIGH, LOW, "other"]
    await endpoint.open()
    assert endpoint.proxy is proxy, "a proxy that renders with each of them is taken"
