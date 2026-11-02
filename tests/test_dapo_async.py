"""Tests for the shipped dapo-async recipe: workers roll a task out `group_size` times as one job
and hand its group over once its last trial ends; each step takes the groups in the order they
finished until a batch's worth have rewards that differ, at most a pass over the tasks, scores
them as dapo does, and updates. The fake proxy records two sampled tokens a trial."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard.groups import Group, Trial
from shipyard.jobs.admit import Verdict
from shipyard.jobs.harnesses import Harness
from shipyard.run.data import Lender, tasks
from shipyard.run.directory import open_run, trainer_class
from tests.fakes import FakeProxy, Harbor, write_result
from tests.runs import ONE_STEP, rows, run_toml, train
from tests.service import FakeService

TRAINER = Path(__file__).parents[1] / "recipes" / "dapo-async" / "trainer.py"
#: The recipe's own keys beyond the tests' config: one optimizer step, no overlong penalty.
KEYS = (
    "substeps = 1",
    "clip_low = 0.2",
    "clip_high = 0.28",
    "overlong_penalty = 0",
    "overlong_buffer = 0.2",
)


class Scripted:
    """Fake Harbor trial runner: a task's trials score the rewards `script` lists for it, in call
    order and round again; 1 for a task it does not name."""

    def __init__(self, script: dict[str, list[float]]) -> None:
        self.script = script
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        task = Path(str(config.task.path)).name
        made = self.seen[task] = self.seen.get(task, 0) + 1
        rewards = self.script.get(task, [1.0])
        reward = rewards[(made - 1) % len(rewards)]
        write_result(Path(config.trials_dir) / config.trial_name, reward=reward)


def config(tmp_path: Path, *keys: str, **replacements: str) -> Path:
    return run_toml(tmp_path, *KEYS, *keys, **replacements)


def drawn(tmp_path: Path, n: int) -> list[str]:
    """The first `n` tasks the run's lender gives out, one at a time, as its workers draw them."""
    lender = Lender(tasks("four", root=tmp_path / "tasks"), 2, 0)
    return [lender.draw(1)[0].name for _ in range(n)]


def advantages(service: FakeService) -> list[float]:
    """The advantages the updates sent, at every token that has one, in order."""
    return sorted(
        value
        for one in service.client.sent
        if "advantages" in one.loss_fn_inputs
        for value in one.loss_fn_inputs["advantages"].tolist()
        if value
    )


async def test_a_step_trains_on_the_next_groups_to_finish_each_normalized_within_itself(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """One worker hands two groups over in turn, each a solved and a failed trial: each trial's
    advantage is its reward less its group's mean, over their spread, at its every sampled
    token."""
    _, service = fakes
    home = config(tmp_path, "in_flight = 1", **ONE_STEP)
    first, second = drawn(tmp_path, 2)
    harbor(Scripted({first: [1.0, 0.0], second: [1.0, 0.0]}))
    directory = await train(home, tmp_path / "runs", TRAINER)
    assert advantages(service) == pytest.approx([-1.0] * 4 + [1.0] * 4, abs=1e-4)
    (row,) = rows(directory)
    assert row["trained"] and row["loss"] == "ppo"
    assert (row["groups"], row["lag_max"], row["stale"]) == (2, 0, 0)


async def test_a_step_takes_groups_until_a_batchs_worth_have_rewards_that_differ(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """The first group to finish is all solved: the step takes it and two more, which differ."""
    home = config(tmp_path, "in_flight = 1", **ONE_STEP)
    first, second, third = drawn(tmp_path, 3)
    harbor(Scripted({first: [1.0], second: [1.0, 0.0], third: [0.0, 1.0]}))
    directory = await train(home, tmp_path / "runs", TRAINER)
    (row,) = rows(directory)
    assert (row["groups"], row["uniform"]) == (3, 1) and row["trained"]


async def test_a_step_takes_at_most_a_pass_over_the_tasks(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """Every trial is solved, so no group differs: the step stops after the four tasks' groups,
    and trains nothing."""
    home = config(tmp_path, "in_flight = 1", **ONE_STEP)
    harbor(Scripted({}))
    directory = await train(home, tmp_path / "runs", TRAINER)
    (row,) = rows(directory)
    assert (row["groups"], row["uniform"]) == (4, 4) and not row["trained"]


def graded(task: Path, tmp_path: Path, rewards: list[float]) -> Group:
    """A group of `task`'s trials, graded with `rewards`."""
    trials = [
        Trial(
            "j",
            task,
            Harness(name="pi"),
            tmp_path / "jobs" / "j" / f"{task.name}__{n}",
            verdict=Verdict(reward, None, None),
        )
        for n, reward in enumerate(rewards)
    ]
    return Group(task, trials)


async def test_a_group_past_max_lag_is_left_out_and_its_task_goes_out_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # where the run finds tasks/four
    directory, loaded = open_run(
        config(tmp_path, "max_lag = 1"), root=tmp_path / "runs", trainer=TRAINER
    )
    with trainer_class(directory)(loaded, directory) as trainer:
        lent = trainer.lender.draw(3)
        groups = [graded(task, tmp_path, [1.0, 0.0]) for task in lent]
        lags = dict(zip(groups, [0, 2, 1], strict=True))
        trainer.agent.lag = lags.__getitem__  # how far each group lags the weights served
        finished: asyncio.Queue[Group] = asyncio.Queue()
        for one in groups:
            finished.put_nowait(one)
        taken, batch, stale = await trainer.take(finished)
        assert (taken, batch, stale) == (groups, [groups[0], groups[2]], 1)
        assert trainer.lender.draw(1) == [lent[1]], "the stale group's task goes out first"


@pytest.mark.parametrize(("keys", "expected"), [((), 1), (("in_flight = 5",), 5)])
def test_in_flight_defaults_to_half_again_the_groups_the_sandboxes_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keys: tuple[str, ...], expected: int
) -> None:
    """Two sandboxes hold one group of two: half again is one, rounded down."""
    monkeypatch.chdir(tmp_path)
    directory, loaded = open_run(config(tmp_path, *keys), root=tmp_path / "runs", trainer=TRAINER)
    with trainer_class(directory)(loaded, directory) as trainer:
        assert trainer.in_flight == expected
    wide = config(tmp_path, **{"concurrency = 2": "concurrency = 2\nsandboxes = 8"})
    directory, loaded = open_run(wide, root=tmp_path / "runs", trainer=TRAINER)
    with trainer_class(directory)(loaded, directory) as trainer:
        assert trainer.in_flight == 6, "eight sandboxes hold four groups of two: six"
