"""Tests of a trial's two limits: the sandboxes alive at once, from a sandbox's start to its
tests' end, and the trials sampling at once, from an agent's start to its end. An evaluation's
trials take a slot that frees before any other."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.trial.hooks import TrialEvent
from pydantic import ValidationError

from shipyard.config import Rollout
from shipyard.jobs.engine import GATE, Gate, HarborEngine, Job, Slots, _run_trial, _trial_config
from shipyard.jobs.harnesses import Harness
from shipyard.models.endpoints import ProviderEndpoint
from tests.fakes import write_result

PROVIDER = ProviderEndpoint("some-model", "anthropic")


class Phased:
    """Stands in for Harbor's trial, through a trial's life: it gets ready, waits at its agent's
    start for its turn to sample, samples, gives its turn back and runs its tests. It notes the
    most trials alive and sampling at once, and the job of each trial as it starts sampling."""

    def __init__(self, ready: float = 0.01, agent: float = 0.02, tests: float = 0.01) -> None:
        self.ready, self.agent, self.tests = ready, agent, tests
        self.alive = self.sampling = self.most_alive = self.most_sampling = 0
        self.sampled: list[str] = []

    async def __call__(self, config: TrialConfig) -> None:
        gate = GATE.get()
        assert gate is not None, "the engine gives every trial its gate"
        self.alive += 1
        self.most_alive = max(self.most_alive, self.alive)
        try:
            await asyncio.sleep(self.ready)
            await gate.start()
            self.sampling += 1
            self.most_sampling = max(self.most_sampling, self.sampling)
            self.sampled.append(Path(config.trials_dir).name)
            try:
                await asyncio.sleep(self.agent)
            finally:
                self.sampling -= 1
                await gate.end()
            await asyncio.sleep(self.tests)
            write_result(Path(config.trials_dir) / config.trial_name, reward=1.0)
        finally:
            self.alive -= 1


def _tasks(tmp_path: Path, *names: str) -> list[Path]:
    for name in names:
        (tmp_path / "tasks" / name).mkdir(parents=True, exist_ok=True)
    return [tmp_path / "tasks" / name for name in names]


def _engine(tmp_path: Path, runner: Any, **rollout: Any) -> HarborEngine:
    table = {"sandbox": "docker", "jobs_dir": str(tmp_path / "jobs"), **rollout}
    return HarborEngine(Rollout(**table), run_trial=runner)


def _job(name: str, purpose: str = "training") -> Job:
    return Job(name, purpose=purpose, harness=Harness(name="pi"))


async def test_a_slot_that_frees_goes_to_an_evaluation_first_then_in_the_order_asked() -> None:
    slots, order = Slots(1), []
    await slots.acquire()

    async def wait(name: str, first: bool) -> None:
        await slots.acquire(first)
        order.append(name)
        slots.release()

    waiting = [
        asyncio.create_task(wait(name, first))
        for name, first in [("training 1", False), ("training 2", False), ("evaluation", True)]
    ]
    await asyncio.sleep(0)
    slots.release()
    await asyncio.gather(*waiting)
    assert order == ["evaluation", "training 1", "training 2"]
    assert slots.used == 0


async def test_a_cancelled_wait_takes_no_slot_and_one_handed_over_is_passed_on() -> None:
    slots = Slots(1)
    await slots.acquire()
    gone = asyncio.create_task(slots.acquire())
    await asyncio.sleep(0)
    gone.cancel()
    await asyncio.gather(gone, return_exceptions=True)
    slots.release()
    assert slots.used == 0, "the cancelled wait took nothing"
    await slots.acquire()
    handed = asyncio.create_task(slots.acquire())
    await asyncio.sleep(0)
    slots.release()  # handed to the wait, which is cancelled before it resumes
    handed.cancel()
    await asyncio.gather(handed, return_exceptions=True)
    assert slots.used == 0, "the slot handed over was passed back"


async def test_trials_get_ready_and_run_their_tests_while_as_many_as_sample_at_once_do(
    tmp_path: Path,
) -> None:
    trials = Phased()
    engine = _engine(tmp_path, trials, concurrency=2, sandboxes=4)
    job = await engine.run(_job("j"), _tasks(tmp_path, *"abcdefgh"), 1, endpoint=PROVIDER)
    assert len(job.trials) == 8 and all(one.graded for one in job.trials)
    assert (trials.most_alive, trials.most_sampling) == (4, 2)


async def test_unset_sandboxes_are_as_many_as_sample_as_before(tmp_path: Path) -> None:
    trials = Phased()
    engine = _engine(tmp_path, trials, concurrency=2)
    await engine.run(_job("j"), _tasks(tmp_path, *"abcd"), 1, endpoint=PROVIDER)
    assert (trials.most_alive, trials.most_sampling) == (2, 2)


async def test_an_evaluations_trials_take_the_slots_that_free_before_trainings_waiting(
    tmp_path: Path,
) -> None:
    trials = Phased()
    engine = _engine(tmp_path, trials, concurrency=1)
    training = asyncio.create_task(
        engine.run(_job("training"), _tasks(tmp_path, *"abc"), 1, endpoint=PROVIDER)
    )
    await asyncio.sleep(0.005)  # the first training trial is getting ready, two wait
    await engine.run(_job("evaluation", "evaluation"), _tasks(tmp_path, "x"), 1, endpoint=PROVIDER)
    await training
    assert trials.sampled == ["training", "evaluation", "training", "training"]


async def test_harbors_trial_waits_for_its_turn_at_its_agents_start_and_gives_it_back_at_its_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[int] = []

    class Harbor:
        """Stands in for Harbor's trial: it calls the agent's hooks, as Harbor's run does."""

        def __init__(self, config: TrialConfig) -> None:
            self.hooks: dict[TrialEvent, Any] = {}

        @classmethod
        async def create(cls, config: TrialConfig) -> Harbor:
            return cls(config)

        def add_hook(self, event: TrialEvent, hook: Any) -> None:
            self.hooks[event] = hook

        async def run(self) -> str:
            event = SimpleNamespace()
            await self.hooks[TrialEvent.AGENT_START](event)
            held.append(slots.used)
            await self.hooks[TrialEvent.AGENT_END](event)
            held.append(slots.used)
            return "ran"

    monkeypatch.setattr("harbor.trial.trial.Trial", Harbor)
    task = tmp_path / "tasks" / "task"
    (task / "environment").mkdir(parents=True)
    config = _trial_config(
        task, trials_dir=tmp_path / "jobs", agent=AgentConfig(name="pi"), sandbox="docker"
    )
    slots = Slots(1)
    entered = GATE.set(Gate(slots))
    try:
        assert await _run_trial(config) == "ran"
    finally:
        GATE.reset(entered)
    assert held == [1, 0], "a slot held while the agent ran, and given back before the tests"


def test_sandboxes_below_concurrency_are_refused_and_unset_ones_are_as_many() -> None:
    with pytest.raises(ValidationError, match="sandboxes = 2 is below concurrency = 4"):
        Rollout(concurrency=4, sandboxes=2)
    assert Rollout(concurrency=4).alive == 4 and Rollout(concurrency=4, sandboxes=8).alive == 8
