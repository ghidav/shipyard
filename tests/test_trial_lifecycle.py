"""Tests for a trial's life on the run's proxy, as the engine leads it: opened in its slot just
before Harbor starts it, closed after Harbor is done with it for its records, and a trial that
cannot be opened stopping its job, the trials beside it included."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from tests.fakes import FakeProxy, FakeTrials
from tests.test_engine import _batch, _engine, _run, lane


class Refusing(FakeProxy):
    """Fake proxy that refuses to open the trials of `task`, as a proxy that cannot serve the
    weights refuses."""

    def __init__(self, task: str) -> None:
        super().__init__()
        self.task = task

    async def open(self, trial: str, **named: Any) -> str:
        if trial.startswith(f"{self.task}__"):
            raise RuntimeError(f"the proxy refused to open trial {trial}: 400")
        return await super().open(trial, **named)


class Waiting(FakeTrials):
    """Fake Harbor that holds every trial until it is cancelled, and notes the ones that were."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled: list[str] = []

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled.append(config.trial_name)
            raise


async def test_a_trial_harbor_could_not_run_is_still_closed_for_its_records(
    tmp_path: Path,
) -> None:
    proxy = FakeProxy()
    endpoint = lane(tmp_path, proxy)
    batch = _batch(tmp_path, "alpha", "beta")
    job = await _run(
        _engine(tmp_path, FakeTrials(breaks=("beta",))), "j", batch, 1, endpoint=endpoint
    )
    alpha, beta = job.trials
    assert sorted(proxy.closed) == sorted([alpha.name, beta.name]), "both closed"
    assert beta.records.recorded and len(beta.records) == 1, "its records came back with it"
    assert endpoint.trials == set(), "none left open"


async def test_a_trial_that_cannot_be_opened_stops_its_job_and_the_trials_beside_it(
    tmp_path: Path,
) -> None:
    proxy, harbor = Refusing("beta"), Waiting()
    endpoint = lane(tmp_path, proxy)
    engine = _engine(tmp_path, harbor)
    batch = _batch(tmp_path, "alpha", "beta")
    with pytest.raises(RuntimeError, match="refused to open trial beta"):
        await _run(engine, "j", batch, 1, endpoint=endpoint)
    (alpha,) = [name for name, *_ in proxy.opened]
    assert alpha.startswith("alpha__") and harbor.cancelled == [alpha], "stopped, not left running"
    assert proxy.closed == [], "a run that is ending closes nothing more"
    assert [c.trial_name for c in harbor.configs] == [alpha], "beta never started"
