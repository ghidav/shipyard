"""The dead-endpoint stop, once in `Run.sample` for every recipe: a served job none of whose
trials reached the proxy, or whose every request failed at the sampler (the weights expired
under the run), is recorded and then stops the run with `NothingServed`; a job where some
trials were answered is masked trial by trial; a provider-served job is never this; and a step
whose groups are all degenerate is an untrained step, not a dead endpoint."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from shipyard import record, serving, session
from shipyard.recipes import run_recipe
from shipyard.run import Run
from shipyard.serving import NothingServed, served_nothing
from tests.records import FakeProxy, made
from tests.test_gepa_recipe import BLUEPRINT as GEPA
from tests.test_gepa_recipe import VALID
from tests.trainers import FakeService, capable
from tests.trials import FIXTURES, FakeTrials, fixture_tasks, write_blueprint

BLUEPRINTS = Path(__file__).parent / "blueprints"
MODEL = "Qwen/Qwen3-8B"
DEAD = "no trial reached the proxy; the endpoint is dead or unreachable"
POISONED = "every request failed at the sampler (sampler: BadRequestError)"


def _four(tmp_path: Path, kind: str, **replacements: str) -> Path:
    """The kind's fixture over a four-task dataset, two tasks per batch, groups of two."""
    for name in ("a", "b", "c", "d"):
        shutil.copytree(FIXTURES / "fixture" / "alpha", tmp_path / "tasks" / "four" / name)
    text = (BLUEPRINTS / kind / "run.toml").read_text(encoding="utf-8")
    text = text.replace('"aime-train"', '"four"').replace('"fixture"', '"four"')
    text = text.replace("group_size = 4", "group_size = 2")
    text = text.replace("batch_size = 1", "batch_size = 2")
    for old, new in replacements.items():
        text = text.replace(old, new)
    home = tmp_path / "bp" / kind
    home.mkdir(parents=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


@pytest.fixture
def proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    """A proxy nothing reaches: every trial's records come back empty."""
    monkeypatch.chdir(tmp_path)
    fake = FakeProxy(default=[])
    monkeypatch.setattr(serving, "Proxy", fake)
    return fake


SERVED_EVALUATE = {
    'provider = "openrouter"': 'provider = "tinker"',
    'sandbox = "modal"': 'sandbox = "docker"',
}


async def test_an_evaluate_job_nothing_reached_is_recorded_and_stops_the_run(
    tmp_path: Path, proxy: FakeProxy
) -> None:
    opened = Run.open(_four(tmp_path, "evaluate", **SERVED_EVALUATE), root=tmp_path / "runs")
    opened.run_trial = FakeTrials()
    with pytest.raises(NothingServed, match=DEAD), opened:
        await run_recipe(opened)
    note = Run.read(opened.directory)
    assert note["failed"] is True
    assert note["error"] == f"NothingServed: job {opened.id}-0000: {DEAD}"
    [job] = list(record.read(opened.directory / record.JOBS))
    assert job["masked"] == {"env_error": 4} and job["graded"] == 0 and job["served"] == "tinker"
    assert not (opened.directory / record.METRICS).exists(), "stopped before any row"
    assert len(opened.run_trial.configs) == 4, "the next batch was never paid for"  # type: ignore[union-attr]
    assert proxy.closed == 1


async def test_a_job_where_some_trials_reached_the_proxy_masks_the_rest_and_goes_on(
    tmp_path: Path, proxy: FakeProxy
) -> None:
    """`a`'s two rollouts reached nothing; whichever batch it lands in, its partner did."""
    proxy.answers = {"b": [made()], "c": [made()], "d": [made()]}
    opened = Run.open(_four(tmp_path, "evaluate", **SERVED_EVALUATE), root=tmp_path / "runs")
    opened.run_trial = FakeTrials()
    with opened:
        await run_recipe(opened)
    assert Run.read(opened.directory)["failed"] is False
    jobs = list(record.read(opened.directory / record.JOBS))
    assert len(jobs) == 2
    assert sum(job.get("masked", {}).get("env_error", 0) for job in jobs) == 2


async def test_a_provider_served_job_with_no_result_is_masked_not_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    opened = Run.open(_four(tmp_path, "evaluate"), root=tmp_path / "runs")
    opened.run_trial = FakeTrials(breaks=("a", "b", "c", "d"))
    with opened:
        await run_recipe(opened)
    assert Run.read(opened.directory)["failed"] is False
    assert len(list(record.read(opened.directory / record.JOBS))) == 2


async def test_a_job_whose_every_request_failed_at_the_sampler_stops_the_run(
    tmp_path: Path, proxy: FakeProxy
) -> None:
    """d2ab39b's night: the proxy is reached, but expired weights poisoned its client."""
    proxy.default = [made(error="sampler: BadRequestError", seq=n) for n in (1, 2)]
    opened = Run.open(_four(tmp_path, "evaluate", **SERVED_EVALUATE), root=tmp_path / "runs")
    opened.run_trial = FakeTrials()
    with pytest.raises(NothingServed, match=re.escape(POISONED)), opened:
        await run_recipe(opened)
    assert Run.read(opened.directory)["error"].startswith(
        f"NothingServed: job {opened.id}-0000: {POISONED}; the weights it serves are gone"
    )
    assert len(list(record.read(opened.directory / record.JOBS))) == 1
    assert len(opened.run_trial.configs) == 4, "the next batch was never paid for"  # type: ignore[union-attr]


def test_served_nothing_is_about_records_not_rewards(tmp_path: Path) -> None:
    from shipyard.rollout import Rollouts

    trials = [tmp_path / "a__0000000", tmp_path / "b__0000001"]

    def why(**records: list) -> str | None:
        return served_nothing(Rollouts("j", trials, [], records=records))

    assert why(a__0000000=[], b__0000001=[]) == DEAD
    failed = [made(error="sampler: ConnectError"), made(error="sampler: BadRequestError")]
    assert why(a__0000000=failed) == (
        "every request failed at the sampler (sampler: BadRequestError, sampler: ConnectError); "
        "the weights it serves are gone or the backend refuses them"
    )
    assert why(a__0000000=[made(error="context")]) is None, "a refusal is an answer"
    assert why(a__0000000=failed, b__0000001=[made()]) is None, "one was served"
    assert served_nothing(Rollouts("j", trials, [])) is None, "not harvested: not served"
    assert served_nothing(Rollouts("j", [], [], records={})) is None


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeService:
    made_service = FakeService(capabilities=capable((MODEL, True)))
    monkeypatch.setattr(session, "service_client", lambda metadata=None: made_service)
    return made_service


async def test_a_dapo_step_nothing_reached_stops_before_credit_and_closes_as_errored(
    tmp_path: Path, proxy: FakeProxy, service: FakeService
) -> None:
    opened = Run.open(_four(tmp_path, "dapo"), root=tmp_path / "runs")
    opened.run_trial = FakeTrials()
    with pytest.raises(NothingServed), opened:
        await run_recipe(opened)
    assert not any(c[0] == "forward_backward" for c in service.client.calls)
    assert service.closed == [("errored", f"NothingServed: job {opened.id}-0000: {DEAD}")]
    assert not (opened.directory / record.METRICS).exists()
    assert not (opened.directory / record.CHECKPOINTS).exists(), "no final checkpoint"
    assert len(list(record.read(opened.directory / record.JOBS))) == 1


async def test_a_step_whose_groups_are_all_degenerate_is_untrained_not_dead(
    tmp_path: Path, proxy: FakeProxy, service: FakeService
) -> None:
    proxy.default = None  # one record per trial: served, every reward 1
    home = _four(tmp_path, "dapo", **{"batch_size = 2": "batch_size = 4"})
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = FakeTrials()
    with opened:
        await run_recipe(opened)
    [row] = list(record.read(opened.directory / record.METRICS))
    assert row["trained"] is False and Run.read(opened.directory)["failed"] is False


async def test_a_gepa_search_whose_seed_nothing_reached_stops(
    tmp_path: Path, proxy: FakeProxy
) -> None:
    fixture_tasks(tmp_path)
    text = GEPA.replace('provider = "openrouter"', 'provider = "tinker"')
    home = write_blueprint(tmp_path, text)
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = FakeTrials()
    with pytest.raises(NothingServed, match=DEAD), opened:
        await run_recipe(opened)
    [job] = list(record.read(opened.directory / record.JOBS))
    assert job["purpose"] == "rollout" and job["batch"] == 0
    assert not (opened.directory / record.METRICS).exists()
