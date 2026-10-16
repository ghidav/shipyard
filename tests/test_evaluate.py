"""Test the evaluate kind through the runner: one row for the test set, one job per batch."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipyard import record
from shipyard.cli.run import Run
from shipyard.sampling.admit import BUDGET
from shipyard.training.recipes import trainer_for
from tests.records import FakeProxy, made
from tests.trials import BLUEPRINTS, FakeTrials, Harbor, dataset

EVALUATE = BLUEPRINTS / "evaluate"
#: One outcome per task of the four-task dataset: two graded, one timeout, one crash.
OUTCOMES = {
    "a": {"reward": 1.0},
    "b": {"reward": 0.0},
    "c": {"reward": 1.0, "exception": "AgentTimeoutError"},
}


def _blueprint(tmp_path: Path, **replacements: str) -> Path:
    """The evaluate fixture over a four-task test set, two tasks per batch, one rollout each."""
    dataset(tmp_path / "tasks", "four", *"abcd")
    text = (EVALUATE / "run.toml").read_text(encoding="utf-8")
    text = text.replace('"fixture"', '"four"').replace("batch_size = 1", "batch_size = 2")
    text = text.replace("group_size = 2", "group_size = 1")
    for old, new in replacements.items():
        text = text.replace(old, new)
    home = tmp_path / "bp"
    home.mkdir()
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


async def _run(home: Path, tmp_path: Path) -> Run:
    opened = Run.open(home, root=tmp_path / "runs")
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
    return opened


async def test_the_test_set_is_measured_in_one_row_over_one_job_per_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    monkeypatch.chdir(tmp_path)
    harbor(FakeTrials(breaks=("d",), outcomes=OUTCOMES))
    opened = await _run(_blueprint(tmp_path), tmp_path)
    (row,) = list(record.read(opened.directory / record.METRICS))
    assert row["evaluation"] is True and row["step"] == 0
    assert (row["tasks"], row["rollouts"], row["graded"], row["masked"]) == (4, 4, 2, 2)
    assert row["reward_mean"] == 0.5
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["job"] for job in jobs] == [f"{opened.id}-0000", f"{opened.id}-0001"]
    assert [(job["purpose"], job["tasks"], job["trials"]) for job in jobs] == [
        ("evaluation", 2, 2),
        ("evaluation", 2, 2),
    ]
    assert sum(job["graded"] for job in jobs) == 2
    masked = [job.get("masked", {}) for job in jobs]
    assert sum(m.get("timeout", 0) for m in masked) == 1
    assert sum(m.get("env_error", 0) for m in masked) == 1
    assert record.read_json(opened.directory / record.COSTS)["sandboxes"] == {
        "modal": {"trials": 3, "seconds": 0.0}
    }
    assert not (opened.directory / record.CHECKPOINTS).exists()
    note = Run.read(opened.directory)
    assert note["failed"] is False and note["finished_at"]


async def test_the_test_group_size_sets_the_attempts_per_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _blueprint(tmp_path, **{"group_size = 1": "group_size = 1\ntest_group_size = 2"})
    harbor(FakeTrials(outcomes=OUTCOMES))
    opened = await _run(home, tmp_path)
    (row,) = list(record.read(opened.directory / record.METRICS))
    assert (row["tasks"], row["rollouts"], row["graded"], row["masked"]) == (4, 8, 6, 2)
    assert [job["trials"] for job in record.read(opened.directory / record.JOBS)] == [4, 4]


async def test_nothing_graded_has_no_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    harbor(FakeTrials(breaks=("a", "b", "c", "d")))
    opened = await _run(_blueprint(tmp_path), tmp_path)
    (row,) = list(record.read(opened.directory / record.METRICS))
    assert row["reward_mean"] is None and row["graded"] == 0 and row["masked"] == 4


async def test_a_served_model_runs_through_its_proxy_and_is_judged_by_its_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    """`a` graded 1 with two turns, `b` cut by its budget (0, overriding the verifier), `c`
    timed out, `d` never came up (no records, no result)."""
    monkeypatch.chdir(tmp_path)
    fake = FakeProxy(
        answers={
            "a": [made("one", "ok", seq=1), made("oneoktwo", "fine", seq=2, cached=3)],
            "b": [made(seq=1), made(seq=2, error=BUDGET)],
            "c": [made(seq=1)],
            "d": [],
        },
    )
    monkeypatch.setattr("shipyard.sampling.endpoint.Proxy", fake)
    replacements = {
        'provider = "openrouter"': 'provider = "tinker"',
        'sandbox = "modal"': 'sandbox = "docker"\nhost = "proxy.lan"',
    }
    home = _blueprint(tmp_path, **replacements)
    opened = Run.open(home, root=tmp_path / "runs")
    assert opened.config.agent.served
    trials = harbor(FakeTrials(breaks=("d",), outcomes={**OUTCOMES, "b": {"reward": 1.0}}))
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
    assert fake.placed[0][0] == "local" and fake.placed[0][2]["host"] == "proxy.lan"
    assert len(fake.placed) == 1, "started once, by the first job"
    assert fake.closed == 1, "stopped when the policy closed"
    assert sorted(fake.addressed) == sorted(fake.fetched) and len(fake.fetched) == 4
    for config in trials.configs:
        assert config.agent.model_name == "openai/some-hosted-model"
        assert config.agent.env["OPENAI_BASE_URL"].endswith(f"/r/trial/{config.trial_name}/v1")
        assert config.agent.env["OPENAI_API_KEY"] == fake.token
        assert config.agent.kwargs == {"version": "0.85.1", "model_api": "openai-completions"}
    (row,) = list(record.read(opened.directory / record.METRICS))
    assert (row["rollouts"], row["graded"], row["masked"]) == (4, 2, 2)
    assert row["reward_mean"] == 0.5, "a's 1 and b's budget-cut 0; c and d masked"
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["served"] for job in jobs] == ["tinker", "tinker"]
    assert [job["party"] for job in jobs] == ["tinker", "tinker"]
    masked = {key: n for job in jobs for key, n in job.get("masked", {}).items()}
    assert masked == {"timeout": 1, "env_error": 1}
    assert sum(job.get("turned_away", 0) for job in jobs) == 1
    assert sum(job["input_tokens"] for job in jobs) == 3 + 8 + 2 + 2 + 2, "the whole prompt"
    assert sum(job["cache_tokens"] for job in jobs) == 3
    assert sum(job["output_tokens"] for job in jobs) == 2 + 4 + 2 + 2
    rows = list(record.read(opened.directory / record.REQUESTS))
    assert len(rows) == 5 and {row["job"] for row in rows} == {job["job"] for job in jobs}
    assert sum(row["error"] == BUDGET for row in rows) == 1
    costs = record.read_json(opened.directory / record.COSTS)
    assert costs["parties"] == {
        "tinker": {"trials": 3, "input_tokens": 17, "cache_tokens": 3, "output_tokens": 10}
    }
    note = Run.read(opened.directory)
    assert note["tinker_base_url"].startswith("https://") and note["failed"] is False
