"""The evaluate recipe end to end with a stand-in for Harbor: the rows per batch and the
final row carry the right counts, the mean leaves the masked out, and a served model
runs through a fake proxy whose records admission and the bill are read from."""

from __future__ import annotations

import shutil
from pathlib import Path
from statistics import fmean

import pytest

from shipyard import record, serving
from shipyard.admit import BUDGET
from shipyard.data import batches
from shipyard.recipes import run_recipe
from shipyard.recipes.evaluate import mean
from shipyard.run import Run
from tests.records import FakeProxy, made
from tests.trials import FIXTURES, FakeTrials

EVALUATE = Path(__file__).parent / "blueprints" / "evaluate"
#: One outcome per task of the four-task dataset: two graded, one timeout, one crash.
OUTCOMES = {
    "a": {"reward": 1.0},
    "b": {"reward": 0.0},
    "c": {"reward": 1.0, "exception": "AgentTimeoutError"},
}
GRADED = {"a": 1.0, "b": 0.0}


def _blueprint(tmp_path: Path, **replacements: str) -> Path:
    """The evaluate fixture over a four-task dataset, two tasks per batch, one rollout each."""
    for name in ("a", "b", "c", "d"):
        shutil.copytree(FIXTURES / "fixture" / "alpha", tmp_path / "tasks" / "four" / name)
    text = (EVALUATE / "run.toml").read_text(encoding="utf-8")
    text = text.replace('"fixture"', '"four"').replace("batch_size = 1", "batch_size = 2")
    text = text.replace("group_size = 2", "group_size = 1")
    for old, new in replacements.items():
        text = text.replace(old, new)
    home = tmp_path / "bp"
    home.mkdir()
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


async def test_the_rows_count_each_batch_and_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _blueprint(tmp_path)
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = FakeTrials(breaks=("d",), outcomes=OUTCOMES)
    with opened:
        await run_recipe(opened)
    planned = list(batches(["four"], size=2, seed=0, epochs=1))
    assert [len(batch) for batch in planned] == [2, 2]
    *per_batch, final = list(record.read(opened.directory / record.METRICS))
    assert len(per_batch) == 2
    for index, (row, batch) in enumerate(zip(per_batch, planned, strict=True)):
        graded = [GRADED[task.name] for task in batch if task.name in GRADED]
        assert row["seq"] == index + 1 and row["batch"] == index
        assert row["tasks"] == 2 and row["rollouts"] == 2
        assert row["graded"] == len(graded) and row["masked"] == 2 - len(graded)
        assert row["mean"] == (fmean(graded) if graded else None)
    assert final["seq"] == 3 and final["evaluation"] is True
    assert final["batches"] == 2 and final["rollouts"] == 4
    assert final["graded"] == 2 and final["masked"] == 2 and final["mean"] == 0.5
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["job"] for job in jobs] == [f"{opened.id}-0000", f"{opened.id}-0001"]
    assert sum(job["graded"] for job in jobs) == 2
    masked = [job.get("masked", {}) for job in jobs]
    assert sum(m.get("timeout", 0) for m in masked) == 1
    assert sum(m.get("env_error", 0) for m in masked) == 1
    assert record.read_json(opened.directory / record.COSTS)["sandbox"] == {
        "modal": {"trials": 3, "seconds": 0.0}
    }
    note = Run.read(opened.directory)
    assert note["failed"] is False and note["finished_at"]


async def test_a_batch_with_nothing_graded_has_no_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    opened = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")
    opened.run_trial = FakeTrials(breaks=("a", "b", "c", "d"))
    await run_recipe(opened)
    rows = list(record.read(opened.directory / record.METRICS))
    assert [row["mean"] for row in rows] == [None, None, None]
    assert rows[-1]["graded"] == 0 and rows[-1]["masked"] == 4
    assert mean([]) is None and mean([1.0, 0.0]) == 0.5


async def test_a_served_model_runs_through_its_proxy_and_is_judged_by_its_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four tasks, two per batch: `a` graded 1 with two turns, `b` cut by its budget (0,
    overriding the verifier), `c` timed out, `d` never came up (no records, no result)."""
    monkeypatch.chdir(tmp_path)
    fake = FakeProxy(
        answers={
            "a": [made("one", "ok", seq=1), made("oneoktwo", "fine", seq=2, cached=3)],
            "b": [made(seq=1), made(seq=2, error=BUDGET)],
            "c": [made(seq=1)],
            "d": [],
        },
        counts={"b": {"turned_away": 1, "cut": 0, "spoke": 2}},
    )
    monkeypatch.setattr(serving, "Proxy", fake)
    replacements = {
        'provider = "openrouter"': 'provider = "tinker"',
        'sandbox = "modal"': 'sandbox = "docker"\nhost = "proxy.lan"',
    }
    home = _blueprint(tmp_path, **replacements)
    opened = Run.open(home, root=tmp_path / "runs")
    assert opened.config.model.served and opened.serving is not None
    opened.run_trial = FakeTrials(breaks=("d",), outcomes={**OUTCOMES, "b": {"reward": 1.0}})
    with opened:
        await run_recipe(opened)
    assert fake.placed[0][0] == "local" and fake.placed[0][2]["host"] == "proxy.lan"
    assert len(fake.placed) == 1, "started once, by the first job"
    assert fake.closed == 1, "stopped when the run closed"
    assert sorted(fake.addressed) == sorted(fake.fetched) and len(fake.fetched) == 4
    for config in opened.run_trial.configs:
        assert config.agent.model_name == "openai/some-hosted-model"
        assert config.agent.env["OPENAI_BASE_URL"].endswith(f"/r/trial/{config.trial_name}/v1")
        assert config.agent.env["OPENAI_API_KEY"] == fake.token
        assert config.agent.kwargs == {"version": "0.85.1", "model_api": "openai-completions"}
    *per_batch, final = list(record.read(opened.directory / record.METRICS))
    assert final["rollouts"] == 4 and final["graded"] == 2 and final["masked"] == 2
    assert final["mean"] == 0.5, "a's 1 and b's budget-cut 0; c and d masked"
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["served"] for job in jobs] == ["tinker", "tinker"]
    assert [job["party"] for job in jobs] == ["tinker", "tinker"]
    masked = {key: n for job in jobs for key, n in job.get("masked", {}).items()}
    assert masked == {"timeout": 1, "env_error": 1}
    assert sum(job.get("turned_away", 0) for job in jobs) == 1
    assert "cut" not in jobs[0] and "cut" not in jobs[1]
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
