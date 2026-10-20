"""Tests for `shipyard.run.logs`: the rows a run writes to its directory."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from shipyard.config import load
from shipyard.groups import Trial
from shipyard.jobs.engine import Job
from shipyard.jobs.harnesses import harness
from shipyard.run import record
from shipyard.run.costs import Costs
from shipyard.run.logs import Logs
from shipyard.trainer import Evaluation
from tests.fakes import made

BLUEPRINT = """[agent]
model = "m"
harness = "pi"

[data]
train = "four"
batch_size = 1

[recipe]
steps = 1
"""


def _logs(tmp_path: Path) -> Logs:
    (tmp_path / "run.toml").write_text(BLUEPRINT, encoding="utf-8")
    return Logs(load(tmp_path), tmp_path / "runs" / "r")


def test_a_job_writes_its_row_and_a_row_per_model_call_the_proxy_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    logs = _logs(tmp_path)
    alpha, beta = (
        Trial("j", Path(task), harness("pi"), tmp_path / "jobs" / "j" / f"{task}__1")
        for task in ("alpha", "beta")
    )
    alpha.records = [
        made("one", "ok", seq=1, request_id="r-1"),
        made("oneoktwo", "fine", seq=2, cached=5, bridged=True),
    ]
    beta.records = [made("go", error="context", seq=1)]
    logs.job(Job("j", trials=[alpha, beta]), SimpleNamespace(provider="tinker", model="m"))
    [job] = list(record.read(logs.directory / record.JOBS))
    assert (job["job"], job["trials"], job["backend"]) == ("j", 2, "tinker")
    rows = list(record.read(logs.directory / record.REQUESTS))
    assert [row["trial"] for row in rows] == [alpha.name, alpha.name, beta.name]
    first = rows[0]
    assert set(first) == {
        "at",
        "job",
        "trial",
        "call",
        "sample",
        "samples",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "stop_reason",
        "sample_ms",
        "weights",
        "request_id",
        "bridged",
        "error",
    }
    assert (first["job"], first["call"], first["prompt_tokens"], first["completion_tokens"]) == (
        "j",
        1,
        3,
        2,
    )
    assert (first["request_id"], first["weights"], first["error"], first["bridged"]) == (
        "r-1",
        None,
        None,
        False,
    )
    assert (rows[1]["cached_tokens"], rows[1]["bridged"], rows[1]["sample_ms"]) == (5, True, 12.0)
    assert (rows[2]["error"], rows[2]["completion_tokens"], rows[2]["stop_reason"]) == (
        "context",
        0,
        "",
    )
    # What the job spent is the agent's, filed under the job, and the costs are its totals.
    spent = list(record.read(logs.directory / record.USAGE))
    assert {(row["role"], row["job"]) for row in spent} == {("agent", "j")}
    assert [row["model"] for row in spent if "provider" in row] == ["m"]
    totals = record.read_json(logs.directory / record.COSTS)
    assert Costs.of(spent).to_dict() == totals and totals["tokens"]["tinker"]["m"]["sample"] == 6


def test_every_spend_is_a_row_of_the_usage_ledger_and_on_disk_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run.toml").write_text(BLUEPRINT, encoding="utf-8")
    logs = Logs(load(tmp_path), tmp_path / "runs" / "r", step=lambda: 3)
    logs.spend("embedder", "openai", "openai/text-embedding-3-small", prefill=120)
    logs.spend("embedder", "openai", "text-embedding-3-small", prefill=0)  # nothing spent
    with pytest.raises(ValueError, match="tokens are counted as prefill, cache, sample, train"):
        logs.spend("embedder", "openai", "text-embedding-3-small", embedded=5)
    [row] = list(record.read(logs.directory / record.USAGE))
    assert {key: row[key] for key in ("step", "role", "provider", "model", "prefill")} == {
        "step": 3,
        "role": "embedder",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "prefill": 120,
    }
    totals = record.read_json(logs.directory / record.COSTS)
    assert totals["tokens"] == {"openai": {"text-embedding-3-small": {"prefill": 120}}}


def test_a_steps_row_holds_its_evaluation_and_the_recipes_numbers_beside_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    logs = _logs(tmp_path)
    each = {"reward_mean_low": 0.25}
    evaluation = Evaluation(
        1, tasks=2, trials=4, graded=3, masked=1, reward_mean=0.5, breakdown=each
    )
    logs.step(1, [], None, evaluation, {"trials": 9, "own": 1})
    logs.step(2, [], None, None, {})
    first, second = record.read(logs.directory / record.METRICS)
    assert (first["step"], first["trained"], first["trials"]) == (1, False, 0)
    evaluated = {key: value for key, value in first.items() if key.startswith("eval_")}
    assert evaluated == {
        "eval_tasks": 2,
        "eval_trials": 4,
        "eval_graded": 3,
        "eval_masked": 1,
        "eval_reward_mean": 0.5,
        "eval_reward_mean_low": 0.25,
    }
    assert (first["recipe_trials"], first["own"]) == (9, 1), "named like the row's own: beside it"
    assert second["step"] == 2 and not any(key.startswith("eval_") for key in second)
    assert not {"section", "seq", "attempt"} & set(first), "one kind of row, written by one process"
