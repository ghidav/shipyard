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
from shipyard.run.logs import Logs
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
    assert (job["job"], job["trials"], job["served"]) == ("j", 2, "tinker")
    rows = list(record.read(logs.directory / record.REQUESTS))
    assert [row["trial"] for row in rows] == [alpha.name, alpha.name, beta.name]
    first = rows[0]
    assert set(first) == {
        "at",
        "job",
        "trial",
        "seq",
        "sample",
        "samples",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "stop_reason",
        "sample_ms",
        "served",
        "request_id",
        "bridged",
        "error",
    }
    assert (first["job"], first["seq"], first["prompt_tokens"], first["completion_tokens"]) == (
        "j",
        1,
        3,
        2,
    )
    assert (first["request_id"], first["served"], first["error"], first["bridged"]) == (
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


def test_each_metrics_row_says_its_section_and_continues_the_numbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    logs = _logs(tmp_path)
    logs.log("train", step=0)
    logs.log("eval", step=1, reward_mean=0.5)
    again = Logs(load(tmp_path), logs.directory)
    again.log("train", step=1)
    rows = list(record.read(logs.directory / record.METRICS))
    assert [(row["seq"], row["section"], row["step"]) for row in rows] == [
        (1, "train", 0),
        (2, "eval", 1),
        (3, "train", 1),
    ]
    assert "evaluation" not in rows[1]
