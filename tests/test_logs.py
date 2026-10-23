"""Tests for `shipyard.run.logs`: the rows a run writes to its directory."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from shipyard.config import load
from shipyard.groups import Trial
from shipyard.jobs.engine import Job
from shipyard.jobs.harnesses import Harness
from shipyard.records import Records
from shipyard.run import files
from shipyard.run.costs import Costs
from shipyard.run.logs import Logs
from shipyard.trainer import Evaluation, Update
from tests.fakes import Alternating, FakeProxy, Harbor, made
from tests.runs import GRPO, ONE_STEP, rows, run_toml, train
from tests.service import FakeService

RUN_TOML = """[models.agent]
model = "m"

[data]
train = "four"
batch_size = 1

[recipe]
steps = 1
"""


def _logs(tmp_path: Path) -> Logs:
    (tmp_path / "run.toml").write_text(RUN_TOML, encoding="utf-8")
    return Logs(load(tmp_path), tmp_path / "runs" / "r")


def test_a_job_writes_its_row_and_a_row_per_model_call_the_proxy_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    logs = _logs(tmp_path)
    alpha, beta = (
        Trial("j", Path(task), Harness(name="pi"), tmp_path / "jobs" / "j" / f"{task}__1")
        for task in ("alpha", "beta")
    )
    alpha.records = Records(
        [
            made("one", "ok", seq=1, request_id="r-1"),
            made("oneoktwo", "fine", seq=2, cached=5, bridged=True),
        ]
    )
    beta.records = Records([made("go", error="context", seq=1)])
    at_host = SimpleNamespace(provider="tinker@api.example.com", model="m", kind="tinker")
    logs.job(Job("j", role="agent", trials=[alpha, beta]), at_host)
    [job] = list(files.read(logs.directory / files.JOBS))
    assert (job["job"], job["trials"], job["backend"]) == ("j", 2, "tinker"), "named by its kind"
    rows = list(files.read(logs.directory / files.REQUESTS))
    assert [row["trial"] for row in rows] == [alpha.name, alpha.name, beta.name]
    first = rows[0]
    assert set(first) == {
        "at",
        "job",
        "trial",
        "call",
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
    spent = list(files.read(logs.directory / files.USAGE))
    assert {(row["role"], row["job"]) for row in spent} == {("agent", "j")}
    assert [row["model"] for row in spent if "provider" in row] == ["m"]
    totals = files.read_json(logs.directory / files.COSTS)
    assert Costs.of(spent).to_dict() == totals
    assert totals["tokens"]["tinker@api.example.com"]["m"]["sample"] == 6, "billed to the host"


def test_every_spend_is_a_row_of_the_usage_ledger_and_on_disk_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run.toml").write_text(RUN_TOML, encoding="utf-8")
    logs = Logs(load(tmp_path), tmp_path / "runs" / "r", step=lambda: 3)
    logs.spend("embedder", "openai", "openai/text-embedding-3-small", prefill=120)
    logs.spend("embedder", "openai", "text-embedding-3-small", prefill=0)  # nothing spent
    with pytest.raises(ValueError, match="tokens are counted as prefill, cache, sample, train"):
        logs.spend("embedder", "openai", "text-embedding-3-small", embedded=5)
    [row] = list(files.read(logs.directory / files.USAGE))
    assert {key: row[key] for key in ("step", "role", "provider", "model", "prefill")} == {
        "step": 3,
        "role": "embedder",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "prefill": 120,
    }
    totals = files.read_json(logs.directory / files.COSTS)
    assert totals["tokens"] == {"openai": {"text-embedding-3-small": {"prefill": 120}}}


def test_a_steps_row_holds_the_recipes_numbers_beside_its_own_and_an_evaluation_has_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    logs = _logs(tmp_path)
    logs.step(1, [], {}, {"trials": 9, "own": 1})
    logs.evaluation(Evaluation(1, tasks=2, trials=4, graded=3, masked=1, reward_mean=None))
    logs.step(2, [], {}, {})
    first, evaluated, second = files.read(logs.directory / files.METRICS)
    assert (first["step"], first["trained"], first["trials"]) == (1, False, 0)
    assert (first["recipe_trials"], first["own"]) == (9, 1), "named like the row's own: beside it"
    assert not any(key.startswith("eval_") for key in (*first, *second))
    assert {key: value for key, value in evaluated.items() if key != "at"} == {
        "step": 1,
        "eval_tasks": 2,
        "eval_trials": 4,
        "eval_graded": 3,
        "eval_masked": 1,
    }, "at the step whose weights it measured; a mean with nothing graded is left out"
    assert not {"section", "seq", "attempt"} & set(first), "one kind of row, written by one process"


def _update(role: str, sequences: int) -> Update:
    return Update(
        role=role,
        sequences=sequences,
        train_tokens=10 * sequences,
        substeps=1,
        learning_rate=2e-5,
        loss="importance_sampling",
        update_seconds=0.5,
        kl_v1=0.01,
        kl_v2=0.02,
        entropy=0.3,
    )


def _fields(update: Update) -> dict[str, object]:
    return {key: value for key, value in vars(update).items() if key != "role"}


def test_a_steps_row_holds_one_models_update_as_it_is_and_each_of_several_by_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    logs = _logs(tmp_path)
    agent, teacher = _update("agent", 4), _update("teacher", 2)
    logs.step(1, [], {"agent": agent}, {})
    logs.step(2, [], {"agent": agent, "teacher": teacher}, {})
    one, two = files.read(logs.directory / files.METRICS)
    assert one["trained"] is True and {key: one[key] for key in _fields(agent)} == _fields(agent)
    assert not any(key.startswith(("agent_", "teacher_")) for key in one), "unprefixed"
    assert two["trained"] is True and not set(_fields(agent)) & set(two), "none unprefixed"
    for update in (agent, teacher):
        prefixed = {key: two[f"{update.role}_{key}"] for key in _fields(update)}
        assert prefixed == _fields(update), "each field under its model's role"


class Twice(GRPO):  # type: ignore[misc, valid-type]
    """GRPO's agent rolled out at its first task twice in a step, as two jobs."""

    async def step(self) -> None:
        task = self.draw()[0]
        for _ in range(2):
            await self.rollout(
                [task], model=self.agent, harness=self.recipe.harness, attempts=2, modules=None
            )


async def test_a_steps_row_counts_the_groups_as_its_rollouts_returned_them(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """A task rolled out twice is two groups, one per rollout: the row merges none into
    another."""
    harbor(Alternating())
    directory = await train(run_toml(tmp_path, **ONE_STEP), tmp_path / "runs", Twice)
    (row,) = rows(directory)
    assert (row["groups"], row["trials"], len(row["jobs"])) == (2, 4, 2)
