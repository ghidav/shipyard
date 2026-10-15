"""Each case replayed against its frozen file; `GOLDEN_UPDATE=1` rewrites them."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests import golden
from tests.golden import CASES, CREDITED, JOBBED, TRAINERS, differences, replay, trained

#: The cases the trainer replays: every credit case of a recipe class, without a batch
#: limit, which the accumulating `sample` has no use for, and without a preset field the
#: class cannot express; and the two one-job cases.
THROUGH_THE_TRAINER = tuple(
    name
    for name, case in CREDITED.items()
    if case.limit is None and not case.preset_only and case.kind in TRAINERS
) + ("job-served", "job-provider")


@pytest.mark.parametrize("name", CASES)
async def test_the_case_reproduces_its_frozen_output(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = await replay(name, tmp_path, monkeypatch)
    if golden.updating():
        golden.write_frozen(name, found)
        return
    frozen = golden.read_frozen(name)
    assert frozen is not None, (
        f"no frozen output for {name!r}: run `{golden.UPDATE}=1 pytest tests/test_golden.py`"
    )
    found_differences = differences(found, frozen)
    assert not found_differences, "\n".join(found_differences)


def test_every_case_has_a_file_and_every_file_a_case() -> None:
    if golden.updating():
        pytest.skip("regenerating")
    frozen = {path.stem for path in golden.GOLDEN.glob("*.json")}
    assert frozen == set(CASES), f"missing {set(CASES) - frozen}, orphaned {frozen - set(CASES)}"
    assert not set(CREDITED) & set(JOBBED), "a case name is one case"


async def test_a_credit_case_freezes_the_datums_the_calls_and_the_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = await replay("dapo-mixed", tmp_path, monkeypatch)
    assert set(found) == {"batch", "client", "sent", "anchor", "update", "row", "updates"}
    batch = found["batch"]
    assert batch["surplus"] == 1 and batch["degenerate"] == 1 and batch["credited"] == 7
    datum = batch["datums"][0]
    assert set(datum) == {"input", "target_tokens", "logprobs", "advantages", "mask"}
    sent = [call[0] for call in found["client"] if call[0] != "consumed"]
    assert sent == [
        "forward",
        "forward_backward",
        "optim_step",
        "forward_backward",
        "optim_step",
    ], "one reference pass, then dapo's substeps capped at the two groups"
    assert "consumed" in [call[0] for call in found["client"]], "the pipelining order is frozen"
    assert found["row"]["trained"] is True and "seconds" not in found["row"]
    assert found["updates"] == 1


async def test_a_job_case_freezes_the_verdicts_the_groups_and_the_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = await replay("job-served", tmp_path, monkeypatch)
    assert [one["mask"] for one in found["verdicts"]] == [
        None,
        None,
        "context_overflow",
        "api_error",
        "env_error",
        None,
        None,
        "timeout",
    ]
    assert [one["reward"] for one in found["verdicts"][:2]] == [1.0, 0.0], "the budget's cut"
    (job,) = found["jobs"]
    assert job["graded"] == 4 and job["served"] == "tinker" and job["trials"] == 8
    assert job["masked"] == {"context_overflow": 1, "api_error": 1, "env_error": 1, "timeout": 1}
    assert [one["trial"] for one in found["requests"]] == [0, 0, 1, 1, 2, 3, 3, 5, 6, 7]
    assert [one["task"] for one in found["groups"]] == ["a", "b"]
    assert found["batch"]["credited"] == 4 and found["batch"]["overlong"] == 0


async def test_an_fst_step_merges_the_texts_jobs_before_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = await replay("job-fst-share-2", tmp_path, monkeypatch)
    first, second = found["jobs"]
    assert first["modules"] != second["modules"] and first["trials"] == second["trials"] == 4
    alpha, beta = found["groups"]
    assert [one["reward"] for one in alpha["members"]] == [1.0, 1.0, 0.0, 0.0]
    assert [one["reward"] for one in beta["members"]] == [1.0, 0.0, 0.0, 1.0]
    assert found["batch"]["credited"] == 8, "task a's spread comes from the two texts"
    assert found["batch"]["anchor_kl"] is not None, "fst's kl term"


def test_differences_report_a_moved_number_and_pass_over_rounding() -> None:
    frozen = {"batch": {"datums": [{"advantages": [0.0, 1.5, -1.5]}], "owners": [0, 0]}, "x": "a"}
    same = copy.deepcopy(frozen)
    same["batch"]["datums"][0]["advantages"][1] += 1e-9
    assert differences(same, frozen) == []
    moved = copy.deepcopy(frozen)
    moved["batch"]["datums"][0]["advantages"][1] += 1e-3
    moved["x"] = "b"
    del moved["batch"]["owners"]
    moved["batch"]["extra"] = 1
    assert differences(moved, frozen) == [
        ".batch.datums[0].advantages[1]: 1.501, frozen 1.5",
        ".batch.extra: new (found 1)",
        ".batch.owners: missing (frozen [0, 0])",
        ".x: 'b', frozen 'a'",
    ]
    assert differences([1, 2], [1, 2, 3]) == [": 2 item(s), frozen 3"]


@pytest.mark.parametrize("name", THROUGH_THE_TRAINER)
async def test_the_trainer_reproduces_the_frozen_case(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same datums, calls, anchor queries, update and row as the frozen case, from the
    trainer's `job`, `shape`, `advantage`, `credit`, `step` and `log`."""
    if golden.updating():
        pytest.skip("regenerating")
    frozen = golden.read_frozen(name)
    assert frozen is not None
    found = await trained(name, tmp_path, monkeypatch)
    expected = {key: frozen[key] for key in found if key != "groups"}
    if "groups" in found:
        expected["groups"] = [
            {"task": one["task"], "members": one["members"]} for one in frozen["groups"]
        ]
    found_differences = differences(found, expected)
    assert not found_differences, "\n".join(found_differences)
