"""Tests for `shipyard.run.costs`: token counts by provider, model and kind, and per-sandbox
seconds, which are never summed into a total or priced."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from shipyard.groups import Trial
from shipyard.jobs.harnesses import Harness
from shipyard.records import Records
from shipyard.run.costs import Costs, billed_by, billed_tokens, reported, sandbox_seconds
from tests.fakes import made, write_result


def _trials(*dirs: Path, records: Mapping[str, Sequence[Any]] | None = None) -> list[Trial]:
    """One trial per directory; with `records`, each holds those under its name, else none."""
    found = [Trial("j", Path(one.name), Harness(name="pi"), one) for one in dirs]
    if records is not None:
        for one in found:
            one.records = Records(records.get(one.name) or [])
    return found


def test_reported_bills_the_provider_and_takes_the_cached_part_out_of_the_prompt(
    tmp_path: Path,
) -> None:
    first = write_result(tmp_path / "first", reward=1.0, tokens=(100, 40, 10))
    second = write_result(tmp_path / "second", reward=0.0, tokens=(50, None, 5))
    silent = write_result(tmp_path / "silent", reward=1.0)
    gone = tmp_path / "gone"
    broken = write_result(tmp_path / "broken", raw="{not json")
    found = reported(_trials(first, second, silent, gone, broken), "anthropic", "claude-x")
    assert found == {"anthropic": {"claude-x": {"prefill": 110, "cache": 40, "sample": 15}}}
    assert reported(_trials(silent, gone), "anthropic", "claude-x") == {}, "nothing reported"


def test_reported_counts_each_model_a_trial_names_and_its_steps(tmp_path: Path) -> None:
    named = write_result(
        tmp_path / "named",
        reward=1.0,
        tokens=(130, 40, 12),
        usage={"claude-x": (100, 40, 10), "claude-haiku": (30, 0, 2)},
    )
    stepped = write_result(tmp_path / "stepped", reward=1.0, steps=[(10, 0, 1), (20, 5, 2)])
    found = reported(_trials(named, stepped), "anthropic", "anthropic/claude-x")
    assert found == {
        "anthropic": {
            "claude-x": {"prefill": 60 + 25, "cache": 40 + 5, "sample": 10 + 3},
            "claude-haiku": {"prefill": 30, "sample": 2},
        }
    }, "the totals beside a breakdown are not counted twice; the prefix is the provider's"


def test_sampled_adds_up_each_trials_records_as_the_endpoints_model(tmp_path: Path) -> None:
    graded = write_result(tmp_path / "graded", reward=1.0)
    gone = tmp_path / "gone"
    records = {
        graded.name: [made("one", "ok", seq=1, bridged=True)],
        gone.name: [made("go", error="context", seq=1), made("go", "fine", seq=2, cached=1)],
    }
    tokens, noted = billed_tokens(_trials(graded, gone, records=records), "tinker", "Qwen/Qwen3-8B")
    assert tokens == {"tinker": {"Qwen/Qwen3-8B": {"prefill": 3 + 1, "cache": 1, "sample": 6}}}
    assert noted == {"turned_away": 1, "bridged": 1}
    empty = _trials(gone, records={})
    assert billed_tokens(empty, "tinker", "m") == ({}, {"turned_away": 0, "bridged": 0})


def test_billed_by_needs_every_trial_that_recorded_a_provider_to_agree(tmp_path: Path) -> None:
    one = write_result(tmp_path / "one", reward=1.0, provider="Anthropic")
    two = write_result(tmp_path / "two", reward=1.0, provider="anthropic ")
    none = write_result(tmp_path / "none", reward=1.0)
    other = write_result(tmp_path / "other", reward=1.0, provider="openai")
    assert billed_by(_trials(one, two, none, tmp_path / "gone")) == "anthropic"
    assert billed_by(_trials(one, other)) is None
    assert billed_by(_trials(none, tmp_path / "gone")) is None


def test_sandbox_seconds_span_the_timed_phases_of_trials_with_a_result(tmp_path: Path) -> None:
    long = write_result(tmp_path / "long", reward=1.0, seconds=90.0)
    short = write_result(tmp_path / "short", reward=0.0, seconds=30.5)
    untimed = write_result(tmp_path / "untimed", reward=1.0)
    gone = tmp_path / "gone"
    assert sandbox_seconds(_trials(long, short, untimed, gone)) == (3, 120.5)
    assert sandbox_seconds(_trials(gone)) == (0, 0.0)


def test_costs_add_up_by_provider_model_and_kind_and_per_sandbox() -> None:
    costs = Costs()
    costs.add("anthropic", "claude-x", prefill=100, cache=10, sample=20)
    costs.add("anthropic", "anthropic/claude-x", prefill=50, sample=5)
    costs.add("tinker", "Qwen/Qwen3-8B", train=12, prefill=0)
    costs.add("tinker", "Qwen/Qwen3-32B")
    costs.merge({"tinker": {"Qwen/Qwen3-8B": {"train": 12, "sample": 1}}})
    costs.add_sandbox("docker", trials=6, seconds=12.5)
    costs.add_sandbox("docker", trials=1, seconds=0.5)
    assert costs.to_dict() == {
        "tokens": {
            "anthropic": {"claude-x": {"prefill": 150, "cache": 10, "sample": 25}},
            "tinker": {"Qwen/Qwen3-8B": {"sample": 1, "train": 24}},
        },
        "sandboxes": {"docker": {"trials": 7, "seconds": 13.0}},
    }
    assert Costs.from_dict(costs.to_dict()).to_dict() == costs.to_dict()
    assert Costs.from_dict({}).to_dict() == {"tokens": {}, "sandboxes": {}}


def test_costs_hold_no_total_and_no_price() -> None:
    costs = Costs()
    costs.add("anthropic", "claude-x", prefill=1, cache=1, sample=1)
    costs.add_sandbox("modal", trials=1, seconds=1.0)
    text = json.dumps(costs.to_dict()).lower()
    assert "total" not in text and "usd" not in text and "price" not in text
    with pytest.raises(ValueError, match="usd"):
        costs.add("anthropic", "claude-x", usd=1)
