"""Counts per party and per sandbox, read off the trials, and never summed into a total."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shipyard.cost import Costs, billed_by, reported, sandbox_seconds
from shipyard.rollout import Rollouts
from tests.trials import write_result


def _rollouts(*trials: Path) -> Rollouts:
    return Rollouts(job="j", trials=list(trials), plan=[])


def test_reported_counts_trials_with_a_result_and_sums_what_their_agents_said(
    tmp_path: Path,
) -> None:
    first = write_result(tmp_path / "first", reward=1.0, tokens=(100, 40, 10))
    second = write_result(tmp_path / "second", reward=0.0, tokens=(50, None, 5))
    silent = write_result(tmp_path / "silent", reward=1.0)
    gone = tmp_path / "gone"
    broken = write_result(tmp_path / "broken", raw="{not json")
    found = reported(_rollouts(first, second, silent, gone, broken))
    assert found == {"trials": 3, "input_tokens": 150, "cache_tokens": 40, "output_tokens": 15}
    # A trial that reported nothing is still a trial the party served; it adds no tokens.
    assert reported(_rollouts(silent, gone)) == {
        "trials": 1,
        "input_tokens": 0,
        "cache_tokens": 0,
        "output_tokens": 0,
    }


def test_reported_reads_a_multi_step_trials_steps(tmp_path: Path) -> None:
    stepped = write_result(tmp_path / "stepped", reward=1.0, steps=[(10, 0, 1), (20, 5, 2)])
    assert reported(_rollouts(stepped)) == {
        "trials": 1,
        "input_tokens": 30,
        "cache_tokens": 5,
        "output_tokens": 3,
    }


def test_billed_by_needs_every_trial_that_recorded_a_provider_to_agree(tmp_path: Path) -> None:
    one = write_result(tmp_path / "one", reward=1.0, provider="Anthropic")
    two = write_result(tmp_path / "two", reward=1.0, provider="anthropic ")
    none = write_result(tmp_path / "none", reward=1.0)
    other = write_result(tmp_path / "other", reward=1.0, provider="openai")
    assert billed_by(_rollouts(one, two, none, tmp_path / "gone")) == "anthropic"
    assert billed_by(_rollouts(one, other)) is None
    assert billed_by(_rollouts(none, tmp_path / "gone")) is None


def test_sandbox_seconds_span_the_timed_phases_of_trials_with_a_result(tmp_path: Path) -> None:
    long = write_result(tmp_path / "long", reward=1.0, seconds=90.0)
    short = write_result(tmp_path / "short", reward=0.0, seconds=30.5)
    untimed = write_result(tmp_path / "untimed", reward=1.0)
    gone = tmp_path / "gone"
    assert sandbox_seconds(_rollouts(long, short, untimed, gone)) == (3, 120.5)
    assert sandbox_seconds(_rollouts(gone)) == (0, 0.0)


def test_costs_accumulate_per_party_and_per_sandbox() -> None:
    costs = Costs()
    costs.add_party("anthropic", trials=4, input_tokens=100, cache_tokens=10, output_tokens=20)
    costs.add_party("anthropic", trials=2, input_tokens=50, output_tokens=5)
    costs.add_party("openai", trials=1, input_tokens=1, cache_tokens=0, output_tokens=1)
    costs.add_sandbox("docker", trials=6, seconds=12.5)
    costs.add_sandbox("docker", trials=1, seconds=0.5)
    assert costs.to_dict() == {
        "parties": {
            "anthropic": {
                "trials": 6,
                "input_tokens": 150,
                "cache_tokens": 10,
                "output_tokens": 25,
            },
            "openai": {"trials": 1, "input_tokens": 1, "cache_tokens": 0, "output_tokens": 1},
        },
        "sandbox": {"docker": {"trials": 7, "seconds": 13.0}},
    }
    assert Costs.from_dict(costs.to_dict()).to_dict() == costs.to_dict()
    assert Costs.from_dict({}).to_dict() == {"parties": {}, "sandbox": {}}


def test_training_counts_join_a_party_only_once_spent_and_survive_a_round_trip() -> None:
    costs = Costs()
    costs.add_party("tinker", trials=2, input_tokens=10, cache_tokens=0, output_tokens=4)
    assert "train_tokens" not in costs.to_dict()["parties"]["tinker"]
    costs.add_party("tinker", train_tokens=12, reference_tokens=12)
    costs.add_party("tinker", train_tokens=12, anchor_tokens=16)
    assert costs.to_dict()["parties"]["tinker"] == {
        "trials": 2,
        "input_tokens": 10,
        "cache_tokens": 0,
        "output_tokens": 4,
        "train_tokens": 24,
        "reference_tokens": 12,
        "anchor_tokens": 16,
    }
    assert Costs.from_dict(costs.to_dict()).to_dict() == costs.to_dict()
    costs.add_party("fresh", train_tokens=3)
    assert costs.to_dict()["parties"]["fresh"] == {
        "trials": 0,
        "input_tokens": 0,
        "cache_tokens": 0,
        "output_tokens": 0,
        "train_tokens": 3,
    }


def test_costs_hold_no_total_and_no_price() -> None:
    costs = Costs()
    costs.add_party("anthropic", trials=1, input_tokens=1, cache_tokens=1, output_tokens=1)
    costs.add_sandbox("modal", trials=1, seconds=1.0)
    text = json.dumps(costs.to_dict()).lower()
    assert "total" not in text and "usd" not in text and "price" not in text
    with pytest.raises(ValueError, match="usd"):
        costs.add_party("anthropic", usd=1)
