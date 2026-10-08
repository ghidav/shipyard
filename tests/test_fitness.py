"""What the search selects on: a frontier of column winners with ties kept, means that
leave the unmeasured out, and a tally over the pool alone."""

from __future__ import annotations

import pytest

from shipyard.gepa.fitness import Fitness, Outcome, mean, tally

TASKS = ("t1", "t2", "t3", "t4")


def _seen(**rows: list[Outcome]) -> dict[str, dict[str, Outcome]]:
    return {digest: {outcome.task: outcome for outcome in row} for digest, row in rows.items()}


def test_the_frontier_keeps_a_candidate_that_wins_one_task() -> None:
    """`wide` is better on average and on three of four tasks. `narrow` alone solves t4,
    so a ranking by mean would lose it."""
    seen = _seen(
        wide=[Outcome(task, 0.9) for task in TASKS[:3]] + [Outcome("t4", 0.0)],
        narrow=[Outcome(task, 0.1) for task in TASKS[:3]] + [Outcome("t4", 0.5)],
    )
    scores = tally(["wide", "narrow"], seen)
    assert scores.frontier() == {"wide", "narrow"}
    assert scores.aggregate("wide") > scores.aggregate("narrow")
    assert scores.scores["narrow"]["t4"] > scores.scores["wide"]["t4"]
    assert len(scores.scores) == 2 and scores.tasks == list(TASKS)


def test_a_tie_on_a_column_keeps_both_and_there_is_no_tolerance() -> None:
    seen = _seen(
        a=[Outcome("t1", 0.6), Outcome("t2", 1.0)], b=[Outcome("t1", 0.6), Outcome("t3", 1.0)]
    )
    scores = tally(["a", "b"], seen)
    assert scores.frontier() == {"a", "b"}, "tied on t1, and each leads a task of its own"
    assert scores.leads() == {"a": 2, "b": 2}
    close = tally(["a", "b"], _seen(a=[Outcome("t1", 0.62)], b=[Outcome("t1", 0.60)]))
    assert close.frontier() == {"a"}, "lost by a hair is lost"
    with pytest.raises(TypeError):
        close.frontier(0.05)  # type: ignore[call-arg]


def test_an_unmeasured_outcome_is_absent_rather_than_zero() -> None:
    seen = _seen(a=[Outcome("t1", 1.0, count=4), Outcome("t2", None)])
    scores = tally(["a"], seen)
    assert scores.scores["a"] == {"t1": 1.0} and scores.coverage("a") == 1
    assert scores.aggregate("a") == 1.0
    assert mean([Outcome("t1", None)]) is None and mean([]) is None
    assert mean([Outcome("t1", 1.0), Outcome("t2", None), Outcome("t3", 0.0)]) == 0.5
    assert Outcome("t1", 0.0).measured and not Outcome("t1", None).measured


def test_an_aggregate_over_nothing_is_none_and_over_named_tasks_is_trimmed() -> None:
    assert Fitness().aggregate("missing") is None and Fitness().frontier() == set()
    scores = tally(["a"], _seen(a=[Outcome("t1", 0.4), Outcome("t2", 0.8)]))
    assert scores.aggregate("a") == pytest.approx(0.6)
    assert scores.aggregate("a", tasks=["t1"]) == pytest.approx(0.4)
    assert scores.aggregate("a", tasks=["t9"]) is None


def test_the_tally_covers_the_pool_alone() -> None:
    """A child turned down on its minibatch is measured but not pooled: its partial row
    must not win columns it was never compared on."""
    seen = _seen(seed=[Outcome("t1", 0.5), Outcome("t2", 0.5)], child=[Outcome("t1", 1.0)])
    scores = tally(["seed"], seen)
    assert list(scores.scores) == ["seed"] and scores.frontier() == {"seed"}
    unmeasured = tally(["seed", "fresh"], seen)
    assert unmeasured.scores["fresh"] == {} and unmeasured.aggregate("fresh") is None
    assert unmeasured.coverage("fresh") == 0 and unmeasured.coverage("seed") == 2
    assert unmeasured.frontier() == {"seed"}


def test_an_outcome_carries_what_the_reflector_is_shown() -> None:
    outcome = Outcome("t1", 0.0, feedback="Expected: 385", inputs="Aya walks", transcript="...")
    assert (outcome.feedback, outcome.inputs, outcome.transcript) == (
        "Expected: 385",
        "Aya walks",
        "...",
    )
    assert outcome.count == 1
    assert not hasattr(outcome, "forks")


def test_a_dominated_candidate_leaves_the_frontier() -> None:
    """GEPA's selection: b leads only t1, which a leads too, so b is dominated."""
    seen = _seen(a=[Outcome("t1", 0.6), Outcome("t2", 1.0)], b=[Outcome("t1", 0.6)])
    scores = tally(["a", "b"], seen)
    assert scores.fronts() == {"t1": {"a", "b"}, "t2": {"a"}}
    assert scores.frontier() == {"a"} and scores.leads() == {"a": 2}
