"""The search on a fake scorer and a fake proposer: four tasks and one rule, a text scores
1.0 on a task whose name it mentions and `BASELINE` on one it does not, so every reward
in a case is visible in the case."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from shipyard.gepa.cycle import Evolution, best, evolve
from shipyard.gepa.fitness import Outcome, tally
from shipyard.gepa.propose import Reflection, component_for, propose
from shipyard.modules import SKILL_FILE, Candidate, Module

TASKS = tuple(Path("tasks/four") / name for name in ("t1", "t2", "t3", "t4"))
BASELINE = 0.1
ROW = (
    "round",
    "parent",
    "component",
    "child",
    "parent_mean",
    "child_mean",
    "accepted",
    "pool",
    "frontier",
    "spent",
)


def _module(text: str, name: str = "guide") -> Module:
    return Module(name, "skill", {SKILL_FILE: text})


def seed(text: str = "", *names: str) -> Candidate:
    names = names or ("guide",)
    return Candidate({name: _module(text, name) for name in names})


@dataclass
class World:
    """A scorer: 1.0 on a task the candidate's text names, `BASELINE` otherwise; what it
    was asked, and in which round."""

    asked: list[tuple[str, str]] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)

    async def __call__(
        self, candidate: Candidate, tasks: Sequence[Path], round_index: int
    ) -> list[Outcome]:
        text = " ".join(module.text for module in candidate.components.values())
        out = []
        for task in tasks:
            self.asked.append((candidate.digest, task.name))
            self.rounds.append(round_index)
            hit = task.name in text
            out.append(Outcome(str(task), 1.0 if hit else BASELINE, feedback="seen"))
        return out


@dataclass
class Scripted:
    """A proposer reading from a list: each round the next text, then it declines."""

    replies: list[str | None]
    seen: list[Reflection] = field(default_factory=list)

    async def __call__(self, reflection: Reflection) -> Module | None:
        self.seen.append(reflection)
        if len(self.seen) > len(self.replies):
            return None
        answered = self.replies[len(self.seen) - 1]
        return None if answered is None else _module(answered, reflection.component)


@dataclass
class Fresh:
    """A proposer that always answers with something new and useless."""

    seen: list[Reflection] = field(default_factory=list)

    async def __call__(self, reflection: Reflection) -> Module | None:
        self.seen.append(reflection)
        return _module(f"nothing to see here {len(self.seen)}", reflection.component)


@dataclass
class Logged:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **row: Any) -> None:
        self.rows.append(row)


async def run(*args: Any, **kwargs: Any) -> Evolution:
    kwargs.setdefault("rng", random.Random(0))
    kwargs.setdefault("minibatch", 2)
    kwargs.setdefault("budget", 100)
    kwargs.setdefault("patience", 3)
    return await evolve(*args, **kwargs)


# ------------------------------------------------------------------------ propose


async def test_a_proposal_that_says_nothing_is_not_a_candidate() -> None:
    parent = seed("keep going")
    assert await propose(parent, "guide", [], Scripted([None])) is None
    assert await propose(parent, "guide", [], Scripted(["   "])) is None
    assert await propose(parent, "guide", [], Scripted(["keep going"])) is None
    never = Scripted(["anything"])
    assert await propose(parent, "absent", [], never) is None and never.seen == []
    child = await propose(parent, "guide", [Outcome("t", 0.0)], Scripted(["go on"]))
    assert child is not None and child.components["guide"].text == "go on"
    assert child.components["guide"].name == "guide" and child.digest != parent.digest


def test_a_reflection_carries_the_candidate_the_component_and_the_outcomes() -> None:
    parent = seed("the text")
    reflection = Reflection(parent, "guide", (Outcome("t1", 0.0, feedback="wrong dir"),))
    assert reflection.module is parent.components["guide"]
    assert reflection.module.text == "the text"
    assert reflection.outcomes[0].feedback == "wrong dir"
    assert [component_for(i, ("a", "b", "c")) for i in range(4)] == ["a", "b", "c", "a"]


# ------------------------------------------------------------------------- evolve


async def test_the_seed_is_measured_on_every_task_first_and_once() -> None:
    world = World()
    result = await run(seed("hold"), TASKS, write=Scripted([None]), score=world, rollouts=2)
    assert world.asked[:4] == [(result.best.digest, task.name) for task in TASKS]
    assert world.rounds == [0] * 4, "the seed's measurement is round 0"
    assert result.spent == 8, "four tasks at two rollouts each, before any proposal"
    assert result.rounds == 3 and result.pool == {result.best.digest: result.best}
    assert len(world.asked) == 4, "a declined round scores nothing"


async def test_a_task_named_twice_is_one_column() -> None:
    world = World()
    result = await run(
        seed("hold"), (TASKS[0], TASKS[1], TASKS[0]), write=Scripted([None]), score=world
    )
    assert world.asked == [(result.best.digest, "t1"), (result.best.digest, "t2")]
    assert result.fitness.tasks == [str(TASKS[0]), str(TASKS[1])]


async def test_a_child_is_accepted_on_a_strict_minibatch_win_and_then_fully_evaluated() -> None:
    world, log = World(), Logged()
    result = await run(seed(), TASKS, write=Scripted(["t1 t2"]), score=world, log=log)
    child = next(member for digest, member in result.pool.items() if digest != seed().digest)
    assert child.components["guide"].text == "t1 t2"
    assert {task for digest, task in world.asked if digest == child.digest} == {
        "t1",
        "t2",
        "t3",
        "t4",
    }
    assert world.asked[4:6] == [(child.digest, "t1"), (child.digest, "t2")], "the window first"
    assert world.rounds == [0] * 4 + [1] * 4, "the window and the rest, both in round 1"
    assert result.spent == 4 + 2 + 2 and result.best == child
    assert log.rows[0]["accepted"] is True and log.rows[0]["child"] == child.digest
    assert (log.rows[0]["parent_mean"], log.rows[0]["child_mean"]) == (BASELINE, 1.0)
    # The child leads t1 and t2 and ties the seed on the rest: the seed is dominated.
    assert (log.rows[0]["pool"], log.rows[0]["frontier"], log.rows[0]["spent"]) == (2, 1, 8)


async def test_a_child_that_did_not_beat_its_parent_costs_one_minibatch() -> None:
    world, log = World(), Logged()
    result = await run(seed(), TASKS, write=Scripted(["mentions nothing"]), score=world, log=log)
    assert result.pool == {seed().digest: seed()} and result.best == seed()
    assert result.spent == 4 + 2, "the window only, never the rest of the list"
    assert result.rounds == 1 + 3, "scored and turned down, then three declines"
    assert log.rows[0]["accepted"] is False and log.rows[0]["child"] is not None
    assert log.rows[0]["child_mean"] == BASELINE == log.rows[0]["parent_mean"]
    assert [row["child"] for row in log.rows[1:]] == [None, None, None]
    tied = await run(seed("t1 t2"), TASKS, write=Scripted(["t2 t1"]), score=World())
    assert tied.best == seed("t1 t2"), "a tie buys nothing"


async def test_patience_counts_rounds_with_nothing_to_score_in_a_row() -> None:
    world, log = World(), Logged()
    result = await run(seed(), TASKS, write=Scripted([None]), score=world, patience=2, log=log)
    assert result.rounds == 2 and result.spent == 4
    assert [row["child"] for row in log.rows] == [None, None]
    spread = await run(
        seed(), TASKS, write=Scripted([None, None, "t1 t2", None, None, "t3"]), score=World()
    )
    assert len(spread.pool) == 3, "two quiet stretches of two, and the search ran through both"
    repeated = await run(seed(), TASKS, write=Scripted(["mentions nothing"] * 20), score=World())
    assert repeated.rounds == 1 + 3, "the same rejected rewrite is a decline, not a loop"


async def test_the_budget_ends_the_search_in_rollouts() -> None:
    world, log = World(), Logged()
    result = await run(seed(), TASKS, write=Fresh(), score=world, budget=10, log=log)
    assert result.spent == 10 and result.rounds == 3, "4 for the seed, then 2 per rejected child"
    assert [row["spent"] for row in log.rows] == [6, 8, 10]
    doubled = await run(seed(), TASKS, write=Fresh(), score=World(), budget=10, rollouts=2)
    assert doubled.spent == 12 and doubled.rounds == 1, "8 for the seed, 4 for one window"
    assert (await run(seed(), TASKS, write=Fresh(), score=World(), budget=3)).rounds == 0


async def test_every_component_is_rotated_and_reflected_over_the_whole_list() -> None:
    writes = Fresh()
    await run(seed("x", "a", "b", "c"), TASKS, write=writes, score=World(), budget=4 + 2 * 6)
    assert [reflection.component for reflection in writes.seen] == ["a", "b", "c"] * 2
    reflected: dict[str, set[str]] = {}
    for reflection in writes.seen:
        reflected.setdefault(reflection.component, set()).update(
            Path(outcome.task).name for outcome in reflection.outcomes
        )
    assert reflected == {name: {"t1", "t2", "t3", "t4"} for name in ("a", "b", "c")}


async def test_the_log_is_called_once_per_round_with_the_row() -> None:
    log = Logged()
    result = await run(seed(), TASKS, write=Scripted(["t1 t2", None]), score=World(), log=log)
    assert len(log.rows) == result.rounds == 1 + 3
    assert all(tuple(row) == ROW for row in log.rows)
    assert [row["round"] for row in log.rows] == [1, 2, 3, 4]
    assert log.rows[1]["parent"] == result.best.digest, "the frontier's winner is the parent"
    assert all(row["component"] == "guide" for row in log.rows)


async def test_a_dominated_candidate_is_not_a_parent_and_the_best_has_the_top_mean() -> None:
    writes = Scripted(["t1 t2 t3 t4", "t3"])
    result = await run(seed("do the work"), TASKS, write=writes, score=World())
    assert writes.seen[0].module.text == "do the work"
    assert {reflection.module.text for reflection in writes.seen[1:]} == {"t1 t2 t3 t4"}
    assert result.best.components["guide"].text == "t1 t2 t3 t4"
    wide, narrow, balanced = seed("t1 t2"), seed("t3"), seed("mid")
    pool = {member.digest: member for member in (wide, narrow, balanced)}
    seen = {
        wide.digest: {str(t): Outcome(str(t), 1.0 if i < 2 else 0.0) for i, t in enumerate(TASKS)},
        narrow.digest: {
            str(t): Outcome(str(t), 1.0 if i == 2 else 0.0) for i, t in enumerate(TASKS)
        },
        balanced.digest: {str(t): Outcome(str(t), 0.6) for t in TASKS},
    }
    scores = tally(pool, seen)
    assert scores.frontier() == {wide.digest, narrow.digest, balanced.digest}
    assert best(pool, scores) == balanced, "on the frontier by t4, and the highest aggregate"
    assert best({"x": wide}, tally(["x"], {})) == wide, "nothing measured: the pool's first"


def test_a_thinly_measured_row_does_not_outrank_a_complete_one() -> None:
    """`lucky` holds one cell at 1.0 and nothing else: on the frontier by that column,
    and on the mean alone the winner. Coverage comes first."""
    broad, lucky = seed("broad"), seed("lucky")
    pool = {member.digest: member for member in (broad, lucky)}
    seen = {
        broad.digest: {str(t): Outcome(str(t), 0.9) for t in TASKS},
        lucky.digest: {
            str(t): Outcome(str(t), 1.0 if i == 0 else None) for i, t in enumerate(TASKS)
        },
    }
    scores = tally(pool, seen)
    assert scores.frontier() == {broad.digest, lucky.digest}
    assert scores.aggregate(lucky.digest) > scores.aggregate(broad.digest)
    assert (scores.coverage(broad.digest), scores.coverage(lucky.digest)) == (4, 1)
    assert best(pool, scores) == broad


async def test_a_child_whose_full_evaluation_died_is_not_the_winner() -> None:
    """The live path to that row: the child wins its window, and the trials over the rest
    of the list produce nothing; it stays in the pool and on the frontier, and loses."""

    class Dying:
        async def __call__(
            self, candidate: Candidate, tasks: Sequence[Path], round_index: int
        ) -> list[Outcome]:
            text = candidate.components["guide"].text
            if "lucky" not in text:
                return [Outcome(str(task), 0.9) for task in tasks]
            return [Outcome(str(t), 1.0 if t.name in text else None) for t in tasks]

    result = await run(seed("steady"), TASKS, write=Scripted(["lucky t1 t2"]), score=Dying())
    assert len(result.pool) == 2 and result.fitness.frontier() == set(result.pool)
    assert result.best == seed("steady") and result.spent == 4 + 2 + 2


async def test_a_proposer_that_raises_costs_a_round_and_a_scorer_that_raises_the_run() -> None:
    class Broken:
        async def __call__(self, reflection: Reflection) -> Module | None:
            raise TimeoutError("the reflection model never answered")

    result = await run(seed("hold"), TASKS, write=Broken(), score=World())
    assert result.pool == {seed("hold").digest: seed("hold")} and result.rounds == 3

    class NoRuntime:
        async def __call__(
            self, candidate: Candidate, tasks: Sequence[Path], round_index: int
        ) -> list[Outcome]:
            raise RuntimeError("no container runtime")

    with pytest.raises(RuntimeError, match="no container runtime"):
        await run(seed(), TASKS, write=Scripted(["t1"]), score=NoRuntime())


async def test_nothing_to_evolve_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to evolve"):
        await run(seed("hold"), (), write=Scripted(["t1"]), score=World())
    with pytest.raises(ValueError, match="nothing to evolve"):
        await run(Candidate({}), TASKS, write=Scripted(["t1"]), score=World())


async def test_a_scorer_that_returns_nothing_still_spends() -> None:
    @dataclass
    class Silent:
        asked: int = 0

        async def __call__(
            self, candidate: Candidate, tasks: Sequence[Path], round_index: int
        ) -> list[Outcome]:
            self.asked += len(tasks)
            return []

    scorer = Silent()
    result = await run(seed("hold"), TASKS, write=Scripted(["one", "two", "three"]), score=scorer)
    assert scorer.asked > len(TASKS) and result.spent == scorer.asked
    assert result.pool == {seed("hold").digest: seed("hold")} and result.best == seed("hold")
    assert result.fitness.frontier() == set()


def test_the_winner_is_the_best_aggregate_of_the_pool_and_top_k_takes_the_frontier() -> None:
    """GEPA picks the best aggregate over every candidate, so a balanced 0.6 beats two
    specialists that each lead half the tasks; fst's top K stays the frontier's."""
    left, right, even = seed("left"), seed("right"), seed("even")
    pool = {member.digest: member for member in (left, right, even)}
    cells = [(left, (1, 1, 0, 0)), (right, (0, 0, 1, 1)), (even, (0.6, 0.6, 0.6, 0.6))]
    seen = {
        member.digest: {
            str(t): Outcome(str(t), float(v)) for t, v in zip(TASKS, values, strict=True)
        }
        for member, values in cells
    }
    scores = tally(pool, seen)
    assert scores.frontier() == {left.digest, right.digest}, "even leads no task"
    assert best(pool, scores) == even
    found = Evolution(even, pool, scores, rounds=0, spent=0)
    assert {member.digest for member in found.top(2)} == {left.digest, right.digest}


async def test_patience_waits_for_every_component_to_be_tried_once() -> None:
    world = World()
    result = await run(
        seed("", "a", "b", "c", "d"), TASKS, write=Scripted([None] * 8), score=world, patience=1
    )
    assert result.rounds == 4, "one quiet round per component, not one in all"
