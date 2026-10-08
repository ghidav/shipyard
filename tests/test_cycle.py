"""The search on a fake scorer and a fake proposer: four tasks and one rule. A text scores
1.0 on a task whose name it mentions and `BASELINE` on one it does not, so each reward in a
case follows from the case. Minibatches are drawn at random, so a case that needs a child to
win its minibatch uses `Echo`, whose child names the minibatch it was shown. A text that
names every task of a minibatch is perfect there, and its round is skipped."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from shipyard.gepa.cycle import Evolution, best, evolve, minibatches, perfect
from shipyard.gepa.fitness import Outcome, tally
from shipyard.gepa.propose import Reflection, component_for, propose
from shipyard.modules import SKILL_FILE, Candidate, Module

TASKS = tuple(Path("tasks/four") / name for name in ("t1", "t2", "t3", "t4"))
HELD = tuple(Path("tasks/held") / name for name in ("h1", "h2", "h3"))
BASELINE = 0.1
ROW = (
    "round",
    "parent",
    "component",
    "skipped",
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
    """A scorer: 1.0 on a task the candidate's text names, `BASELINE` otherwise. It records
    what it was asked, in which round, and one call per job. The feedback names the round."""

    asked: list[tuple[str, str]] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)
    jobs: list[tuple[str, tuple[str, ...], int]] = field(default_factory=list)

    async def __call__(
        self, candidate: Candidate, tasks: Sequence[Path], round_index: int
    ) -> list[Outcome]:
        text = " ".join(module.text for module in candidate.components.values())
        self.jobs.append((candidate.digest, tuple(task.name for task in tasks), round_index))
        out = []
        for task in tasks:
            self.asked.append((candidate.digest, task.name))
            self.rounds.append(round_index)
            hit = task.name in text
            reward = 1.0 if hit else BASELINE
            out.append(Outcome(str(task), reward, feedback=f"seen in round {round_index}"))
        return out


@dataclass
class Table:
    """A scorer reading each task's reward off a table, for any candidate. None is a task
    with no measured rollout."""

    rewards: dict[str, float | None]

    async def __call__(
        self, candidate: Candidate, tasks: Sequence[Path], round_index: int
    ) -> list[Outcome]:
        return [Outcome(str(task), self.rewards[task.name]) for task in tasks]


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
class Echo:
    """A proposer whose child names the tasks it was shown, after the parent's text, so it
    wins any minibatch its parent did not already solve."""

    seen: list[Reflection] = field(default_factory=list)

    async def __call__(self, reflection: Reflection) -> Module | None:
        self.seen.append(reflection)
        named = " ".join(Path(outcome.task).name for outcome in reflection.outcomes)
        return _module(f"{reflection.module.text} {named}".strip(), reflection.component)


@dataclass
class Logged:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **row: Any) -> None:
        self.rows.append(row)


def drawn(feedback: Sequence[Path], size: int, rng: random.Random, rounds: int) -> list[list[str]]:
    """By task name, the minibatches of a search's first `rounds` rounds: what
    `minibatches(feedback, size, rng)` yields, each round's parent drawn from the same `rng`
    just before its minibatch, as `evolve` draws them."""
    sampled = minibatches(feedback, size, rng)
    found = []
    for _ in range(rounds):
        rng.random()
        found.append([task.name for task in next(sampled)])
    return found


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
    assert world.rounds[:4] == [0] * 4, "the seed's measurement is round 0"
    assert result.rounds == 3 and result.pool == {result.best.digest: result.best}
    rounds = [(job[0], len(job[1]), job[2]) for job in world.jobs[1:]]
    assert rounds == [(seed("hold").digest, 2, at) for at in (1, 2, 3)], (
        "after it, a declined round scores its parent on the minibatch and nothing else"
    )
    assert result.spent == 8 + 3 * 2 * 2, "four tasks at two rollouts, then a minibatch a round"


async def test_a_task_named_twice_is_one_column() -> None:
    world = World()
    result = await run(
        seed("hold"), (TASKS[0], TASKS[1], TASKS[0]), write=Scripted([None]), score=world
    )
    assert world.asked[:2] == [(result.best.digest, "t1"), (result.best.digest, "t2")]
    assert result.fitness.tasks == [str(TASKS[0]), str(TASKS[1])]
    assert all(sorted(job[1]) == ["t1", "t2"] for job in world.jobs[1:])


async def test_a_child_is_accepted_on_a_strict_minibatch_win_and_then_fully_evaluated() -> None:
    world, log, writes = World(), Logged(), Echo()
    result = await run(seed(), TASKS, write=writes, score=world, log=log, patience=1, budget=12)
    child = next(member for digest, member in result.pool.items() if digest != seed().digest)
    (parent_job, child_job, full_job) = world.jobs[1:]
    assert parent_job[0] == seed().digest and child_job[0] == child.digest
    assert child_job[1] == parent_job[1], "the child runs on the parent's minibatch"
    assert child.components["guide"].text == " ".join(parent_job[1])
    assert full_job == (child.digest, ("t1", "t2", "t3", "t4"), 1), "then every task, afresh"
    assert world.rounds == [0] * 4 + [1] * 8, "the minibatch twice and the rest, in round 1"
    assert result.spent == 4 + 2 + 2 + 4 and result.best == child
    assert log.rows[0]["accepted"] is True and log.rows[0]["child"] == child.digest
    assert (log.rows[0]["parent_mean"], log.rows[0]["child_mean"]) == (BASELINE, 1.0)
    # The child leads its two tasks and ties the seed on the rest: the seed is dominated.
    assert (log.rows[0]["pool"], log.rows[0]["frontier"], log.rows[0]["spent"]) == (2, 1, 12)


async def test_a_child_that_did_not_beat_its_parent_costs_two_minibatches() -> None:
    world, log = World(), Logged()
    result = await run(seed(), TASKS, write=Scripted(["mentions nothing"]), score=world, log=log)
    assert result.pool == {seed().digest: seed()} and result.best == seed()
    assert [row["spent"] for row in log.rows] == [4 + 2 + 2, 10, 12], (
        "the parent's minibatch and the child's, never the rest of the list"
    )
    assert result.rounds == 1 + 2, "scored and turned down, then two declines: patience = 3"
    assert log.rows[0]["accepted"] is False and log.rows[0]["child"] is not None
    assert log.rows[0]["child_mean"] == BASELINE == log.rows[0]["parent_mean"]
    assert [row["child"] for row in log.rows[1:]] == [None, None]
    tied = await run(seed("t1 t2"), TASKS, write=Scripted(["t2 t1"]), score=World())
    assert tied.best == seed("t1 t2"), "a tie buys nothing"


async def test_patience_counts_rounds_in_which_the_best_pareto_mean_did_not_rise() -> None:
    """GEPA's NoImprovementStopper (gepa 0.1.4, utils/stop_condition.py): every round
    counts, skipped ones too, and only a higher best Pareto mean in the pool clears it."""
    log = Logged()
    declined = await run(seed(), TASKS, write=Scripted([None]), score=World(), patience=2, log=log)
    assert declined.rounds == 2 and declined.spent == 4 + 2 + 2, (
        "a round with no child runs its parent"
    )
    assert [row["child"] for row in log.rows] == [None, None]
    turned = await run(seed(), TASKS, write=Fresh(), score=World(), patience=2)
    assert turned.rounds == 2, "a child scored and turned down is no rise"
    log = Logged()
    held = await run(seed(), TASKS, pareto=HELD, write=Echo(), score=World(), patience=2, log=log)
    assert [row["accepted"] for row in log.rows] == [True, True] and held.rounds == 2, (
        "accepted on its minibatch, and BASELINE on the Pareto tasks like the seed: no rise"
    )
    log = Logged()
    rising = await run(seed(), TASKS, write=Echo(), score=World(), patience=1, log=log)
    assert [(row["accepted"], row["skipped"]) for row in log.rows] == [
        (True, None),
        (True, None),
        (False, "perfect"),
    ], "two rises, then a child perfect on every task: its round is skipped, and counts"
    assert rising.rounds == 3 and rising.fitness.aggregate(rising.best.digest) == 1.0


async def test_with_no_patience_the_budget_alone_ends_the_search() -> None:
    """Patience is unset by default, as in GEPA's code. Every round spends its parent's
    rollouts, so the budget ends even a search whose reflector never answers."""
    result = await run(seed(), TASKS, write=Scripted([]), score=World(), patience=None, budget=20)
    assert result.rounds == 8 and result.spent == 20, "4 for the seed, then 2 a round"
    with pytest.raises(ValueError, match="rollouts must be at least 1"):
        await run(seed(), TASKS, write=Scripted([]), score=World(), rollouts=0)


async def test_a_round_whose_parent_is_perfect_on_its_minibatch_asks_for_no_rewrite() -> None:
    """GEPA's code skips it by default (gepa 0.1.4, api.py: skip_perfect_score = True,
    perfect_score = 1.0): no child can beat a parent at 1.0 on every task. The parent's
    run is still spent."""
    world, log, writes = World(), Logged(), Scripted(["anything"])
    solved = seed("t1 t2 t3 t4")
    result = await run(solved, TASKS, write=writes, score=world, log=log, patience=None, budget=10)
    assert writes.seen == [] and result.pool == {solved.digest: solved}
    assert result.rounds == 3 and result.spent == 4 + 3 * 2, "each round spends the parent's run"
    assert [job[0] for job in world.jobs] == [solved.digest] * 4, "the seed, then three parents"
    for row in log.rows:
        assert tuple(row) == ROW
        assert (row["skipped"], row["component"], row["child"], row["child_mean"]) == (
            "perfect",
            None,
            None,
            None,
        )
        assert row["parent_mean"] == 1.0 and row["accepted"] is False
    shown, log = Scripted([]), Logged()
    table = Table({"t1": 1.0, "t2": None})
    await run(seed(), TASKS[:2], write=shown, score=table, log=log, patience=1)
    assert (log.rows[0]["parent_mean"], log.rows[0]["skipped"]) == (1.0, None)
    assert len(shown.seen) == 1, "a mean of 1.0 over the measured tasks, and one unmeasured"


def test_a_minibatch_is_perfect_when_every_task_is_not_when_the_mean_is() -> None:
    """Task by task, as GEPA's code compares each example's score. A task's score is the
    mean over its measured rollouts."""
    one, two = (str(task) for task in TASKS[:2])
    assert perfect(TASKS[:2], [Outcome(one, 1.0), Outcome(two, 1.0)])
    assert not perfect(TASKS[:2], [Outcome(one, 1.5), Outcome(two, 0.5)]), "a mean of 1.0"
    assert not perfect(TASKS[:2], [Outcome(one, 1.0), Outcome(two, None)]), "none measured"
    assert not perfect(TASKS[:2], [Outcome(one, 1.0)]), "no outcome back"
    assert not perfect(TASKS[:2], [Outcome(one, 1.0), Outcome(two, 0.5)])


async def test_a_skipped_round_takes_no_components_turn() -> None:
    """GEPA's code consults its round-robin selector only past the skip: a text that names
    t1..t3 is perfect on three minibatches of one in every four, and each component is
    offered in turn on the fourth."""
    writes, log = Scripted([]), Logged()
    perfect_on_three = seed("t1 t2 t3", "a", "b", "c")
    budget = 4 + 12
    await run(
        perfect_on_three,
        TASKS,
        write=writes,
        score=World(),
        log=log,
        minibatch=1,
        budget=budget,
        patience=None,
    )
    assert [reflection.component for reflection in writes.seen] == ["a", "b", "c"]
    assert [row["component"] for row in log.rows if row["skipped"] is None] == ["a", "b", "c"]
    assert [row["skipped"] for row in log.rows].count("perfect") == 9


async def test_the_budget_ends_the_search_in_rollouts() -> None:
    world, log = World(), Logged()
    result = await run(seed(), TASKS, write=Fresh(), score=world, budget=10, log=log)
    assert result.spent == 12 and result.rounds == 2, "4 for the seed, then 2 + 2 a round"
    assert [row["spent"] for row in log.rows] == [8, 12]
    doubled = await run(seed(), TASKS, write=Fresh(), score=World(), budget=10, rollouts=2)
    assert doubled.spent == 16 and doubled.rounds == 1, "8 for the seed, 4 + 4 for one round"
    assert (await run(seed(), TASKS, write=Fresh(), score=World(), budget=3)).rounds == 0


async def test_every_component_is_rotated_and_each_round_draws_its_own_minibatch() -> None:
    writes, world = Fresh(), World()
    budget = 4 + 4 * 6
    await run(
        seed("x", "a", "b", "c"), TASKS, write=writes, score=world, budget=budget, patience=None
    )
    assert [reflection.component for reflection in writes.seen] == ["a", "b", "c"] * 2
    shown = [{Path(outcome.task).name for outcome in r.outcomes} for r in writes.seen]
    assert shown[0] | shown[1] == shown[2] | shown[3] == {"t1", "t2", "t3", "t4"}, (
        "a pass over the tasks every two rounds, whatever the component"
    )


async def test_the_parent_runs_afresh_every_round_and_its_traces_are_that_rounds() -> None:
    """GEPA Alg. 1 lines 10 and 13: the parent is run on the minibatch each round, the
    reflector reads those traces, and both means come from the same round."""
    world, log, writes = World(), Logged(), Fresh()
    result = await run(seed(), TASKS, write=writes, score=world, log=log, budget=4 + 4 * 3)
    parents = [job for job in world.jobs[1:] if job[0] == seed().digest]
    children = [job for job in world.jobs[1:] if job[0] != seed().digest]
    assert [job[2] for job in parents] == [job[2] for job in children] == [1, 2, 3]
    assert [job[1] for job in parents] == [job[1] for job in children]
    for at, reflection in enumerate(writes.seen, start=1):
        assert {outcome.feedback for outcome in reflection.outcomes} == {f"seen in round {at}"}
    assert result.spent == 4 + 3 * (2 + 2), "both halves of every round are spent"


async def test_the_pareto_tasks_score_what_is_kept_and_the_feedback_tasks_teach() -> None:
    """GEPA Alg. 1: minibatches come from D_feedback; the seed and every accepted child are
    scored on D_pareto, which alone holds the frontier and picks the winner."""
    world, writes = World(), Echo()
    result = await run(seed(), TASKS, pareto=HELD, write=writes, score=world, budget=40)
    held = tuple(task.name for task in HELD)
    assert world.jobs[0] == (seed().digest, held, 0), "the seed on the Pareto tasks"
    scored = [job for job in world.jobs[1:] if job[1] != held]
    assert scored and all(set(job[1]) <= {"t1", "t2", "t3", "t4"} for job in scored)
    assert [len(job[1]) for job in scored] == [2] * len(scored)
    admitted = [job for job in world.jobs[1:] if job[1] == held]
    assert [job[0] for job in admitted] == [d for d in result.pool if d != seed().digest]
    assert admitted, "a child that names its minibatch beats its parent there"
    assert result.fitness.tasks == [str(task) for task in HELD]
    assert result.fitness.aggregate(result.best.digest) == pytest.approx(BASELINE), (
        "a feedback task's win is no Pareto score: every text scores BASELINE on h1..h3"
    )
    assert result.spent == len(world.asked)


async def test_minibatches_are_sampled_in_seeded_passes_without_repeats() -> None:
    """GEPA Alg. 1 line 9 samples each minibatch from D_feedback; each pass is a fresh
    shuffle drawn with the seed, so a sorted dataset does not fix the order."""
    five = tuple(Path(f"t{at}") for at in range(5))
    drawn = minibatches(five, 2, random.Random(7))
    passes = [[next(drawn) for _ in range(3)] for _ in range(4)]
    for each in passes:
        assert {task for batch in each for task in batch} == set(five), "a pass covers them all"
        assert all(len(batch) == len(set(batch)) == 2 for batch in each)
        assert each[2][1] == each[0][0], "the short last one is filled from the pass's start"
    again = minibatches(five, 2, random.Random(7))
    assert [next(again) for _ in range(12)] == [batch for each in passes for batch in each]
    firsts = {tuple(next(minibatches(five, 2, random.Random(s)))) for s in range(10)}
    assert len(firsts) > 1, "the seed, not the listing, orders them"
    assert next(minibatches(five[:1], 3, random.Random(0))) == [five[0]], (
        "never wider than the list"
    )


async def test_the_rng_draws_the_minibatches() -> None:
    """Each round's minibatch is the next one `minibatches` yields from `rng`: the same seed
    draws the same sequence, and another seed another one."""
    found = {}
    for at in (1, 2):
        world = World()
        rng = random.Random(at)
        await run(seed(), TASKS, write=Fresh(), score=world, rng=rng, budget=28, patience=None)
        parents = [list(job[1]) for job in world.jobs[1::2]]
        assert parents == drawn(TASKS, 2, random.Random(at), 6), (
            "each of six rounds opens with its parent's job"
        )
        found[at] = parents
    assert found[1] != found[2]


async def test_a_dominated_candidate_is_not_a_parent_and_the_best_has_the_top_mean() -> None:
    writes, log = Scripted(["t1 t2 t3 t4", "t3"]), Logged()
    result = await run(seed("do the work"), TASKS, write=writes, score=World(), log=log)
    assert writes.seen[0].module.text == "do the work"
    assert result.best.components["guide"].text == "t1 t2 t3 t4"
    assert {row["parent"] for row in log.rows[1:]} == {result.best.digest}, "never the seed"
    assert len(writes.seen) == 1, "the child is perfect on every minibatch: no more rewrites"
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
    """`lucky` holds one cell at 1.0 and no others. It is on the frontier by that column and
    has the higher mean, but coverage comes first."""
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
    """The live path to that row. The child wins its minibatch, and the trials over the
    Pareto tasks produce nothing past the two it names. It stays in the pool and on the
    frontier, and loses."""

    class Dying:
        async def __call__(
            self, candidate: Candidate, tasks: Sequence[Path], round_index: int
        ) -> list[Outcome]:
            text = candidate.components["guide"].text
            if not any(task.name in text for task in TASKS):
                return [Outcome(str(task), 0.9) for task in tasks]
            return [Outcome(str(t), 1.0 if t.name in text else None) for t in tasks]

    result = await run(seed("steady"), TASKS, write=Echo(), score=Dying(), budget=12)
    assert len(result.pool) == 2 and result.fitness.frontier() == set(result.pool)
    child = next(digest for digest in result.pool if digest != seed("steady").digest)
    assert result.fitness.coverage(child) == 2
    assert result.best == seed("steady") and result.spent == 4 + 2 + 2 + 4


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
    with pytest.raises(ValueError, match="0 to select on"):
        await run(seed("hold"), TASKS, pareto=(), write=Scripted(["t1"]), score=World())
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
