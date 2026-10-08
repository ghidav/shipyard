"""The search (GEPA, Agrawal et al., arXiv 2507.19457, Alg. 1).

The seeds are scored on the Pareto tasks. Each round then draws a parent off the frontier,
with weight proportional to the tasks it leads, and runs it on a minibatch sampled from the
feedback tasks. Unless the parent is perfect on every task of the minibatch, a reflector
rewrites one component from those traces. The child runs on the same minibatch and is scored
on the Pareto tasks only if it beat its parent on the minibatch."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shipyard.gepa.fitness import Fitness, Outcome, Scorer, mean, tally
from shipyard.gepa.propose import Proposer, component_for, propose
from shipyard.modules import Candidate

logger = logging.getLogger(__name__)

#: Called once per round with `round`, `parent`, `component` (None when skipped),
#: `skipped` (why no rewrite was asked for, or None), `child` (None when no child was
#: scored), `parent_mean` and `child_mean` over the minibatch, `accepted`, `pool`,
#: `frontier` and `spent`. A run's `log` accepts these keys.
Log = Callable[..., Any]

#: Harbor's maximum reward. A round whose parent reaches it on every task of the minibatch
#: asks for no rewrite, since a child must score strictly above its parent. GEPA's released
#: code skips such rounds by default (gepa 0.1.4, api.py: skip_perfect_score = True,
#: perfect_score = 1.0).
PERFECT = 1.0
#: What a skipped round's row says in `skipped`.
SKIPPED_PERFECT = "perfect"


@dataclass(frozen=True)
class Evolution:
    """The result of one search: the best of the pool, the pool by digest, the fitness over
    the Pareto tasks, the rounds it ran and the rollouts it spent."""

    best: Candidate
    pool: dict[str, Candidate]
    fitness: Fitness
    rounds: int
    spent: int

    def top(self, k: int) -> list[Candidate]:
        """The k candidates to carry on: the frontier ranked as `best` ranks it, then the
        rest of the pool. The ranking repeats when the pool holds fewer than k."""
        ranked = _ranked(self.pool, self.fitness)
        return [ranked[at % len(ranked)] for at in range(k)]


async def evolve(
    seed: Candidate | Sequence[Candidate],
    tasks: Sequence[Path],
    *,
    pareto: Sequence[Path] | None = None,
    write: Proposer,
    score: Scorer,
    minibatch: int,
    budget: int,
    patience: int | None = None,
    rollouts: int = 1,
    rng: random.Random | None = None,
    log: Log | None = None,
) -> Evolution:
    """The loop over GEPA's two task lists.

    Minibatches are drawn from `tasks` (D_feedback). Every seed and accepted child is scored
    on `pareto` (D_pareto), and the frontier and the winner are read from those scores.
    `pareto=None` reuses `tasks` (GEPA 6, FST's anchor set).

    `budget` counts rollouts: `rollouts` per task scored, every scoring included. Every round
    spends the parent's, so the budget ends the search.

    `patience`, when set, ends the search after that many consecutive rounds, skipped ones
    included, in which the best Pareto mean of the pool did not rise. This is GEPA's
    `NoImprovementStopper` (gepa 0.1.4, utils/stop_condition.py), which its code adds only
    when asked.

    `seed` is one candidate or a population. `rng` draws the parents and shuffles the
    minibatches. An exception from the scorer propagates."""
    chosen = random.Random() if rng is None else rng
    told = log if log is not None else _silent
    feedback = tuple(dict.fromkeys(Path(task) for task in tasks))
    columns = feedback if pareto is None else tuple(dict.fromkeys(Path(task) for task in pareto))
    seeds = [seed] if isinstance(seed, Candidate) else list(seed)
    names = tuple(seeds[0].components) if seeds else ()
    if not feedback or not columns or not names:
        raise ValueError(
            f"nothing to evolve: {len(feedback)} task(s) to reflect on, {len(columns)} to "
            f"select on, {len(names)} component(s)"
        )
    if rollouts < 1:
        raise ValueError(f"rollouts must be at least 1; got {rollouts}")
    pool: dict[str, Candidate] = {member.digest: member for member in seeds}
    declined: set[str] = set()
    seen: dict[str, dict[str, Outcome]] = {}
    drawn = minibatches(feedback, minibatch, chosen)
    spent = rounds = asked = stale = 0

    async def measure(candidate: Candidate, over: Sequence[Path]) -> list[Outcome]:
        """Score `candidate` over `over` in the current round. `spent` grows by the rollouts
        requested, even if fewer outcomes come back."""
        nonlocal spent
        spent += len(over) * rollouts
        return list(await score(candidate, tuple(over), rounds))

    async def admit(candidate: Candidate) -> None:
        """Score the candidate afresh on every Pareto task and store its row. Reusing its
        minibatch scores would favour lucky children."""
        seen[candidate.digest] = {found.task: found for found in await measure(candidate, columns)}

    for member in list(pool.values()):
        await admit(member)
    high = peak(tally(pool, seen))
    while spent < budget and (patience is None or stale < patience):
        leads = tally(pool, seen).leads() or {seeds[0].digest: 1}
        parent = pool[chosen.choices(list(leads), weights=list(leads.values()))[0]]
        batch = next(drawn)
        rounds += 1
        # The parent runs afresh, as in GEPA, so the reflector reads traces of this round
        # and the child is compared with a score from the same round.
        before = await measure(parent, batch)
        row: dict[str, Any] = {
            "round": rounds,
            "parent": parent.digest,
            "component": None,
            "skipped": None,
            "child": None,
            "parent_mean": mean(before),
            "child_mean": None,
            "accepted": False,
        }
        child: Candidate | None = None
        if perfect(batch, before):
            row["skipped"] = SKIPPED_PERFECT
        else:
            # Only rounds that ask for a rewrite take a turn, as GEPA's selector is not
            # consulted on a skip.
            component = component_for(asked, names)
            asked += 1
            row["component"] = component
            try:
                child = await propose(parent, component, before, write)
            except Exception:
                logger.exception("the proposer failed; the pool is unchanged")
        if child is not None and child.digest not in pool and child.digest not in declined:
            after = await measure(child, batch)
            row.update(child=child.digest, child_mean=mean(after))
            if _beats(before, after):
                row["accepted"] = True
                pool[child.digest] = child
                await admit(child)
            else:
                declined.add(child.digest)
        fitness = tally(pool, seen)
        now = peak(fitness)
        stale = 0 if now > high else stale + 1
        high = max(high, now)
        told(**row, pool=len(pool), frontier=len(fitness.frontier()), spent=spent)
    fitness = tally(pool, seen)
    return Evolution(best(pool, fitness), pool, fitness, rounds, spent)


def perfect(batch: Sequence[Path], outcomes: Sequence[Outcome]) -> bool:
    """Whether every task of the minibatch scored `PERFECT` or above. Tasks are compared one
    by one, as GEPA's code compares each example's score, not by their mean. A task's score
    is the mean over its measured rollouts. A task with no measured rollout, or no outcome,
    is not perfect."""
    reached = {
        outcome.task
        for outcome in outcomes
        if outcome.reward is not None and outcome.reward >= PERFECT
    }
    return all(str(task) in reached for task in batch)


def peak(fitness: Fitness) -> float:
    """The best Pareto mean in the pool, which `patience` watches. GEPA's stopper reads the
    highest of `program_full_scores_val_set`. -inf while nothing is measured."""
    found = [fitness.aggregate(digest) for digest in fitness.scores]
    return max((value for value in found if value is not None), default=float("-inf"))


def best(pool: dict[str, Candidate], fitness: Fitness) -> Candidate:
    """The candidate of the whole pool measured on the most tasks, then with the highest
    aggregate, ties broken by digest. GEPA picks the best aggregate. The frontier serves
    only to draw parents. The first of the pool when nothing was measured."""
    measured = [digest for digest in pool if fitness.coverage(digest)]
    if not measured:
        return next(iter(pool.values()))
    key = lambda digest: (fitness.coverage(digest), fitness.aggregate(digest) or 0.0, digest)  # noqa: E731
    return pool[max(measured, key=key)]


def _ranked(pool: dict[str, Candidate], fitness: Fitness) -> list[Candidate]:
    """The frontier by (coverage, aggregate, digest), best first, then the rest of the pool
    by (coverage, aggregate), ties in the order they joined."""

    def key(digest: str) -> tuple[int, float]:
        return fitness.coverage(digest), fitness.aggregate(digest) or 0.0

    front = sorted(fitness.frontier(), key=lambda digest: (*key(digest), digest), reverse=True)
    rest = sorted((digest for digest in pool if digest not in front), key=key, reverse=True)
    return [pool[digest] for digest in front + rest]


def minibatches(tasks: Sequence[Path], size: int, rng: random.Random) -> Iterator[list[Path]]:
    """GEPA's minibatches (Alg. 1 line 9) of `size` tasks. Tasks are sampled without
    replacement in passes, each pass a fresh shuffle, as the authors' sampler does. A pass's
    short last minibatch is filled from the start of that pass, so no minibatch repeats a
    task."""
    width = max(1, min(size, len(tasks)))
    while True:
        order = list(tasks)
        rng.shuffle(order)
        for start in range(0, len(order), width):
            cut = order[start : start + width]
            yield cut + order[: width - len(cut)]


def _beats(parent: Sequence[Outcome], child: Sequence[Outcome]) -> bool:
    """Whether the child's mean is strictly above the parent's over the tasks both measured.
    A tie does not count, since accepting it costs a full evaluation for no gain."""
    shared = {outcome.task for outcome in parent if outcome.measured} & {
        outcome.task for outcome in child if outcome.measured
    }
    if not shared:
        return False
    before = mean([outcome for outcome in parent if outcome.task in shared])
    after = mean([outcome for outcome in child if outcome.task in shared])
    return before is not None and after is not None and after > before


def _silent(**row: Any) -> None:
    """The log used when none is given."""
