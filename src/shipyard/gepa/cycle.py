"""The search: measure the seed on every task, then draw a parent off the frontier, ask
for a rewrite of one component, judge the child on a minibatch, and give it the rest of
the tasks only when it beat its parent there. The weights never move here."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shipyard.gepa.fitness import Fitness, Outcome, Scorer, mean, tally
from shipyard.gepa.propose import Proposer, component_for, propose
from shipyard.modules import Candidate

logger = logging.getLogger(__name__)

#: Told once per round what it did: `round`, `parent`, `component`, `child` (None for a
#: decline), `parent_mean` and `child_mean` over the minibatch, `accepted`, `pool`,
#: `frontier` and `spent`. A run's `log` takes exactly this.
Log = Callable[..., Any]


@dataclass(frozen=True)
class Evolution:
    """What one search ended with: the best of the frontier, the pool by digest, the
    fitness over it, the rounds it ran and the rollouts it spent."""

    best: Candidate
    pool: dict[str, Candidate]
    fitness: Fitness
    rounds: int
    spent: int


async def evolve(
    seed: Candidate,
    tasks: Sequence[Path],
    *,
    write: Proposer,
    score: Scorer,
    minibatch: int,
    budget: int,
    patience: int,
    rollouts: int = 1,
    rng: random.Random | None = None,
    log: Log | None = None,
) -> Evolution:
    """The loop, `budget` counted in rollouts at `rollouts` per task scored: a text is
    scored once per task and the measurement reused; `patience` rounds in a row with no
    child to score end it, since those spend nothing. A scorer that raises is not caught."""
    chosen = random.Random() if rng is None else rng
    told = log if log is not None else _silent
    columns = tuple(dict.fromkeys(Path(task) for task in tasks))
    names = tuple(seed.components)
    if not columns or not names:
        raise ValueError(f"nothing to evolve: {len(columns)} task(s), {len(names)} component(s)")
    pool: dict[str, Candidate] = {seed.digest: seed}
    declined: set[str] = set()
    seen: dict[str, dict[str, Outcome]] = {}
    spent = rounds = quiet = 0

    async def measure(candidate: Candidate, over: Sequence[Path]) -> list[Outcome]:
        """The outcomes over `over`, scoring what this text was not scored on yet under
        the current round; the rollouts asked for are counted, not the outcomes back."""
        nonlocal spent
        row = seen.setdefault(candidate.digest, {})
        todo = [task for task in over if str(task) not in row]
        if todo:
            spent += len(todo) * rollouts
            for outcome in await score(candidate, tuple(todo), rounds):
                row[outcome.task] = outcome
        return [row[str(task)] for task in over if str(task) in row]

    await measure(seed, columns)
    while spent < budget and quiet < patience:
        fitness = tally(pool, seen)
        front = sorted(fitness.frontier()) or [seed.digest]
        parent = pool[chosen.choice(front)]
        # The window walks per component, so each reflects over the whole list in turn.
        window = _minibatch(columns, rounds // len(names), minibatch)
        component = component_for(rounds, names)
        rounds += 1
        before = await measure(parent, window)
        try:
            child = await propose(parent, component, before, write)
        except Exception:
            logger.exception("the proposer failed; the pool stands")
            child = None
        row: dict[str, Any] = {
            "round": rounds,
            "parent": parent.digest,
            "component": component,
            "child": None,
            "parent_mean": mean(before),
            "child_mean": None,
            "accepted": False,
        }
        if child is None or child.digest in pool or child.digest in declined:
            quiet += 1
        else:
            quiet = 0
            after = await measure(child, window)
            row.update(child=child.digest, child_mean=mean(after))
            if _beats(before, after):
                row["accepted"] = True
                pool[child.digest] = child
                await measure(child, columns)
            else:
                declined.add(child.digest)
        told(**row, pool=len(pool), frontier=len(tally(pool, seen).frontier()), spent=spent)
    fitness = tally(pool, seen)
    return Evolution(best(pool, fitness), pool, fitness, rounds, spent)


def best(pool: dict[str, Candidate], fitness: Fitness) -> Candidate:
    """The frontier member measured on the most tasks, then with the highest aggregate,
    ties by digest: a child whose full evaluation mostly died is one lucky cell at 1.0,
    not the winner. The first of the pool when nothing was measured."""
    front = sorted(fitness.frontier())
    if not front:
        return next(iter(pool.values()))
    return pool[
        max(
            front,
            key=lambda digest: (
                fitness.coverage(digest),
                fitness.aggregate(digest) or 0.0,
                digest,
            ),
        )
    ]


def _minibatch(tasks: Sequence[Path], window: int, size: int) -> list[Path]:
    """Window `window` of `size` consecutive tasks in the order given, wrapping: which
    tasks a round reflects on is then something a caller can predict."""
    width = max(1, min(size, len(tasks)))
    start = (window * width) % len(tasks)
    doubled = list(tasks) + list(tasks)
    return doubled[start : start + width]


def _beats(parent: Sequence[Outcome], child: Sequence[Outcome]) -> bool:
    """Whether the child's mean is strictly above the parent's where both were measured;
    a tie buys nothing and costs a full evaluation to find out."""
    shared = {outcome.task for outcome in parent if outcome.measured} & {
        outcome.task for outcome in child if outcome.measured
    }
    if not shared:
        return False
    before = mean([outcome for outcome in parent if outcome.task in shared])
    after = mean([outcome for outcome in child if outcome.task in shared])
    return before is not None and after is not None and after > before


def _silent(**row: Any) -> None:
    """The log when nobody asked for one."""
