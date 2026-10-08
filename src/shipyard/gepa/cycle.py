"""The search (GEPA, Agrawal et al., arXiv 2507.19457, Alg. 1): score the seeds on the Pareto
tasks; then each round draw a parent off the frontier in proportion to the tasks it leads, run
it on a minibatch sampled from the feedback tasks, ask for a rewrite of one component from
those traces, run the child on the same minibatch, and score the child on the Pareto tasks
only when it beat its parent there. The weights never move here."""

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

#: Told once per round what it did: `round`, `parent`, `component`, `child` (None for a
#: decline), `parent_mean` and `child_mean` over the minibatch, `accepted`, `pool`,
#: `frontier` and `spent`. A run's `log` takes exactly this.
Log = Callable[..., Any]


@dataclass(frozen=True)
class Evolution:
    """What one search ended with: the best of the pool, the pool by digest, the fitness
    over the Pareto tasks, the rounds it ran and the rollouts it spent."""

    best: Candidate
    pool: dict[str, Candidate]
    fitness: Fitness
    rounds: int
    spent: int

    def top(self, k: int) -> list[Candidate]:
        """The k candidates to carry on: the frontier ranked as `best` ranks it, then the
        rest of the pool, the ranking repeated when the pool holds fewer than k."""
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
    patience: int,
    rollouts: int = 1,
    rng: random.Random | None = None,
    log: Log | None = None,
) -> Evolution:
    """The loop over GEPA's two task lists: minibatches are drawn from `tasks` (D_feedback),
    and every seed and accepted child is scored on `pareto` (D_pareto), which the frontier
    and the winner are read from; None is `tasks` again (GEPA 6, FST's anchor set). `budget`
    counts rollouts, `rollouts` per task scored, every scoring included; `patience` rounds in
    a row with no child to score end it. `seed` is one candidate or a population. `rng`
    draws the parents and shuffles the minibatches. A scorer that raises is not caught."""
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
    pool: dict[str, Candidate] = {member.digest: member for member in seeds}
    declined: set[str] = set()
    seen: dict[str, dict[str, Outcome]] = {}
    drawn = minibatches(feedback, minibatch, chosen)
    spent = rounds = quiet = 0

    async def measure(candidate: Candidate, over: Sequence[Path]) -> list[Outcome]:
        """Fresh outcomes over `over` under the current round; the rollouts asked for are
        counted, not the outcomes back."""
        nonlocal spent
        spent += len(over) * rollouts
        return list(await score(candidate, tuple(over), rounds))

    async def admit(candidate: Candidate) -> None:
        """The candidate's Pareto row, scored afresh on every Pareto task: a minibatch it
        won on is a draw it was picked for, and would favour lucky children."""
        seen[candidate.digest] = {found.task: found for found in await measure(candidate, columns)}

    for member in list(pool.values()):
        await admit(member)
    while spent < budget and quiet < max(patience, len(names)):
        leads = tally(pool, seen).leads() or {seeds[0].digest: 1}
        parent = pool[chosen.choices(list(leads), weights=list(leads.values()))[0]]
        batch = next(drawn)
        component = component_for(rounds, names)
        rounds += 1
        # The parent runs afresh, as in GEPA: the reflector reads traces of this round, and
        # the child is compared with a score from the same round.
        before = await measure(parent, batch)
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
            after = await measure(child, batch)
            row.update(child=child.digest, child_mean=mean(after))
            if _beats(before, after):
                row["accepted"] = True
                pool[child.digest] = child
                await admit(child)
            else:
                declined.add(child.digest)
        told(**row, pool=len(pool), frontier=len(tally(pool, seen).frontier()), spent=spent)
    fitness = tally(pool, seen)
    return Evolution(best(pool, fitness), pool, fitness, rounds, spent)


def best(pool: dict[str, Candidate], fitness: Fitness) -> Candidate:
    """The candidate measured on the most tasks, then with the highest aggregate, ties by
    digest, over the whole pool: GEPA's pick is the best aggregate, and the frontier is
    for drawing parents. The first of the pool when nothing was measured."""
    measured = [digest for digest in pool if fitness.coverage(digest)]
    if not measured:
        return next(iter(pool.values()))
    key = lambda digest: (fitness.coverage(digest), fitness.aggregate(digest) or 0.0, digest)  # noqa: E731
    return pool[max(measured, key=key)]


def _ranked(pool: dict[str, Candidate], fitness: Fitness) -> list[Candidate]:
    """The frontier by (coverage, aggregate, digest), best first, then the rest of the pool
    by (coverage, aggregate) in the order it joined."""

    def key(digest: str) -> tuple[int, float]:
        return fitness.coverage(digest), fitness.aggregate(digest) or 0.0

    front = sorted(fitness.frontier(), key=lambda digest: (*key(digest), digest), reverse=True)
    rest = sorted((digest for digest in pool if digest not in front), key=key, reverse=True)
    return [pool[digest] for digest in front + rest]


def minibatches(tasks: Sequence[Path], size: int, rng: random.Random) -> Iterator[list[Path]]:
    """GEPA's minibatches (Alg. 1 line 9), `size` tasks each, sampled without replacement
    in passes over the tasks, each pass a fresh shuffle, as the authors' sampler does; a
    pass's short last minibatch is filled from the start of that pass, so none repeats a task."""
    width = max(1, min(size, len(tasks)))
    while True:
        order = list(tasks)
        rng.shuffle(order)
        for start in range(0, len(order), width):
            cut = order[start : start + width]
            yield cut + order[: width - len(cut)]


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
