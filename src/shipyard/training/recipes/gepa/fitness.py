"""Per-task fitness for the GEPA search, and Pareto selection over it.

A candidate's fitness is a vector over tasks, so the frontier can keep a candidate that alone
solves one task. A rollout that could not be measured is left out of every mean."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from shipyard.modules import Candidate


@dataclass(frozen=True)
class Outcome:
    """Result of one candidate on one task.

    `reward` is the mean over the measured rollouts, or None when none was measured, and
    `count` is the number of rollouts it averages. `feedback` is the grader's output, `inputs`
    the task's instruction, and `transcript` the tail of the worst rollout's transcript."""

    task: str
    reward: float | None
    feedback: str = ""
    inputs: str = ""
    count: int = 1
    transcript: str = ""

    @property
    def measured(self) -> bool:
        return self.reward is not None


class Scorer(Protocol):
    """Callable that rolls out a candidate on tasks in round `round_index` (0 for the seeds)
    and returns one outcome per task. A task may have no outcome if it produced nothing; its
    rollouts still count as spent."""

    async def __call__(
        self, candidate: Candidate, tasks: Sequence[Path], round_index: int
    ) -> Sequence[Outcome]: ...


def mean(outcomes: Iterable[Outcome]) -> float | None:
    """Return the mean reward over the measured outcomes, or None when there are none."""
    rewards = [outcome.reward for outcome in outcomes if outcome.reward is not None]
    return sum(rewards) / len(rewards) if rewards else None


@dataclass(frozen=True)
class Fitness:
    """Rewards per task for each candidate digest. Candidates with the same text share a row."""

    scores: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def tasks(self) -> list[str]:
        """All tasks that any row holds, in order of first appearance."""
        seen: dict[str, None] = {}
        for row in self.scores.values():
            for task in row:
                seen.setdefault(task, None)
        return list(seen)

    def fronts(self) -> dict[str, set[str]]:
        """Return, for each task, the digests holding its best score, ties included."""
        found: dict[str, set[str]] = {}
        for task in self.tasks:
            column = {digest: row[task] for digest, row in self.scores.items() if task in row}
            top = max(column.values())
            found[task] = {digest for digest, value in column.items() if value >= top}
        return found

    def leads(self) -> dict[str, int]:
        """Apply GEPA's Pareto selection and return each kept digest with the number of tasks
        it leads, which is its weight when a parent is drawn.

        Candidates that are best on at least one task are kept, minus dominated ones: those
        whose every led task is also led by another kept candidate, removed weakest first."""
        fronts = list(self.fronts().values())
        kept = {digest for front in fronts for digest in front}
        for digest in sorted(kept, key=lambda each: (self.aggregate(each) or 0.0, each)):
            others = kept - {digest}
            if others and all(front & others for front in fronts if digest in front):
                kept = others
        return {digest: sum(digest in front for front in fronts) for digest in sorted(kept)}

    def frontier(self) -> set[str]:
        """Return the Pareto frontier: the digests `leads` keeps."""
        return set(self.leads())

    def aggregate(self, digest: str, tasks: Sequence[str] | None = None) -> float | None:
        """Return the mean of a row, over `tasks` when given, or None when it holds none of them."""
        row = self.scores.get(digest) or {}
        wanted = [task for task in (row if tasks is None else tasks) if task in row]
        if not wanted:
            return None
        return sum(row[task] for task in wanted) / len(wanted)

    def coverage(self, digest: str) -> int:
        """Return the number of tasks the row holds a measurement for. A mean over one task
        and a mean over ten are not comparable, and a thin row is usually thin because its
        trials failed."""
        return len(self.scores.get(digest) or {})


def tally(pool: Iterable[str], seen: Mapping[str, Mapping[str, Outcome]]) -> Fitness:
    """Build the pool's fitness from the measured outcomes: a row per member, a cell per
    measured outcome. A child rejected on its minibatch is not in the pool, so it leaves no
    partial row that could win tasks it was never compared on."""
    scores: dict[str, dict[str, float]] = {}
    for digest in pool:
        row = scores.setdefault(digest, {})
        for task, outcome in seen.get(digest, {}).items():
            if outcome.reward is not None:
                row[task] = float(outcome.reward)
    return Fitness(scores)
