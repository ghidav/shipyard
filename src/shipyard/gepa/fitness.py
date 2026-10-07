"""What the search selects on: a candidate's fitness is a vector over tasks, never a
number, so the frontier can keep the candidate that alone solves one task. A rollout
nobody could measure is absent from every mean, not a zero in it."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from shipyard.modules import Candidate


@dataclass(frozen=True)
class Outcome:
    """One task under one candidate: its mean reward over the measured rollouts (None when
    none was), what the grader said, what the task asked, how many rollouts the mean
    averages, and the tail of the worst rollout's transcript."""

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
    """Rolls a candidate out over tasks in round `round_index` (0 for the seed) and says
    how it did, one outcome per task; fewer outcomes than tasks means those produced
    nothing, and they are spent all the same."""

    async def __call__(
        self, candidate: Candidate, tasks: Sequence[Path], round_index: int
    ) -> Sequence[Outcome]: ...


def mean(outcomes: Iterable[Outcome]) -> float | None:
    """The mean over the measured outcomes; None of nothing, since a 0 would be a score."""
    rewards = [outcome.reward for outcome in outcomes if outcome.reward is not None]
    return sum(rewards) / len(rewards) if rewards else None


@dataclass(frozen=True)
class Fitness:
    """A reward per candidate digest per task. By digest, so two candidates carrying the
    same text are one row."""

    scores: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def tasks(self) -> list[str]:
        """Every task any row holds, in order of first appearance."""
        seen: dict[str, None] = {}
        for row in self.scores.values():
            for task in row:
                seen.setdefault(task, None)
        return list(seen)

    def fronts(self) -> dict[str, set[str]]:
        """Per task, the candidates holding its best score, ties included."""
        found: dict[str, set[str]] = {}
        for task in self.tasks:
            column = {digest: row[task] for digest, row in self.scores.items() if task in row}
            top = max(column.values())
            found[task] = {digest for digest, value in column.items() if value >= top}
        return found

    def leads(self) -> dict[str, int]:
        """GEPA's Pareto selection: the candidates best on at least one task, less those
        dominated (every task they lead is led by another kept candidate, weakest removed
        first), each with how many tasks it leads, the weight a parent is drawn with."""
        fronts = list(self.fronts().values())
        kept = {digest for front in fronts for digest in front}
        for digest in sorted(kept, key=lambda each: (self.aggregate(each) or 0.0, each)):
            others = kept - {digest}
            if others and all(front & others for front in fronts if digest in front):
                kept = others
        return {digest: sum(digest in front for front in fronts) for digest in sorted(kept)}

    def frontier(self) -> set[str]:
        """The Pareto frontier: the candidates `leads` keeps."""
        return set(self.leads())

    def aggregate(self, digest: str, tasks: Sequence[str] | None = None) -> float | None:
        """The mean of a row, over `tasks` when given; None when it holds nothing of them."""
        row = self.scores.get(digest) or {}
        wanted = [task for task in (row if tasks is None else tasks) if task in row]
        if not wanted:
            return None
        return sum(row[task] for task in wanted) / len(wanted)

    def coverage(self, digest: str) -> int:
        """How many tasks the row holds a measurement for: a mean over one column and a
        mean over ten are means of different things, and the thin one is usually thin
        because its trials died."""
        return len(self.scores.get(digest) or {})


def tally(pool: Iterable[str], seen: Mapping[str, Mapping[str, Outcome]]) -> Fitness:
    """The fitness over the pool's digests from what was measured: a row per member, a
    cell per measured outcome. A child declined on its minibatch is not in the pool and
    leaves no partial row to win columns it was never compared on."""
    scores: dict[str, dict[str, float]] = {}
    for digest in pool:
        row = scores.setdefault(digest, {})
        for task, outcome in seen.get(digest, {}).items():
            if outcome.reward is not None:
                row[task] = float(outcome.reward)
    return Fitness(scores)
