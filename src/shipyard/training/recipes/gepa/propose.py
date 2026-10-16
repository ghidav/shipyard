"""The proposer interface: what a proposer is shown about one component and what it returns.

The traces drive the search: a score says a text did worse, the traces say why. A
reflector's findings survive only if it writes them into the text."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from shipyard.modules import Candidate, Module
from shipyard.training.recipes.gepa.fitness import Outcome


@dataclass(frozen=True)
class Reflection:
    """Input to a proposer: the candidate, the component to rewrite, and the candidate's
    outcomes on the round's tasks.

    The outcomes must be this candidate's own; traces of another text would have the
    reflector fix failures this text did not have."""

    candidate: Candidate
    component: str
    outcomes: tuple[Outcome, ...] = ()

    @property
    def module(self) -> Module:
        return self.candidate.components[self.component]


class Proposer(Protocol):
    """Callable that turns a reflection into a new module for the component, or returns None
    to decline (the traces call for no change, the model refused, or the trial failed). The
    search treats an exception as a decline."""

    async def __call__(self, reflection: Reflection) -> Module | None: ...


def component_for(turn: int, names: Sequence[str]) -> str:
    """Return the component that the `turn`-th rewrite changes, rotating round-robin through
    `names`. A round skipped for a perfect parent asks for no rewrite and takes no turn."""
    return names[turn % len(names)]


async def propose(
    parent: Candidate, component: str, outcomes: Iterable[Outcome], write: Proposer
) -> Candidate | None:
    """Ask `write` to rewrite `component` of `parent` and return the resulting child.

    Return None when the parent lacks the component, the proposer declines, or the result is
    blank or has the same files. A blank result counts as a decline, since it cannot be told
    apart from a deletion here."""
    module = parent.components.get(component)
    if module is None:
        return None
    rewritten = await write(Reflection(parent, component, tuple(outcomes)))
    if rewritten is None or not rewritten.text.strip() or rewritten.files == module.files:
        return None
    return parent.with_component(component, rewritten)
