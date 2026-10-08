"""Asking for a rewrite: what a proposer is shown about one component and what it returns.
The traces drive the search: a score says a text was worse, the traces say why. A
reflector's findings survive only if it writes them into the text."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from shipyard.gepa.fitness import Outcome
from shipyard.modules import Candidate, Module


@dataclass(frozen=True)
class Reflection:
    """What the proposer is shown: the candidate, the component to rewrite, and how that
    candidate did on the round's tasks. The outcomes must be the candidate's own: traces of
    another text would have the reflector fix what this text did not do."""

    candidate: Candidate
    component: str
    outcomes: tuple[Outcome, ...] = ()

    @property
    def module(self) -> Module:
        return self.candidate.components[self.component]


class Proposer(Protocol):
    """Turns a reflection into the component's new module, or None to decline (the traces
    call for no change, the model refused, or the trial failed). The search counts a
    decline, and an exception counts as one."""

    async def __call__(self, reflection: Reflection) -> Module | None: ...


def component_for(turn: int, names: Sequence[str]) -> str:
    """The component the `turn`-th rewrite changes. Components rotate round-robin, so each
    gets its turn. A round skipped for a perfect parent asks for no rewrite and takes no
    turn."""
    return names[turn % len(names)]


async def propose(
    parent: Candidate, component: str, outcomes: Iterable[Outcome], write: Proposer
) -> Candidate | None:
    """One rewrite of `component`, as the child it makes, or None when the parent lacks the
    component, the proposer declined, or the result is blank or has the same files. A blank
    result counts as a decline; it cannot be told from a deletion here."""
    module = parent.components.get(component)
    if module is None:
        return None
    rewritten = await write(Reflection(parent, component, tuple(outcomes)))
    if rewritten is None or not rewritten.text.strip() or rewritten.files == module.files:
        return None
    return parent.with_component(component, rewritten)
