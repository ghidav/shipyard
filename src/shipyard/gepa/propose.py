"""Asking for a rewrite: what a proposer is shown about one component, and what it hands
back. What makes the search work is the feedback: a number says a text was worse, the
traces say why, and nothing a reflector learns survives unless it writes it into the text."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from shipyard.gepa.fitness import Outcome
from shipyard.modules import Candidate, Module


@dataclass(frozen=True)
class Reflection:
    """What the proposer is shown: the candidate, which of its components to rewrite, and
    how that candidate did on the tasks the round is about. Its own outcomes, not another
    text's: a reflector shown someone else's traces is asked to fix what it did not do."""

    candidate: Candidate
    component: str
    outcomes: tuple[Outcome, ...] = ()

    @property
    def module(self) -> Module:
        return self.candidate.components[self.component]


class Proposer(Protocol):
    """Turns a reflection into the component's new module, or None to decline: nothing in
    the traces asked for a change, the model refused, the trial failed. A decline is an
    answer the search counts; a raise is read as one too."""

    async def __call__(self, reflection: Reflection) -> Module | None: ...


def component_for(turn: int, names: Sequence[str]) -> str:
    """The component the `turn`-th rewrite asked for changes: round-robin, so every one is
    attended to in turn rather than whichever was rewritten first and so has evidence. A
    round skipped for a perfect parent asks for none and takes no turn."""
    return names[turn % len(names)]


async def propose(
    parent: Candidate, component: str, outcomes: Iterable[Outcome], write: Proposer
) -> Candidate | None:
    """One rewrite of `component`, as the child it makes, or None: the parent lacks the
    component, the proposer declined, or what came back was blank or the same files. A
    blank is a decline, not a deletion, since the two cannot be told apart here."""
    module = parent.components.get(component)
    if module is None:
        return None
    rewritten = await write(Reflection(parent, component, tuple(outcomes)))
    if rewritten is None or not rewritten.text.strip() or rewritten.files == module.files:
        return None
    return parent.with_component(component, rewritten)
