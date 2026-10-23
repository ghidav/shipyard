from __future__ import annotations

from collections.abc import Iterator, Mapping
from collections.abc import Sequence as Values
from typing import TYPE_CHECKING, Any

import torch

from shipyard.groups import Sequence
from shipyard.models.clients import CLIENTS, TOGETHER

if TYPE_CHECKING:
    from shipyard.config import ModelTable
    from shipyard.models.backends import Backend

__all__ = ["Model", "Models"]


class Model:
    """A model the run reaches, as its `[models.<role>]` table names it: `rollout` runs it through
    a harness, `logprobs` scores sequences with it, and `update` trains it when the recipe marks
    it `training`. Only a model the run trains moves, at each update; any other answers the same
    all run."""

    def __init__(self, role: str, table: ModelTable) -> None:
        """Raise `ValueError` for renderers of a model a provider serves to its harnesses, and
        for sampler replicas of a model off Together."""
        if table.renderers and table.provider not in CLIENTS:
            raise ValueError(
                f"[models.{role}] renderers: the proxy renders the calls of a model on Tinker or "
                f"Together, and {table.provider} serves {table.model} to its harnesses itself"
            )
        if table.sampler_replicas != 1 and table.provider != TOGETHER:
            raise ValueError(
                f"[models.{role}] sampler_replicas: the run sets how many replicas the sampler of "
                f"a model on Together has, and {table.provider} runs {table.model}'s itself"
            )
        #: The name run.toml gives it, which its lane, spend and saves are filed under.
        self.role = role
        #: Its `[models.<role>]` table.
        self.table = table
        self._training = False
        self._backend: Backend | None = None

    def __repr__(self) -> str:
        return f"Model({self.role}: {self.name})"

    @property
    def name(self) -> str:
        """What it loads: a model's name."""
        return self.table.model

    @property
    def provider(self) -> str:
        """Who bills its tokens; once bound, its backend's."""
        known = self._backend.provider if self._backend is not None else None
        return known or self.table.provider

    @property
    def renderers(self) -> list[str]:
        """The renderers its lane holds, one of which a rollout of it names."""
        return list(self.table.renderers)

    @property
    def trainable(self) -> bool:
        """Whether a run could train it: a model Tinker or Together trains."""
        return self.table.provider in CLIENTS

    @property
    def training(self) -> bool:
        """Whether the run trains it, as in PyTorch; the recipe sets it in its `__init__`."""
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        if self._backend is not None:
            raise RuntimeError(
                f"{self!r} is already open: set `training` in the trainer's __init__, before "
                "the run starts"
            )
        self._training = bool(value)

    @property
    def budget(self) -> int | None:
        """Most tokens one trial may sample, when the model is served to rollouts."""
        return self._backend.budget if self._backend is not None else None

    @property
    def endpoint(self) -> Any:
        """What a harness calls to reach the model; None before the run binds it."""
        return self._backend.endpoint if self._backend is not None else None

    @property
    def backend(self) -> Backend | None:
        """What it runs on: its session, which trains and saves it when the recipe trains it,
        or its provider's API; None before the run binds it."""
        return self._backend

    def bind(self, backend: Backend) -> None:
        """Give the model its backend."""
        self._backend = backend

    async def logprobs(self, sequences: Values[Sequence]) -> list[torch.Tensor]:
        """Return the logprob of each target token of each sequence, as the model reads it."""
        if self._backend is None:
            raise RuntimeError(
                "this model is not connected yet: the run connects the models when it starts"
            )
        if not sequences:
            return []
        return [torch.tensor(one) for one in await self._backend.logprobs(sequences)]


class Missing(KeyError):
    """Raised for a role run.toml names no model under."""

    def __str__(self) -> str:
        return str(self.args[0])


class Models(Mapping[str, Model]):
    """The run's models by role, as run.toml's `[models.<role>]` tables name them. A recipe reads
    the ones it uses in its `__init__`; the run refuses one it never read."""

    def __init__(self, tables: Mapping[str, ModelTable]) -> None:
        self._held = {role: Model(role, table) for role, table in tables.items()}
        #: The roles the recipe read.
        self.read: set[str] = set()

    def __getitem__(self, role: str) -> Model:
        """Return the model `role` names; `Missing` when run.toml names none."""
        found = self._held.get(role)
        if found is None:
            named = ", ".join(sorted(self._held))
            raise Missing(
                f"run.toml has no [models.{role}], which the recipe reads: it names {named}"
            )
        self.read.add(role)
        return found

    def __contains__(self, role: object) -> bool:
        return role in self._held

    def __iter__(self) -> Iterator[str]:
        return iter(self._held)

    def __len__(self) -> int:
        return len(self._held)

    @property
    def unread(self) -> list[str]:
        """The roles the recipe never read."""
        return sorted(set(self._held) - self.read)

    @property
    def all(self) -> list[Model]:
        """Every model, in run.toml's order, without counting it read."""
        return list(self._held.values())
