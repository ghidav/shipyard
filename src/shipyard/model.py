from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence as Values
from typing import TYPE_CHECKING, Any

import torch

from shipyard.groups import Sequence
from shipyard.models.clients import CLIENTS
from shipyard.models.clients.tinker import SAMPLER_WEIGHTS, TINKER
from shipyard.models.context import insert

if TYPE_CHECKING:
    from shipyard.models.backends import Backend

__all__ = ["Model"]


class Model:
    """A model the run reaches: `rollout` runs it through a harness, and `logprobs` scores
    sequences with it. What it can do follows from what it is: `trainable` from its provider,
    `training` from `train=`, which a recipe may change until the run starts. Only a model the
    run trains moves, at each update; any other answers the same all run."""

    def __init__(self, name: str, *, provider: str = TINKER, train: bool = False) -> None:
        """Raise `ValueError` for no name. Whether a model with `train` can be trained is
        checked when the run starts, before anything opens."""
        if not name:
            raise ValueError("a model needs its name: a model's name, or a tinker:// path")
        #: What it loads: a model's name, or a `tinker://` path.
        self.name = name
        self._provider = provider
        self._training = bool(train)
        self._backend: Backend | None = None
        #: The model whose tokenizer places a text read first, and the context it must fit.
        self._reader = ""
        self._limit: Callable[[], int | None] = lambda: None

    def __repr__(self) -> str:
        at = "" if self._provider == TINKER else f", provider={self._provider!r}"
        return f"Model({self.name!r}{at})"

    @property
    def provider(self) -> str:
        """Who bills its tokens; once bound, its backend's."""
        known = self._backend.provider if self._backend is not None else None
        return known or self._provider

    @property
    def trainable(self) -> bool:
        """Whether a run could train it: a model Tinker or Together trains, and not weights
        saved for sampling. The run checks it with the backend when it starts."""
        return self._provider in CLIENTS and SAMPLER_WEIGHTS not in self.name

    @property
    def training(self) -> bool:
        """Whether the run trains it, as in PyTorch; set it before the run starts."""
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
        """Most tokens one rollout may sample, when the model is served to rollouts."""
        return self._backend.budget if self._backend is not None else None

    @property
    def endpoint(self) -> Any:
        """What a harness calls to reach the model; None for one the run does not serve, or
        before binding."""
        return self._backend.endpoint if self._backend is not None else None

    def bind(self, backend: Backend, *, reader: str, limit: Callable[[], int | None]) -> None:
        """Give the model its backend, and the model and context that place a `prefix`."""
        self._backend, self._reader, self._limit = backend, reader, limit

    async def logprobs(
        self, sequences: Values[Sequence], *, prefix: str | Values[str | None] | None = None
    ) -> list[torch.Tensor | None]:
        """Return the logprob of each target token of each sequence, reading `prefix` first."""
        backend = self._bound()
        read: list[Sequence | None] = list(sequences)
        if prefix is not None:
            texts = [prefix] * len(read) if isinstance(prefix, str) else list(prefix)
            read = insert(sequences, texts, self._reader, self._limit())
        asked = [one for one in read if one is not None]
        scored = iter(await backend.score(asked) if asked else [])
        return [None if one is None else torch.tensor(next(scored)) for one in read]

    def _bound(self) -> Backend:
        if self._backend is None:
            raise RuntimeError(
                "this model is not connected yet: the run connects the models the trainer "
                "holds when it starts, so keep it on the trainer, as self.<name>"
            )
        return self._backend
