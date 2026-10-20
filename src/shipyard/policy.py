from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence as Values
from typing import Any

import torch

from shipyard.groups import Sequence
from shipyard.models.agent import INITIAL, Frozen, Provider, Session
from shipyard.models.clients.tinker import TINKER
from shipyard.models.context import insert

__all__ = ["INITIAL", "Policy"]

#: What a policy runs on, once the run binds it.
Backend = Session | Frozen | Provider


class Policy:
    """A model at some weights, which `sample`s text and scores sequences with `logprobs`."""

    def __init__(self, model: str, *, provider: str = TINKER) -> None:
        """Raise `ValueError` for no model, or for `INITIAL` at a provider."""
        if not model:
            raise ValueError("a policy needs its model: a name, a tinker:// path, or INITIAL")
        if model == INITIAL and provider != TINKER:
            raise ValueError(f"INITIAL is the agent's weights on Tinker, not a model at {provider}")
        #: What it loads: a model's name, a `tinker://` checkpoint, or `INITIAL`.
        self.load = model
        self._provider = provider
        self._model: str | None = (
            None if model == INITIAL or model.startswith("tinker://") else model
        )
        self._backend: Backend | None = None
        #: The model whose tokenizer places a text read first, and the context it must fit.
        self._reader = ""
        self._limit: Callable[[], int | None] = lambda: None

    @property
    def provider(self) -> str:
        """Who bills its tokens; once bound, its backend's."""
        known = self._backend.provider if self._backend is not None else None
        return known or self._provider

    @property
    def model(self) -> str | None:
        """The model its tokens are priced as: for a checkpoint or `INITIAL`, its base model."""
        known = self._backend.model if self._backend is not None else None
        return known or self._model

    @property
    def trains(self) -> bool:
        """Whether the run trains this policy: the agent, when Tinker serves it."""
        return isinstance(self._backend, Session)

    @property
    def budget(self) -> int | None:
        """Most tokens one rollout may sample, when the policy is served to rollouts."""
        return self._backend.budget if self._backend is not None else None

    @property
    def endpoint(self) -> Any:
        """What a harness calls to reach the policy; None for a frozen one, or before binding."""
        return self._backend.endpoint if self._backend is not None else None

    def bind(self, backend: Backend, *, reader: str, limit: Callable[[], int | None]) -> None:
        """Give the policy its backend, and the model and context that place a `prefix`."""
        self._backend, self._reader, self._limit = backend, reader, limit

    async def sample(self, prompts: Values[str], **params: Any) -> list[str]:
        """Return the model's reply to each prompt; `params` go to the provider as they are."""
        return await self._bound().sample(prompts, **params)

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
                "this policy is not connected yet: the run connects the policies the trainer "
                "holds when `train` starts, so keep it on the trainer, as self.<name>"
            )
        return self._backend
