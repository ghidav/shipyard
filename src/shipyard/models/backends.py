from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from collections.abc import Sequence as Values
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import tinker
import torch
from tinker_cookbook.rl.metrics import compute_kl_sample_train

from shipyard.groups import Sequence
from shipyard.models.clients import CLIENTS, Client, client_for
from shipyard.models.clients.tinker import datum
from shipyard.models.endpoints import LaneEndpoint, ProviderEndpoint, ProxyEndpoint
from shipyard.run.costs import Spend

if TYPE_CHECKING:
    from shipyard.config import Config, ModelTable
    from shipyard.model import Model

SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class Trained:
    """What one `Session.fit` read, and how far the trained logprobs drifted from the ones
    the ratio was taken against."""

    tokens: int
    kl_v1: float
    kl_v2: float
    entropy: float


@dataclass(frozen=True)
class Saved:
    """A checkpoint `Session.save` made: its training state, its weights for sampling if kept,
    and how long both live, None for good."""

    training_state: str
    sampler_weights: str | None
    ttl_hours: float | None


class API:
    """A provider's model, which a harness calls with its own credentials when `rollout` runs
    it. It gives no logprobs."""

    def __init__(self, model: str, provider: str, spend: Spend) -> None:
        self.model, self.provider, self.spend = model, provider, spend
        #: What a harness calls to reach it.
        self.endpoint = ProviderEndpoint(model, provider)
        #: The provider's limits apply, not a budget of the run's.
        self.budget: int | None = None

    async def logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
        raise RuntimeError(f"{self.provider}'s model gives no logprobs of a sequence")


class Session:
    """A model on Tinker or Together: its client, which trains it when the recipe does, and its
    lane in the run's proxy, which opens its trials on the weights it serves. A model the recipe
    does not train is served and scored as it is, as its client opens it."""

    def __init__(
        self,
        table: ModelTable,
        run: str,
        spend: Spend,
        lane: LaneEndpoint,
        *,
        ttl_hours: float,
        trains: bool = True,
        training_state: str | None = None,
    ) -> None:
        self.client: Client = client_for(table, run, trains=trains)
        #: Its lane in the run's proxy, which its harnesses call.
        self.endpoint = lane
        self.spend = spend
        #: Whether the recipe trains the model, so its client opens a training client.
        self.trains = trains
        #: Hours a checkpoint not kept for good lives.
        self.ttl_hours = ttl_hours
        #: The training state the session opens on, from the checkpoint a run resumes from;
        #: None opens the model's own weights.
        self.training_state = training_state

    @property
    def provider(self) -> str:
        """Who its tokens are billed to, as its client was made with."""
        return self.client.provider

    @property
    def kind(self) -> str:
        """The backend it reaches, as a job's row names it."""
        return self.client.kind

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def budget(self) -> int | None:
        """Most tokens one trial may sample, as the proxy reports it once open."""
        return self.endpoint.budget

    async def __aenter__(self) -> Session:
        """Open the client and a resumed state, in order, and give its lane what the proxy
        samples it through; the proxy starts once every session has opened."""
        try:
            await self.client.open()
            if self.training_state is not None:
                await self.client.load_state(self.training_state)
            self.endpoint.backend = await self.client.serving()
        except BaseException as failed:
            await self.__aexit__(type(failed), failed, failed.__traceback__)
            raise
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        failed: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Finish the session with how the run ended."""
        await self.client.close(*ended(failed))

    async def logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
        """Return the model's logprob at each target of each sequence, as its client scores
        it, and file the tokens that took."""
        found, tokens = await self.client.compute_logprobs(sequences)
        self.count(**tokens)
        return found

    async def fit(
        self,
        parts: Values[Values[Sequence]],
        *,
        logprobs: Mapping[Sequence, Values[float]],
        adam: tinker.AdamParams,
        rate: float,
        loss: str,
        loss_config: Mapping[str, float] | None,
    ) -> Trained:
        """Train the model on each part in turn, a pass and an optimizer step each: every
        sequence at its advantages, its ratio taken against its `logprobs`."""
        datums = [
            [
                datum(
                    one,
                    logprobs=logprobs[one],
                    advantages=one.advantages.tolist(),  # type: ignore[union-attr]
                )
                for one in part
            ]
            for part in parts
        ]
        flat = [one for part in datums for one in part]
        trained = await self.client.fit(datums, adam, rate, loss, loss_config)
        tokens = sum(int(one.model_input.length) for one in flat)
        self.count(train=tokens)
        drift = compute_kl_sample_train(flat, [torch.tensor(row) for row in trained])
        return Trained(
            tokens=tokens,
            kl_v1=drift["optim/kl_sample_train_v1"],
            kl_v2=drift["optim/kl_sample_train_v2"],
            entropy=drift["optim/entropy"],
        )

    async def serve(self, tag: str) -> str | None:
        """Publish the model's weights as `tag`, open its trials from now on at them, and return
        their path: None for the base model, served as it is. Raise `RuntimeError` before
        publishing on Together while one of its trials is open, which the new weights would
        answer."""
        self.endpoint.publishing()
        path = await self.client.publish(tag)
        self.endpoint.serve(path)
        return path

    async def save(self, tag: str, *, keep: bool) -> Saved:
        """Save the model's training state as `tag`, for `[checkpoints] ttl_hours` if its client
        lets what it saves expire; with `keep`, its weights for sampling too, and both for good."""
        hours = None if keep or not self.client.expires else self.ttl_hours
        ttl = None if hours is None else int(hours * SECONDS_PER_HOUR)
        if not keep:
            return Saved(await self.client.save_state(tag, ttl_seconds=ttl), None, hours)
        training_state, sampler_weights = await asyncio.gather(
            self.client.save_state(tag, ttl_seconds=ttl),
            self.client.save_weights_for_sampler(tag, ttl_seconds=ttl),
        )
        return Saved(training_state, sampler_weights, hours)

    def count(self, **tokens: int) -> None:
        """File tokens the model spent, by kind, in the run's costs."""
        self.spend(self.provider, self.model, **tokens)


#: What a model runs on, once the run binds it.
Backend = Session | API


def ended(failed: BaseException | None) -> tuple[str, str | None]:
    """Return how the run ended, as a session is finished with: its status, and why."""
    if failed is None:
        return "success", None
    status = "errored" if isinstance(failed, Exception) else "interrupted"
    return status, f"{type(failed).__name__}: {failed}"


class Connected:
    """The backends of the run's models, which open when the run starts and close when it
    ends: each model on Tinker or Together in a session of its own, then the run's proxy, with a
    lane for each of them."""

    def __init__(self, sessions: Values[Session], proxy: ProxyEndpoint | None) -> None:
        #: Each model's session on Tinker or Together; none for a provider's.
        self.sessions = list(sessions)
        #: The run's proxy; None when no model is on Tinker or Together.
        self.proxy = proxy
        self._open: AsyncExitStack | None = None

    async def __aenter__(self) -> Connected:
        """Open the models the recipe does not train first, so a model the run cannot open
        refuses the run before a trained one's GPUs start; then the trained ones; then the
        proxy, which needs every lane's base model and session; then serve each model's
        starting weights. A failure closes what opened, the proxy first."""
        ordered = sorted(self.sessions, key=lambda one: one.trains)
        async with AsyncExitStack() as stack:
            for one in ordered:
                await stack.enter_async_context(one)
            if self.proxy is not None:
                await stack.enter_async_context(self.proxy)
            for one in ordered:
                await one.serve("start")
            self._open = stack.pop_all()
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        failed: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close every session, each finished with how the run ended."""
        stack, self._open = self._open, None
        if stack is not None:
            await stack.__aexit__(kind, failed, tb)


def connect(
    config: Config,
    directory: Path,
    models: Values[Model],
    spend: Callable[..., None],
    *,
    training_states: Mapping[str, str] | None = None,
) -> Connected:
    """Give each model its backend, by its provider, each filing its spend under its role, and
    each one the recipe trains opening on its training state in `training_states`, when a
    resume gives one. Raise `ValueError` for a model the run cannot train or open, before
    anything opens."""
    states = dict(training_states or {})
    for one in models:
        if one.training and not one.trainable:
            raise ValueError(
                f"[models.{one.role}]: the recipe trains {one.name}, and {one.provider} serves "
                "it, so no run can train it"
            )
    run = Path(directory).name
    served = any(one.provider in CLIENTS for one in models)
    proxy = ProxyEndpoint(config, directory) if served else None
    sessions: list[Session] = []
    for one in models:
        table = one.table
        if proxy is None or one.provider not in CLIENTS:
            one.bind(API(one.name, one.provider, partial(spend, one.role)))
            continue
        lane = proxy.lane(
            one.role,
            table.model,
            renderers=table.renderers,
            max_context=table.max_context or None,
            sampling=table.sampling,
        )
        session = Session(
            table,
            run,
            partial(spend, one.role),
            lane,
            ttl_hours=config.checkpoints.ttl_hours,
            trains=one.training,
            training_state=states.get(one.role) if one.training else None,
        )
        # The lane bills what it serves as the session does.
        lane.opened = session
        sessions.append(session)
        one.bind(session)
    return Connected(sessions, proxy)
