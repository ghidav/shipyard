from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from collections.abc import Sequence as Values
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import tinker
import torch
from tinker_cookbook.rl.metrics import compute_kl_sample_train

from shipyard.groups import Sequence
from shipyard.models import context
from shipyard.models.clients import CLIENTS, Client, client_for
from shipyard.models.clients.tinker import TINKER, datum, finished, sampled, sampling
from shipyard.models.clients.together import TOGETHER
from shipyard.models.endpoints import LaneEndpoint, ProviderEndpoint, ProxyEndpoint
from shipyard.run.costs import UNKNOWN, Spend

if TYPE_CHECKING:
    from shipyard.config import Blueprint
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

    state: str
    sampler: str | None
    ttl_hours: float | None


class Sampling:
    """A model on Tinker the run does not train, beside its agent, scored through a sampling
    client: a model's name, or a `tinker://` path. It opens a Tinker session of its own."""

    #: The backend it reaches, as a job's row names it.
    kind = TINKER

    def __init__(
        self,
        load: str,
        spend: Spend,
        *,
        role: str = "",
        run: str | None = None,
        reader: str | None = None,
    ) -> None:
        #: What it loads: a model's name, or a `tinker://` path.
        self.load = load
        #: The model its tokens are priced as; for a path, its base model, known once open.
        self.model: str | None = None if load.startswith("tinker://") else load
        #: Who its tokens are billed to, known once its session opens.
        self.provider: str | None = None
        self.spend = spend
        #: The role it is held by, which its errors name, and the run its session is tagged by.
        self.role, self.run = role, run
        #: The agent's model, whose token ids it scores; None when the agent writes none.
        self.reader = reader
        self.sampler: Any = None
        self.service: Any = None
        #: Its lane in the run's proxy, which a harness dials when `rollout` runs it.
        self.endpoint: LaneEndpoint | None = None

    @property
    def budget(self) -> int | None:
        """Most tokens one rollout may sample, as the proxy reports it for its lane."""
        return self.endpoint.budget if self.endpoint is not None else None

    async def __aenter__(self) -> Sampling:
        """Open a Tinker session of its own, and a sampling client in it. Raise `ValueError`,
        the session finished, for a model the backend does not serve, or one that reads other
        token ids than the agent's."""
        named = f"{self.role} = Model({self.load!r})"
        try:
            self.service, self.sampler, model, self.provider = await sampling(
                self.load, run=self.run
            )
        except ValueError as refused:
            raise ValueError(f"{named}: {refused}") from None
        self.model = model
        if self.reader is not None and not context.shares_tokenizer(model, self.reader):
            refused = ValueError(
                f"{named}: {model} reads other token ids than the agent, {self.reader}, so its "
                "logprobs would score other tokens"
            )
            await self.__aexit__(ValueError, refused, None)
            raise refused
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        failed: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Finish its own session, if it opened one, with how the run ended."""
        service, self.service = self.service, None
        if service is not None:
            await finished(service, *ended(failed))

    async def score(self, sequences: Values[Sequence]) -> list[list[float]]:
        """Return the logprob at each target of each sequence, through the sampling client."""
        if self.sampler is None:
            raise RuntimeError("this model cannot score yet: it opens when the run starts")
        found, tokens = await sampled(self.sampler, sequences)
        self.spend(self.provider or UNKNOWN, self.model or UNKNOWN, **tokens)
        return found


class Provider:
    """A provider's model, which a harness calls with its own credentials when `rollout` runs
    it. It cannot score."""

    def __init__(self, model: str, provider: str, spend: Spend) -> None:
        self.model, self.provider, self.spend = model, provider, spend
        #: What a harness calls to reach it.
        self.endpoint = ProviderEndpoint(model, provider)
        #: The provider's limits apply, not a budget of the run's.
        self.budget: int | None = None

    async def score(self, sequences: Values[Sequence]) -> list[list[float]]:
        raise RuntimeError(
            f"this model cannot score: {self.provider}'s model gives no logprobs of a sequence"
        )


class Session:
    """The agent on Tinker or Together: its client, which trains it when the run does, and its
    lane in the run's proxy, which its harnesses call and its new weights are served in. An
    agent the run does not train is served and scored as it is, as its client opens it."""

    def __init__(
        self,
        config: Blueprint,
        directory: Path,
        spend: Spend,
        lane: LaneEndpoint,
        *,
        trains: bool = True,
        weights: str | None = None,
    ) -> None:
        self.client: Client = client_for(config, directory, trains=trains)
        #: Its lane in the run's proxy, which its harnesses call.
        self.endpoint = lane
        self.spend = spend
        #: Whether the run trains the agent, so its client opens a training client.
        self.trains = trains
        #: Hours a checkpoint not kept for good lives.
        self.ttl_hours = config.checkpoints.ttl_hours
        #: The training state the session opens on, from the checkpoint a run resumes from;
        #: None opens the initial weights.
        self.weights = weights

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
        """Most tokens one rollout may sample, as the proxy reports it once open."""
        return self.endpoint.budget

    async def __aenter__(self) -> Session:
        """Open the client and a resumed state, in order, and give its lane what the proxy
        samples it through; the proxy starts once every session has opened."""
        try:
            await self.client.open()
            if self.weights is not None:
                await self.client.load_state(self.weights)
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

    async def score(self, sequences: Values[Sequence]) -> list[list[float]]:
        """Return the agent's logprob at each target of each sequence, as its client scores
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
        """Train the agent on each part in turn, a pass and an optimizer step each: every
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
        """Publish the agent's weights as `tag`, point the proxy at them, and return their path:
        None for the base model, served as it is."""
        path = await self.client.publish(tag)
        await self.endpoint.serve(path)
        return path

    async def save(self, tag: str, *, keep: bool) -> Saved:
        """Save the agent's training state as `tag`, for `[checkpoints] ttl_hours` if its client
        lets what it saves expire; with `keep`, its weights for sampling too, and both for good."""
        hours = None if keep or not self.client.expires else self.ttl_hours
        ttl = None if hours is None else int(hours * SECONDS_PER_HOUR)
        if not keep:
            return Saved(await self.client.save_state(tag, ttl_seconds=ttl), None, hours)
        state, weights = await asyncio.gather(
            self.client.save_state(tag, ttl_seconds=ttl),
            self.client.save_weights_for_sampler(tag, ttl_seconds=ttl),
        )
        return Saved(state, weights, hours)

    def count(self, **tokens: int) -> None:
        """File tokens the agent spent, by kind, in the run's costs."""
        self.spend(self.provider, self.model, **tokens)


#: What a model runs on, once the run binds it.
Backend = Session | Sampling | Provider


def ended(failed: BaseException | None) -> tuple[str, str | None]:
    """Return how the run ended, as a session is finished with: its status, and why."""
    if failed is None:
        return "success", None
    status = "errored" if isinstance(failed, Exception) else "interrupted"
    return status, f"{type(failed).__name__}: {failed}"


class Connected:
    """The backends of the models a trainer holds, which open when the run starts and close
    when it ends: each model on Tinker in a session of its own, then the agent's session, then
    the run's proxy, with a lane for each model on Tinker or Together."""

    def __init__(
        self, session: Session | None, others: Values[Sampling], proxy: ProxyEndpoint | None
    ) -> None:
        #: The agent's session; None for a provider's agent.
        self.session = session
        self.others = list(others)
        #: The run's proxy; None when no model is on Tinker or Together.
        self.proxy = proxy
        self._open: AsyncExitStack | None = None

    async def __aenter__(self) -> Connected:
        """Open the other models first, so a model the run cannot score with refuses the run
        before the agent's GPUs start; then the agent's session, then the proxy, which needs
        every lane's base model and session; then serve the agent's starting weights. A
        failure closes what opened, the proxy first."""
        async with AsyncExitStack() as stack:
            for one in self.others:
                await stack.enter_async_context(one)
            if self.session is not None:
                await stack.enter_async_context(self.session)
            if self.proxy is not None:
                await stack.enter_async_context(self.proxy)
            if self.session is not None:
                await self.session.serve("start")
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
    config: Blueprint,
    directory: Path,
    held: Values[tuple[str, Model]],
    spend: Callable[..., None],
    *,
    weights: str | None = None,
) -> Connected:
    """Give each model the trainer holds its backend, by its provider, the agent first, each
    filing its spend under its role. Raise `ValueError` for a model the run cannot train or
    open, before anything opens."""
    (role, agent), *others = held
    given = agent.provider
    if agent.training and not agent.trainable:
        why = f"{given} serves it" if given not in CLIENTS else "it is weights saved for sampling"
        raise ValueError(
            f"the agent, {agent!r}, is training, and {why}, so no run can train it: a recipe "
            "that never updates sets `self.policy.training = False`"
        )
    for named, one in others:
        if one.training:
            raise ValueError(
                f"{named} = {one!r} is training, and a run trains one model, its agent, for now"
            )
        if one.provider == TOGETHER:
            raise ValueError(f"{named} = {one!r}: a model on Together opens as the agent alone")
    run = Path(directory).name
    # The agent's token ids, which a model on Tinker scores; a provider's agent writes none.
    reader = config.agent.model if given in CLIENTS else None
    served = given in CLIENTS or any(one.provider == TINKER for _, one in others)
    proxy = ProxyEndpoint(config, directory) if served else None

    def budget() -> int | None:
        return agent.budget

    session: Session | None = None
    if proxy is not None and given in CLIENTS:
        # The agent's lane: its renderers, and `[rollout] max_context`, are its own.
        rollout = config.rollout
        lane = proxy.lane(
            role,
            config.agent.model,
            renderers=rollout.renderer,
            max_context=rollout.max_context or None,
        )
        session = Session(
            config, directory, partial(spend, role), lane, trains=agent.training, weights=weights
        )
        # The lane bills what it serves as the session does.
        lane.opened = session
        agent.bind(session, reader=config.agent.model, limit=budget)
    else:
        agent.bind(
            Provider(agent.name, given, partial(spend, role)),
            reader=config.agent.model,
            limit=budget,
        )
    hosted: list[Sampling] = []
    for role, one in others:
        backend: Sampling | Provider
        if one.provider == TINKER:
            backend = Sampling(one.name, partial(spend, role), role=role, run=run, reader=reader)
            if proxy is not None:
                backend.endpoint = proxy.lane(lane_of(role), one.name, opened=backend)
            hosted.append(backend)
        else:
            backend = Provider(one.name, one.provider, partial(spend, role))
        one.bind(backend, reader=config.agent.model, limit=budget)
    return Connected(session, hosted, proxy)


def lane_of(role: str) -> str:
    """Return the name of a model's lane in the proxy, from its role: the role, with what an
    address cannot hold, such as `teachers[0]`'s brackets, as dashes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", role).strip("-") or "model"
