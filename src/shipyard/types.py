"""The vocabulary of a run. Nothing here does work."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Iterator, Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pvariance
from typing import Any

import tinker

from shipyard.modules import Candidate
from shipyard.proxy.profiles import Profile, bare_name, profile_for
from shipyard.proxy.wire import Picture, Record

__all__ = [
    "Adam",
    "Agent",
    "Batches",
    "Connection",
    "Datum",
    "Endpoint",
    "Evaluation",
    "Forward",
    "Group",
    "Harness",
    "Member",
    "Policy",
    "Records",
    "RolloutEngine",
    "SERVE_TTL",
    "Sequence",
    "Settings",
    "Task",
    "TrainingClient",
    "Trial",
    "Update",
    "Verdict",
    "harness",
]

Task = Path
Datum = tinker.Datum
Batches = Iterator[list[Task]]
#: A job's model calls by trial name.
Records = dict[str, list[Record]]
#: How long served sampler weights are kept: past the batch sampled from them, and a
#: crashed run's leftovers expire on their own.
SERVE_TTL = 12 * 3600


# ------------------------------------------------------------------ the model


@dataclass(frozen=True)
class Sequence:
    """One chain: the last record's prompt and completion, which hold every earlier one."""

    tokens: tuple[int, ...]
    #: Where each completion sits, in chain order.
    spans: tuple[tuple[int, int], ...]
    #: The sampler's logprobs at the targets; None when a record carried none.
    mu: tuple[float, ...] | None
    images: tuple[Picture, ...] = ()

    @property
    def targets(self) -> tuple[int, ...]:
        return tuple(at for start, end in self.spans for at in range(start, end))


@dataclass(frozen=True)
class Forward:
    """What `forward_backward` returns: each datum's logprobs at every input position."""

    logprobs: list[list[float]]


@dataclass(frozen=True)
class Adam:
    beta1: float
    beta2: float
    eps: float
    weight_decay: float = 0.0
    #: Tinker's `grad_clip_norm`; 0 is none.
    grad_clip_norm: float = 0.0
    #: Updates over which the learning rate rises linearly to its value; 0 is none.
    warmup: int = 0

    def rate(self, learning_rate: float, done: int) -> float:
        if done >= self.warmup:
            return learning_rate
        return learning_rate * (done + 1) / self.warmup


@dataclass(frozen=True)
class Update:
    train_tokens: int
    substeps: int
    learning_rate: float
    loss_fn: str
    kl_v1: float | None
    kl_v2: float | None
    entropy: float | None
    seconds: float
    #: Sequences the step trained on.
    sequences: int = 0
    #: Mean of mu less the anchor's logprobs over those; None without a KL term.
    anchor_kl: float | None = None


class TrainingClient(ABC):
    """Tinker's training client; a step call submits on await and returns the result's awaitable."""

    @abstractmethod
    async def forward_backward(
        self, datums: Values[Datum], loss_fn: str, loss_config: Mapping[str, float] | None
    ) -> Awaitable[Forward]: ...

    @abstractmethod
    async def optim_step(self, adam: Adam, rate: float) -> Awaitable[None]: ...

    @abstractmethod
    async def compute_logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
        """mu at every target of every sequence, from one batched forward."""

    @abstractmethod
    async def save_weights_for_sampler(self, name: str, *, ttl_seconds: int | None) -> str: ...

    @abstractmethod
    async def save_state(self, name: str, *, ttl_seconds: int | None) -> str: ...

    @abstractmethod
    async def load_state(self, path: str) -> None: ...

    @abstractmethod
    async def close(self, status: str, detail: str | None = None) -> None: ...


@dataclass(frozen=True)
class Connection:
    """What a trial's harness dials; a provider's is the model name alone."""

    model: str | None
    address: str | None = None
    token: str | None = field(default=None, repr=False)
    env: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: The proxy's host, for a task on an allowlist.
    host: str | None = None


class Endpoint(ABC):
    """The model as the endpoint a harness dials: served by this run, or by a provider."""

    #: Sampled tokens one rollout may spend; None for a provider.
    budget: int | None = None
    #: The party sampling is billed to; None for a provider, whose trials name it.
    party: str | None = None

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def connection(self, trial: Trial) -> Connection: ...

    @abstractmethod
    async def serve(self, path: str | None) -> None:
        """Point at sampler weights, None for the base model."""

    @abstractmethod
    async def record(self, job: str, trials: Values[Trial]) -> Records | None:
        """The job's model calls by trial name; None for a provider."""

    def check(self, job: str, records: Records) -> None:
        """Raise if the run cannot go on after this job, once its row is written."""
        return None


@dataclass(frozen=True)
class Policy:
    """The model behind an agent; entering serves the start, leaving closes endpoint and client."""

    endpoint: Endpoint
    #: None for a provider.
    client: TrainingClient | None = None
    #: A sampler path to serve first: a checkpoint's, on a resumed run.
    start: str | None = None

    async def __aenter__(self) -> Policy:
        await self.endpoint.open()
        if self.start is not None:
            await self.endpoint.serve(self.start)
        elif self.client is not None:
            await self.sync("start")
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, failed: BaseException | None, tb: Any
    ) -> None:
        self.endpoint.close()
        if self.client is None:
            return
        if failed is None:
            await self.client.close("success")
            return
        status = "errored" if isinstance(failed, Exception) else "interrupted"
        await self.client.close(status, f"{type(failed).__name__}: {failed}")

    async def sync(self, tag: str) -> str:
        """Publish the client's weights as `tag` and serve them."""
        if self.client is None:
            raise RuntimeError("a provider's policy has no weights to sync")
        path = await self.client.save_weights_for_sampler(tag, ttl_seconds=SERVE_TTL)
        await self.endpoint.serve(path)
        return path


# ------------------------------------------------------------------ the agent


@dataclass(frozen=True)
class Harness:
    name: str
    #: Harbor's installed agent: a name with its version, or an import path.
    agent: str
    profile: Profile = field(default_factory=Profile)


def harness(named: str) -> Harness:
    """A built-in harness by the name `[rollout] harness` takes."""
    return Harness(bare_name(named).strip(), named, profile_for(named))


@dataclass(frozen=True)
class Agent:
    policy: Policy
    harness: list[Harness]
    modules: Candidate | None = None


# ----------------------------------------------------------------- the engine


@dataclass(frozen=True)
class Settings:
    """What one `engine.run` call runs under; the engine's own stand when a call brings none."""

    verify: bool = True
    setup_timeout: float | None = None
    timeout: float | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trial:
    job: str
    task: Task
    attempt: int
    harness: Harness
    dir: Path

    @property
    def name(self) -> str:
        return self.dir.name


class RolloutEngine(ABC):
    """Harbor. It takes no model; the agent carries the connection."""

    @abstractmethod
    async def run(
        self,
        job: str,
        tasks: Values[Task],
        attempts: int,
        *,
        agent: Agent,
        settings: Settings | None = None,
    ) -> list[Trial]:
        """Each task under each harness, `attempts` times, in plan order: task, harness, attempt."""


# ------------------------------------------------------------------ the group


@dataclass(frozen=True)
class Verdict:
    """A reward or a mask, never both; how Harbor said it ended; the verifier's numbers."""

    reward: float | None
    mask: str | None
    ended: str | None
    rewards: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Member:
    trial: Trial
    verdict: Verdict
    #: Empty when masked.
    sequences: tuple[Sequence, ...]
    sampled_tokens: int

    @property
    def measured(self) -> bool:
        return self.verdict.mask is None and self.verdict.reward is not None


@dataclass
class Group:
    """A task's attempts under one agent in one harness."""

    task: Task
    agent: Agent
    harness: Harness
    members: list[Member]
    #: Indexed like `members`, None on an unmeasured one; the trainer's rules fill them.
    shaped: list[float | None] = field(default_factory=list)
    advantages: list[float | None] = field(default_factory=list)

    @property
    def measured(self) -> list[Member]:
        return [one for one in self.members if one.measured]

    @property
    def reward_variance(self) -> float:
        """Over the measured members' shaped rewards; 0.0 with fewer than two."""
        found = [one for one in self.shaped if one is not None]
        return pvariance(found) if len(found) >= 2 else 0.0


# ----------------------------------------------------------------- the record


@dataclass(frozen=True)
class Evaluation:
    step: int
    epoch: int
    tasks: int
    rollouts: int
    graded: int
    masked: int
    #: None when nothing was graded.
    mean: float | None
