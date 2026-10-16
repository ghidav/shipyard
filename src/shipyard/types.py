"""Shared types of a run: the model, the agent, the rollout engine, groups and evaluations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pvariance
from typing import Any

import tinker

from shipyard.harnesses import Profile, bare_name, profile_for
from shipyard.modules import Candidate
from shipyard.proxy.wire import Picture, Record

__all__ = [
    "Adam",
    "Agent",
    "Connection",
    "Datum",
    "Endpoint",
    "Evaluation",
    "Forward",
    "Group",
    "Harness",
    "Job",
    "Member",
    "Policy",
    "Records",
    "RolloutEngine",
    "SERVE_TTL",
    "Sequence",
    "Settings",
    "Task",
    "TrainingClient",
    "TrainingStep",
    "Trial",
    "Update",
    "Verdict",
    "harness",
]

Task = Path
Datum = tinker.Datum
#: A job's recorded model calls, keyed by trial name.
Records = dict[str, list[Record]]
#: TTL in seconds of served sampler weights: long enough to outlast the batch sampled from
#: them, and short enough that a crashed run's leftovers expire on their own.
SERVE_TTL = 12 * 3600


# ------------------------------------------------------------------ the model


@dataclass(frozen=True)
class Sequence:
    """One chain of model calls as a single token sequence: the last record's prompt and
    completion, which contain every earlier call."""

    tokens: tuple[int, ...]
    #: (start, end) positions of each completion, in chain order.
    spans: tuple[tuple[int, int], ...]
    #: The sampler's logprobs at the targets; None when a record carried none.
    sampled_logprobs: tuple[float, ...] | None
    images: tuple[Picture, ...] = ()

    @property
    def targets(self) -> tuple[int, ...]:
        """Positions of every completion token, in chain order."""
        return tuple(at for start, end in self.spans for at in range(start, end))


@dataclass(frozen=True)
class Forward:
    """Result of `forward_backward`: each datum's logprobs at every input position."""

    logprobs: list[list[float]]


@dataclass(frozen=True)
class Adam:
    beta1: float
    beta2: float
    eps: float
    weight_decay: float = 0.0
    #: Tinker's `grad_clip_norm`; 0 disables clipping.
    grad_clip_norm: float = 0.0


@dataclass(frozen=True)
class Update:
    """Statistics of the gradient update one training step took."""

    train_tokens: int
    substeps: int
    learning_rate: float
    loss_fn: str
    kl_v1: float | None
    kl_v2: float | None
    entropy: float | None
    seconds: float
    #: Number of sequences the step trained on.
    sequences: int = 0
    #: Mean of the reference logprobs minus the anchor's over those sequences; None without a
    #: KL term.
    anchor_kl: float | None = None


@dataclass(frozen=True)
class TrainingStep:
    """Result of one training step. `update` is None when no gradient was taken: there was
    no training client, or no group had any reward variance."""

    index: int
    groups: list[Group]
    update: Update | None = None


class TrainingClient(ABC):
    """Interface to Tinker's training client. `forward_backward` and `optim_step` submit the
    request when awaited and return an awaitable of its result."""

    @abstractmethod
    async def open(self) -> None:
        """Start the training session. `Policy` calls this on entry."""

    @abstractmethod
    async def anchor(self) -> Any:
        """Return a sampling client on the weights as first opened, for the KL term."""

    @abstractmethod
    async def forward_backward(
        self, datums: Values[Datum], loss_fn: str, loss_config: Mapping[str, float] | None
    ) -> Awaitable[Forward]: ...

    @abstractmethod
    async def optim_step(self, adam: Adam, rate: float) -> Awaitable[None]: ...

    @abstractmethod
    async def compute_logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
        """Return the trainer's logprobs at every target of every sequence, from one batched
        forward pass."""

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
    """How a trial's harness reaches the model. For a provider, only `model` is set."""

    model: str | None
    address: str | None = None
    token: str | None = field(default=None, repr=False)
    env: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: The proxy's host name, added to the allowed hosts of a task in Harbor's allowlist
    #: network mode.
    host: str | None = None


class Endpoint(ABC):
    """The model endpoint a harness calls, served either by this run or by a provider."""

    #: Most tokens one rollout may sample; None for a provider.
    budget: int | None = None
    #: The party sampling is billed to; None for a provider, whose trials report it.
    party: str | None = None

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def connection(self, trial: Trial) -> Connection: ...

    @abstractmethod
    async def serve(self, path: str | None) -> None:
        """Serve the sampler weights at `path`, or the base model when `path` is None."""

    @abstractmethod
    async def record(self, job: str, trials: Values[Trial]) -> Records | None:
        """Return the job's model calls by trial name, or None for a provider."""

    def check(self, job: str, records: Records) -> None:
        """Raise if the run cannot continue after this job. Called after the job's row is
        written."""
        return None


@dataclass
class Policy:
    """The model behind an agent: an endpoint and, for a served model, a training client.

    Entering opens the client and the endpoint and serves the starting weights; exiting
    closes both."""

    endpoint: Endpoint
    #: Training client; None for a provider.
    client: TrainingClient | None = None
    #: Sampler path to serve first, such as a checkpoint's on a resumed run.
    start: str | None = None
    #: State path to load once the client is open, such as a checkpoint's on a resumed run.
    state: str | None = None
    #: Whether to take the KL anchor from the weights as first opened, before any state
    #: is loaded.
    anchored: bool = False
    anchor: Any = field(default=None, init=False)

    async def __aenter__(self) -> Policy:
        try:
            if self.client is not None:
                await self.client.open()
                if self.anchored:
                    self.anchor = await self.client.anchor()
                if self.state is not None:
                    await self.client.load_state(self.state)
            await self.endpoint.open()
            if self.start is not None:
                await self.endpoint.serve(self.start)
            elif self.client is not None:
                await self.sync("start")
        except BaseException as failed:
            await self.__aexit__(type(failed), failed, failed.__traceback__)
            raise
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
        """Save the client's weights for sampling as `tag`, serve them, and return their
        path."""
        if self.client is None:
            raise RuntimeError("a provider's policy has no weights to sync")
        path = await self.client.save_weights_for_sampler(tag, ttl_seconds=SERVE_TTL)
        await self.endpoint.serve(path)
        return path


# ------------------------------------------------------------------ the agent


@dataclass(frozen=True)
class Harness:
    name: str
    #: The Harbor agent to install: a name with an optional version, or an import path.
    agent: str
    profile: Profile = field(default_factory=Profile)


def harness(named: str) -> Harness:
    """Return the harness for a name as `[agent] harness` gives it."""
    return Harness(bare_name(named).strip(), named, profile_for(named))


@dataclass(frozen=True)
class Agent:
    policy: Policy
    harness: list[Harness]
    modules: Candidate | None = None


# ----------------------------------------------------------------- the engine


@dataclass(frozen=True)
class Settings:
    """Settings for one `engine.run` call. A call without them uses the engine's own,
    built from `[rollout]` and `[agent]`."""

    verify: bool = True
    setup_timeout: float | None = None
    timeout: float | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: Whether each trial acquires one of the engine's secrets. Off for a trial that runs no
    #: task.
    secrets: bool = True


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


@dataclass
class Job:
    """One sampling of a policy over a set of tasks.

    It is created with its name and metadata, filled in by the engine and the endpoint,
    judged by admission, then logged by the trainer."""

    name: str
    purpose: str = "training"
    step: int | None = None
    modules: Candidate | None = None
    #: Extra fields for the job's row, such as a search's round or a reflection's component.
    about: dict[str, Any] = field(default_factory=dict)
    trials: list[Trial] = field(default_factory=list)
    records: Records | None = None
    verdicts: list[Verdict] = field(default_factory=list)
    #: Read from the harness logs, by trial name: the model calls each trial asked for, and
    #: the trials whose last model call failed.
    asked: dict[str, int] | None = None
    failed: set[str] | None = None


class RolloutEngine(ABC):
    """Runs trials through Harbor. It holds no model; the agent carries the connection."""

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
        """Run each task under each harness `attempts` times, and return the trials in plan
        order: by task, then harness, then attempt."""


# ------------------------------------------------------------------ the group


@dataclass(frozen=True)
class Verdict:
    """Outcome of one trial: a reward or a mask reason (never both), how Harbor reported it
    ended, and the verifier's individual rewards."""

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
    #: Shaped rewards and advantages, indexed like `members` and None for an unmeasured
    #: member. The trainer fills them in.
    shaped: list[float | None] = field(default_factory=list)
    advantages: list[float | None] = field(default_factory=list)

    @property
    def measured(self) -> list[Member]:
        return [one for one in self.members if one.measured]

    @property
    def reward_variance(self) -> float:
        """Population variance of the measured members' shaped rewards; 0.0 with fewer
        than two."""
        found = [one for one in self.shaped if one is not None]
        return pvariance(found) if len(found) >= 2 else 0.0


# ----------------------------------------------------------------- the record


@dataclass(frozen=True)
class Evaluation:
    step: int
    tasks: int
    rollouts: int
    graded: int
    masked: int
    #: Mean reward over the graded rollouts; None when none were graded.
    reward_mean: float | None
