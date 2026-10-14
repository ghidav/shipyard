"""The vocabulary of a run: the agent and its policy, the three parts as abstract classes,
and what they hand each other. Nothing here does work."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Iterator, Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pstdev
from typing import Any

import tinker

from shipyard.modules import Candidate as Modules
from shipyard.proxy.profiles import Profile, bare_name, profile_for
from shipyard.proxy.wire import Picture, Record

__all__ = [
    "EPSILON",
    "Adam",
    "Agent",
    "Batches",
    "Checkpoint",
    "Connection",
    "Cursor",
    "Datum",
    "Evaluation",
    "Forward",
    "Group",
    "Harness",
    "Harvest",
    "Member",
    "Modules",
    "Picture",
    "Policy",
    "Profile",
    "Record",
    "RolloutEngine",
    "SamplingClient",
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
#: Below this spread a group has no gradient.
EPSILON = 1e-6


# ------------------------------------------------------------------ the model


@dataclass(frozen=True)
class Sequence:
    """One chain: the last record's prompt and completion, which hold every earlier one as
    a prefix."""

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
    metrics: dict[str, float] = field(default_factory=dict)


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


class TrainingClient(ABC):
    """Tinker's training client as a recipe uses it. The step calls return awaitables so a
    step submits every part before it reads the first. A TTL of None keeps a save forever."""

    updates: int = 0

    @abstractmethod
    def forward_backward(
        self, datums: Values[Datum], loss_fn: str, loss_config: Mapping[str, float] | None
    ) -> Awaitable[Forward]: ...

    @abstractmethod
    def optim_step(self, adam: Adam, rate: float) -> Awaitable[None]: ...

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
    async def close(self, status: str) -> None: ...


@dataclass(frozen=True)
class Connection:
    """What a trial's harness dials. A provider's is the model name alone."""

    model: str | None
    address: str | None = None
    token: str | None = field(default=None, repr=False)
    env: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: The proxy's host, for a task on an allowlist.
    host: str | None = None


@dataclass(frozen=True)
class Harvest:
    """One job's records, drained from the proxy once, by trial name."""

    records: dict[str, list[Record]]
    #: Turns each harness asked for, and the trials whose log says the last call failed.
    asked: dict[str, int] = field(default_factory=dict)
    failed: set[str] = field(default_factory=set)
    counted: dict[str, int] = field(default_factory=dict)
    noted: dict[str, int] = field(default_factory=dict)
    party: str = "tinker"
    #: Why the run stops once the job row is written; None to go on.
    stopped: str | None = None


class SamplingClient(ABC):
    """The model as the endpoint a harness dials. The one part that knows whether this run
    serves the model or a provider does."""

    #: Sampled tokens one rollout may spend; None for a provider.
    budget: int | None = None

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def connection(self, trial: Trial) -> Connection: ...

    @abstractmethod
    def serve(self, path: str) -> None: ...

    @abstractmethod
    async def record(self, job: str, trials: Values[Trial]) -> Harvest | None:
        """The job's harvest; None for a provider."""


@dataclass(frozen=True)
class Policy:
    sampling: SamplingClient
    #: None for a provider.
    client: TrainingClient | None = None
    #: A sampler path to serve first: a checkpoint's, on a resumed run.
    start: str | None = None


# ------------------------------------------------------------------ the agent


@dataclass(frozen=True)
class Harness:
    name: str
    #: Harbor's installed agent: a name with its version, or an import path.
    agent: str
    profile: Profile = field(default_factory=Profile)
    effort: str | None = None


def harness(named: str, *, effort: str | None = None) -> Harness:
    """A built-in harness by the name `[rollout] harness` takes."""
    return Harness(bare_name(named).strip(), named, profile_for(named), effort)


@dataclass(frozen=True)
class Agent:
    policy: Policy
    harness: list[Harness]
    modules: Modules | None = None


# ----------------------------------------------------------------- the engine


@dataclass(frozen=True)
class Settings:
    """What one `engine.run` call overrides in `[rollout]`; None keeps the blueprint's."""

    sandbox: str | None = None
    concurrency: int | None = None
    timeout: float | None = None
    setup_timeout: float | None = None
    verify: bool | None = None
    env: Mapping[str, str] | None = None
    kwargs: Mapping[str, Any] | None = None


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

    @property
    def result(self) -> dict[str, Any] | None:
        from shipyard.admit import result_of

        return result_of(self.dir)


class RolloutEngine(ABC):
    """Harbor. It takes no model; the agent carries the connection."""

    @abstractmethod
    async def run(
        self,
        tasks: Values[Task],
        attempts: int,
        *,
        agent: Agent,
        settings: Settings | None = None,
    ) -> list[Trial]:
        """Each task under each of the agent's harnesses, `attempts` times, in plan order:
        task, harness, attempt."""


# ------------------------------------------------------------------ the group


@dataclass(frozen=True)
class Verdict:
    """A trial's reward or its mask, never both, and how Harbor said it ended."""

    reward: float | None
    mask: str | None
    ended: str | None


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
    """A task's attempts under one agent in one harness. `shaped` and `advantages` are
    indexed like `members`, None on an unmeasured one; the trainer's rules fill them."""

    task: Task
    agent: Agent
    harness: Harness
    members: list[Member]
    shaped: list[float | None] = field(default_factory=list)
    advantages: list[float | None] = field(default_factory=list)

    @property
    def measured(self) -> list[Member]:
        return [one for one in self.members if one.measured]

    @property
    def has_gradient(self) -> bool:
        """On the shaped rewards, before the overlong term: two or more, not all equal."""
        found = [one for one in self.shaped if one is not None]
        return len(found) >= 2 and pstdev(found) > EPSILON


# ----------------------------------------------------------------- the record


@dataclass(frozen=True)
class Cursor:
    step: int
    updates: int
    epoch: int
    #: Batches drawn in the epoch, refills included.
    drawn: int


@dataclass(frozen=True)
class Checkpoint:
    """The agent written down, with the trainer's cursor."""

    tag: str
    state_path: str
    sampler_path: str
    #: None keeps it until deleted.
    ttl_hours: float | None
    cursor: Cursor
    modules: str | None = None
    harness: tuple[str, ...] = ()


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
