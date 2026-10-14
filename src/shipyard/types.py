"""The vocabulary of a run: what a blueprint builds, what the parts hand each other, and
the three parts as protocols. Nothing here does work. The agent is what shaped a sample and
the unit a group is keyed on; the policy is the model behind it, as a sampling client the
harness dials and a training client the step moves. A trial is one attempt at one task under
one harness; a group is a task's attempts under one agent in one harness; the trainer's
rules fill the group's shaped rewards and advantages.

The dataclasses that keep their shape across the refactor (`Sequence`, `Verdict`, `Adam`,
`Update`) live here and are re-exported by the modules that defined them. The ones that change
shape (`Group`, `Member`, `Harvest`, `Checkpoint`) are defined here beside their current
versions, which the later slices retire. `Evolution` stays in `gepa.cycle` with the ranking it
depends on."""

from __future__ import annotations

from collections.abc import Awaitable, Iterator, Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pstdev
from typing import Any, Protocol

import tinker

from shipyard.modules import Candidate as Modules
from shipyard.proxy.profiles import Profile, bare_name, profile_for
from shipyard.proxy.wire import Picture, Record

__all__ = [
    "EPSILON",
    "Adam",
    "Agent",
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

#: A task reference: the directory Harbor loads it from.
Task = Path
#: What the training client consumes: the cookbook's datum, built by `pack.datum`.
Datum = tinker.Datum
#: Guards the division when a group has no reward spread.
EPSILON = 1e-6


# ------------------------------------------------------------------ the model


@dataclass(frozen=True)
class Sequence:
    """One chain as the trainer reads it: the last record's prompt and completion, which
    hold every earlier prompt and completion as a prefix. `spans` are where each completion
    sits, in chain order. `mu` holds the sampler's logprobs there, or None when a record had
    none."""

    tokens: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    mu: tuple[float, ...] | None
    #: The pictures behind the `IMAGE_TOKEN` runs in `tokens`, from the chain's last record.
    images: tuple[Picture, ...] = ()

    @property
    def targets(self) -> tuple[int, ...]:
        """Every position the policy wrote, ascending, each once."""
        return tuple(at for start, end in self.spans for at in range(start, end))


@dataclass(frozen=True)
class Forward:
    """What one `forward_backward` returns: the logprobs the weights now give each datum at
    every position of its input, and the backend's metrics."""

    logprobs: list[list[float]]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Adam:
    """AdamW as a recipe's paper sets it: the betas, eps, the decoupled weight decay, the
    global gradient norm each step is clipped to (Tinker's `grad_clip_norm`; 0 is none), and
    the `warmup` steps over which the learning rate rises linearly to its value (0 is none)."""

    beta1: float
    beta2: float
    eps: float
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0
    warmup: int = 0

    def rate(self, learning_rate: float, done: int) -> float:
        """The learning rate for the update after `done` updates: `(done + 1) / warmup` of
        `learning_rate` during the warm-up, so the first update moves the weights, and all
        of `learning_rate` afterwards."""
        if done >= self.warmup:
            return learning_rate
        return learning_rate * (done + 1) / self.warmup


@dataclass(frozen=True)
class Update:
    """What one step did: the tokens sent through `forward_backward`, the substeps run, the
    learning rate and loss function, and the KL and entropy of the step's own logprobs
    against mu (None where unreadable)."""

    train_tokens: int
    substeps: int
    learning_rate: float
    loss_fn: str
    kl_v1: float | None
    kl_v2: float | None
    entropy: float | None
    seconds: float


class TrainingClient(Protocol):
    """Tinker's training client, reduced to what a recipe needs. The two step calls return
    awaitables so a step submits every part before it reads the first. Every save carries a
    TTL in seconds, None to keep forever."""

    #: The gradients applied so far; restored from the state on load.
    updates: int

    def forward_backward(
        self, datums: Values[Datum], loss_fn: str, loss_config: Mapping[str, float] | None
    ) -> Awaitable[Forward]: ...

    def optim_step(self, adam: Adam, rate: float) -> Awaitable[None]: ...

    async def compute_logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
        """mu at every target of every sequence, from one batched forward, pictures
        included."""
        ...

    async def save_weights_for_sampler(self, name: str, *, ttl_seconds: int | None) -> str: ...

    async def save_state(self, name: str, *, ttl_seconds: int | None) -> str: ...

    async def load_state(self, path: str) -> None: ...

    async def close(self, status: str) -> None: ...


@dataclass(frozen=True)
class Connection:
    """What a trial's harness is told to dial. For a served model: the model name on the
    harness's wire, the proxy's address for this trial and its token, with the profile's env
    and kwargs laid over the blueprint's, and the proxy's host for a task on an allowlist.
    For a provider: the model name alone, possibly None for the harness's default, with no
    address and no token, since the harness uses its own credentials."""

    model: str | None
    address: str | None = None
    token: str | None = field(default=None, repr=False)
    env: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    host: str | None = None


@dataclass(frozen=True)
class Harvest:
    """One job's records, drained from the proxy once: each trial's records by trial name,
    the turns each harness asked for and the trials whose harness log says their last call
    failed (both by trial name), the counts and notes the job row bills, the party billed,
    and why the run must stop once the row is written (None to go on)."""

    records: dict[str, list[Record]]
    asked: dict[str, int] = field(default_factory=dict)
    failed: set[str] = field(default_factory=set)
    counted: dict[str, int] = field(default_factory=dict)
    noted: dict[str, int] = field(default_factory=dict)
    party: str = "tinker"
    stopped: str | None = None


class SamplingClient(Protocol):
    """The model as an endpoint: what the harness dials. It is the only part that knows
    whether the model is served by this run or by a provider."""

    #: Sampled tokens one rollout may spend, as the proxy enforces it; None for a provider.
    budget: int | None

    async def open(self) -> None:
        """Start the proxy and note its placement and origin; nothing for a provider."""
        ...

    async def close(self) -> None: ...

    def connection(self, trial: Trial) -> Connection:
        """What this trial's harness dials: per trial, because the proxy files every call
        under the trial's name and the harvest fetches records by it."""
        ...

    def serve(self, path: str) -> None:
        """Point at new sampler weights. A provider refuses."""
        ...

    async def record(self, job: str, trials: Values[Trial]) -> Harvest | None:
        """The job's harvest, or None for a provider, whose calls nobody recorded."""
        ...


@dataclass(frozen=True)
class Policy:
    """What `[model]` names: the sampling client the harness dials, the training client the
    step moves (None for a provider), and a sampler path to serve first (a checkpoint's, on
    a resumed run)."""

    sampling: SamplingClient
    client: TrainingClient | None = None
    start: str | None = None


# ------------------------------------------------------------------ the agent


@dataclass(frozen=True)
class Harness:
    """Where the agent is deployed: the bare name, Harbor's installed agent (a name, with
    its version, or an import path), the proxy profile (its wire, env names, kwargs, turn
    counter), and the reasoning effort Harbor passes under this harness, whose sign flips
    per harness."""

    name: str
    agent: str
    profile: Profile = field(default_factory=Profile)
    effort: str | None = None


def harness(named: str, *, effort: str | None = None) -> Harness:
    """A built-in harness by the name `[rollout] harness` takes, `pi@0.85.1` or an import
    path, with the profile its bare name has. A harness Harbor does not ship gets the
    generic profile, as today."""
    return Harness(bare_name(named).strip(), named, profile_for(named), effort)


@dataclass(frozen=True)
class Agent:
    """Everything that shaped a sample: the policy, the harnesses it runs under (one or
    more), and the modules it carries. A variant is a `replace`: another text, a provider
    teacher, the reflector."""

    policy: Policy
    harness: list[Harness]
    modules: Modules | None = None


# ----------------------------------------------------------------- the engine


@dataclass(frozen=True)
class Settings:
    """What one `engine.run` call overrides in `[rollout]`; None keeps the blueprint's. The
    reflection trial turns the verifier off and takes its own setup timeout."""

    sandbox: str | None = None
    concurrency: int | None = None
    timeout: float | None = None
    setup_timeout: float | None = None
    verify: bool | None = None
    env: Mapping[str, str] | None = None
    kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Trial:
    """One attempt at one task under one harness: the job it ran in, its index among the
    task's attempts, and its directory, where Harbor leaves `result.json` and the logs."""

    job: str
    task: Task
    attempt: int
    harness: Harness
    dir: Path

    @property
    def name(self) -> str:
        """Harbor's trial name, which the proxy files the trial's calls under."""
        return self.dir.name

    @property
    def result(self) -> dict[str, Any] | None:
        """`result.json` as a dict, or None when missing or unreadable."""
        from shipyard.admit import result_of

        return result_of(self.dir)


class RolloutEngine(Protocol):
    """Harbor. It runs tasks under an agent, each task under each of the agent's harnesses,
    `attempts` times, and returns the trials in plan order: task, then harness, then attempt.
    It takes no model; the agent carries the connection."""

    async def run(
        self,
        tasks: Values[Task],
        attempts: int,
        *,
        agent: Agent,
        settings: Settings | None = None,
    ) -> list[Trial]: ...


# ------------------------------------------------------------------ the group


@dataclass(frozen=True)
class Verdict:
    """A trial's reward or its mask (exactly one is set), and how Harbor said it ended."""

    reward: float | None
    mask: str | None
    ended: str | None


@dataclass(frozen=True)
class Member:
    """One attempt of a group: its trial, its verdict, its sequences (none when masked),
    and the tokens the policy wrote across its calls, which the length rule measures."""

    trial: Trial
    verdict: Verdict
    sequences: tuple[Sequence, ...]
    sampled_tokens: int

    @property
    def measured(self) -> bool:
        """Rewarded and unmasked: the members that form a baseline."""
        return self.verdict.mask is None and self.verdict.reward is not None


@dataclass
class Group:
    """A task's attempts under one agent in one harness, in plan order, masked ones included
    with no sequences. `shaped` and `advantages` are indexed like `members`, None on a member
    that was not measured; the trainer's rules fill them after the job is packed."""

    task: Task
    agent: Agent
    harness: Harness
    members: list[Member]
    shaped: list[float | None] = field(default_factory=list)
    advantages: list[float | None] = field(default_factory=list)

    @property
    def measured(self) -> list[Member]:
        """The members with a reward and no mask. Only these form a baseline."""
        return [one for one in self.members if one.measured]

    @property
    def has_gradient(self) -> bool:
        """Judged on the shaped rewards, before the overlong term, over the measured members:
        at least two, and not all equal."""
        found = [one for one in self.shaped if one is not None]
        return len(found) >= 2 and pstdev(found) > EPSILON


# ----------------------------------------------------------------- the record


@dataclass(frozen=True)
class Cursor:
    """Where the trainer stood: steps taken, steps that took a gradient, the epoch, and the
    batches drawn in it (refills included). A resume starts from here."""

    step: int
    updates: int
    epoch: int
    drawn: int


@dataclass(frozen=True)
class Checkpoint:
    """The agent written down, with the trainer's cursor: the state path a training client
    is rebuilt from, the sampler path rollouts are served from, the TTL in hours (None: kept
    until deleted), the modules' digest and the harness names."""

    tag: str
    state_path: str
    sampler_path: str
    ttl_hours: float | None
    cursor: Cursor
    modules: str | None = None
    harness: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evaluation:
    """One measurement of an agent on the test tasks: when it was taken, what it ran, how
    many attempts were graded or masked, and the mean over the graded ones (None when none
    was)."""

    step: int
    epoch: int
    tasks: int
    rollouts: int
    graded: int
    masked: int
    mean: float | None


#: The batches of one epoch, as a generator yields them.
Batches = Iterator[list[Task]]
