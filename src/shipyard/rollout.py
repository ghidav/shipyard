"""One Harbor job: a batch of tasks rolled out as trials under `jobs/<job>/`, in plan order."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harbor.models.task.config import NetworkMode
from harbor.models.task.config import TaskConfig as TaskToml
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)

from shipyard.admit import verdict
from shipyard.modules import Candidate, deliver, merge_fields
from shipyard.proxy.profiles import Profile, bare_name, slug_of

if TYPE_CHECKING:
    from shipyard.proxy.client import Proxy
    from shipyard.proxy.wire import Record
    from shipyard.ui import Reporter

logger = logging.getLogger(__name__)

#: Modal's image builder, and the version that builds a Dockerfile as written: the
#: legacy one injects `python -m pip` into images that have no python.
MODAL_IMAGE_BUILDER = "MODAL_IMAGE_BUILDER_VERSION"
MODAL_IMAGE_BUILDER_VERSION = "2025.06"

#: How one trial is run: Harbor's `Trial`, or a test's stand-in for it.
Runner = Callable[[TrialConfig], Awaitable[Any]]
#: Where a job's modules are delivered: under the job's directory, beside its trials.
DELIVERED = "modules"


@dataclass(frozen=True)
class Rollouts:
    """A finished job: its name, one directory per planned trial, what each one was, and,
    once the run has harvested them, the proxy's records and the harness's turn counts by
    trial name, which admission reads beside each `result.json`."""

    job: str
    trials: list[Path]
    plan: list[tuple[Path, int]]
    records: dict[str, list[Record]] | None = None
    asked: dict[str, int] | None = None
    #: The trials whose harness log says their last model call failed.
    failed: set[str] | None = None
    #: The trainer's update count when this job sampled, for credit to check against.
    updates: int | None = None

    def __len__(self) -> int:
        return len(self.trials)


@dataclass(frozen=True)
class Served:
    """How a trial reaches the run's own proxy: the proxy (an address per trial, the token,
    the host), the harness's profile, and whether the profile's env is laid on too."""

    proxy: Proxy
    profile: Profile = field(default_factory=Profile)
    fill_context: bool = False


async def rollout(
    batch: Sequence[Path],
    job: str,
    *,
    harness: str,
    model: str | None,
    sandbox: str,
    concurrency: int,
    rollouts: int,
    jobs_dir: str | Path,
    env: Mapping[str, str] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    verify: bool = True,
    setup_timeout: float | None = None,
    timeout: float | None = None,
    run_trial: Runner | None = None,
    progress: Reporter | None = None,
    served: Served | None = None,
    modules: Candidate | None = None,
) -> Rollouts:
    """Every task `rollouts` times, `concurrency` trials at once, back in plan order, the
    `modules` delivered under the job's directory and named on the agent config. A trial
    whose runner raised keeps its slot: no result there, which admission masks."""
    if rollouts < 1:
        raise ValueError(f"rollouts must be at least 1; got {rollouts}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1; got {concurrency}")
    if not harness or not harness.strip():
        raise ValueError(
            "name the harness under `[rollout] harness`: Harbor reads an unnamed agent as "
            "its oracle, which runs the task's reference solution instead of the policy"
        )
    name, version = _harness_at_version(harness)
    trials_dir = Path(jobs_dir) / job
    fields: dict[str, Any] = {
        "name": name,
        "model_name": model,
        "env": dict(env or {}),
        "kwargs": {**({"version": version} if version else {}), **(kwargs or {})},
        "override_setup_timeout_sec": setup_timeout,
        "override_timeout_sec": timeout,
    }
    if modules is not None:
        merge_fields(fields, deliver(modules, trials_dir / DELIVERED))
    agent = AgentConfig(**fields)
    plan = [(Path(task), index) for task in batch for index in range(rollouts)]
    configs = [
        _trial_config(task, trials_dir=trials_dir, agent=agent, sandbox=sandbox, verify=verify)
        for task, _ in plan
    ]
    trials_dir.mkdir(parents=True, exist_ok=True)
    if sandbox == "modal":
        os.environ.setdefault(MODAL_IMAGE_BUILDER, MODAL_IMAGE_BUILDER_VERSION)
    limit = asyncio.Semaphore(concurrency)
    run_one = run_trial if run_trial is not None else _run_trial

    async def one(config: TrialConfig) -> Path:
        trial_dir = Path(config.trials_dir) / config.trial_name
        if served is not None:
            config = served_config(config, served)
        async with limit:
            try:
                await run_one(config)
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is None or task.cancelling():
                    raise  # ours: the run is stopping, and every trial with it
                # A cancellation not ours is one trial's failure, not the batch's.
                logger.warning("trial %s ended in a cancellation it did not own", trial_dir.name)
            except Exception:
                logger.exception("trial %s could not run", trial_dir.name)
        if progress is not None:
            progress.finished(trial_dir, masked=verdict(trial_dir).mask is not None)
        return trial_dir

    trials = list(await asyncio.gather(*(one(config) for config in configs)))
    return Rollouts(job=job, trials=trials, plan=plan)


def quiet_litellm() -> None:
    """Stop litellm printing its provider list for every model name it cannot place,
    once per trial, to stdout, over the bars. Idempotent; a no-op without litellm."""
    try:
        import litellm
    except ImportError:  # pragma: no cover - Harbor depends on it
        return
    litellm.suppress_debug_info = True


async def _run_trial(config: TrialConfig) -> Any:
    """Harbor's trial, created and awaited here; imported late, since it drags the agents,
    the backends and the registry in for a caller that may never open a container."""
    from harbor.trial.trial import Trial

    quiet_litellm()
    trial = await Trial.create(config)
    return await trial.run()


def _trial_config(
    task: Path, *, trials_dir: Path, agent: AgentConfig, sandbox: str, verify: bool
) -> TrialConfig:
    """`trial_name` is left empty so Harbor names it `<task>__<7 chars>` as `harbor run`
    does; `delete=True` is Harbor's default said out loud, since a run opens thousands."""
    return TrialConfig(
        task=TaskConfig(path=task),
        trials_dir=trials_dir,
        agent=agent,
        environment=EnvironmentConfig(type=sandbox, delete=True),
        verifier=VerifierConfig(disable=not verify),
    )


def served_config(config: TrialConfig, served: Served) -> TrialConfig:
    """This trial's config pointed at the run's proxy, at the address naming the trial:
    the provider's own env names over the batch's env, the profile's kwargs laid over the
    batch's and, in allowlist mode, the proxy's host. A copy: the agent config is shared
    by the batch."""
    address = served.proxy.address_for(config.trial_name)
    profile = served.profile
    if profile.strip_v1:
        address = address.removesuffix("/v1")
    slug = slug_of(profile).upper()
    env = {
        **dict(config.agent.env or {}),
        **(dict(profile.env) if served.fill_context else {}),
        f"{slug}_BASE_URL": address,
        f"{slug}_API_KEY": served.proxy.token,
    }
    update: dict[str, Any] = {
        "env": env,
        "kwargs": laid_over(config.agent.kwargs or {}, profile.kwargs),
    }
    host = served.proxy.host
    if host and allowlisted(Path(config.task.path)):
        hosts = list(config.agent.extra_allowed_hosts or [])
        update["extra_allowed_hosts"] = hosts + [host] * (host not in hosts)
    return config.model_copy(update={"agent": config.agent.model_copy(update=update)})


def laid_over(under: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """`over` on `under`, table by table: a blueprint's `opencode_config` keeps its own
    keys beside the profile's, and `over` wins where both set one. Every value of `over`
    is copied, so a trial never holds the profile's own."""
    merged = dict(under)
    for key, value in over.items():
        below = merged.get(key)
        if isinstance(value, Mapping):
            merged[key] = laid_over(below if isinstance(below, Mapping) else {}, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def allowlisted(task: Path) -> bool:
    """Whether the task's agent phase runs in Harbor's allowlist network mode: the
    `[agent]` override when it names a mode, else the `[environment]` baseline. A task
    whose `task.toml` cannot be read is not, and Harbor says why at trial time."""
    try:
        config = TaskToml.model_validate_toml((task / "task.toml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - not this module's refusal to make
        return False
    phase = config.agent.explicit_phase_policy()
    policy = phase if phase is not None else config.environment.resolve_baseline()
    return policy.network_mode == NetworkMode.ALLOWLIST


def _harness_at_version(harness: str) -> tuple[str, str | None]:
    """`pi@0.85.1` as the halves Harbor takes: the agent's name, and the version as its
    `version` kwarg; a name with a colon is whole, and has no version."""
    name = bare_name(harness)
    version = harness[len(name) + 1 :] if name != harness else ""
    return name, version or None
