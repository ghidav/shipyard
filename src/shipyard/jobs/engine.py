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

from shipyard.groups import Trial
from shipyard.jobs.admit import judge
from shipyard.jobs.harnesses import Harness, bare_name
from shipyard.jobs.modules import Candidate, deliver
from shipyard.jobs.progress import Bars
from shipyard.jobs.secrets import Secrets, scrub

if TYPE_CHECKING:
    from shipyard.config import Rollout
    from shipyard.models.endpoints import Connection

logger = logging.getLogger(__name__)

#: Modal's image-builder variable, and the version that builds a Dockerfile as written. The
#: legacy builder injects `python -m pip` into images that have no Python.
MODAL_IMAGE_BUILDER = "MODAL_IMAGE_BUILDER_VERSION"
MODAL_IMAGE_BUILDER_VERSION = "2025.06"
#: Directory under the job's directory, next to its trials, where its modules are delivered.
DELIVERED = "modules"

#: Runs one trial: Harbor's `Trial` by default, or a stand-in in tests.
TrialRunner = Callable[[TrialConfig], Awaitable[Any]]


@dataclass
class Job:
    name: str
    purpose: str = "training"
    step: int | None = None
    modules: Candidate | None = None
    #: The harness its trials run, such as `pi@0.85.1`.
    harness: Harness | None = None
    #: Grade a trial cut at its timeout by the verifier's reward, rather than mask it.
    grade_timeouts: bool = False
    trials: list[Trial] = field(default_factory=list)


class HarborEngine:
    def __init__(
        self,
        rollout: Rollout,
        *,
        env: Mapping[str, str] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        secrets: Secrets | None = None,
        run_trial: TrialRunner | None = None,
    ) -> None:
        self.sandbox = rollout.sandbox
        self.jobs_dir = Path(rollout.jobs_dir)
        self.setup_timeout, self.timeout = rollout.setup_timeout, rollout.timeout
        self.env, self.kwargs = dict(env or {}), dict(kwargs or {})
        self.secrets = Secrets() if secrets is None else secrets
        self.run_trial = run_trial
        self.progress = Bars()
        self._limit = asyncio.Semaphore(rollout.concurrency)

    async def run(self, job: Job, tasks: Sequence[Path], attempts: int, *, endpoint: Any) -> Job:
        """Run and judge each task with the job's harness, then return the job, filled in."""
        harness = job.harness
        if harness is None:
            raise ValueError(f"job {job.name} names no harness")
        if not tasks:
            raise ValueError(f"job {job.name} names no task")
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1; got {attempts}")
        trials_dir = self.jobs_dir / job.name
        delivered = deliver(job.modules, trials_dir / DELIVERED) if job.modules is not None else {}
        agent = self._agent_config(harness, delivered)
        plan = [Path(task) for task in tasks for _ in range(attempts)]
        trials_dir.mkdir(parents=True, exist_ok=True)
        if harness.profile.in_process is None:
            cut = sorted({t.name for t in plan if network_mode(t) == NetworkMode.NO_NETWORK})
            if cut:
                raise ValueError(
                    f"job {job.name}: {', '.join(cut)} run their agent with no network, so "
                    f"{harness.name}, which calls the model from the container, cannot reach it"
                )
        first = _trial_config(plan[0], trials_dir=trials_dir, agent=agent, sandbox=self.sandbox)
        preflight(first, endpoint, harness, job.name)
        if self.sandbox == "modal":
            os.environ.setdefault(MODAL_IMAGE_BUILDER, MODAL_IMAGE_BUILDER_VERSION)
        run_one = self.run_trial if self.run_trial is not None else _run_trial

        async def one(task: Path) -> Trial:
            config = _trial_config(task, trials_dir=trials_dir, agent=agent, sandbox=self.sandbox)
            trial = Trial(job.name, task, harness, trials_dir / config.trial_name)
            config = connected(config, endpoint.connection(trial))
            async with self._limit:
                held = await self.secrets.acquire(trial)
                for name in held:  # Harbor fills the task's `${NAME}` from here first
                    os.environ.setdefault(name, "")
                try:
                    await run_one(holding(config, held))
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is None or current.cancelling():
                        raise  # ours: the run is stopping
                    # A cancellation not ours fails only this trial.
                    logger.warning("trial %s ended in a cancellation it did not own", trial.name)
                except Exception:
                    logger.exception("trial %s could not run", trial.name)
                finally:
                    if held:
                        try:
                            await asyncio.to_thread(scrub, trial.dir, held)
                        finally:
                            await self.secrets.release(trial, held)
            self.progress.finished(job.name)
            return trial

        self.progress.started(job.name, trials=len(plan))
        job.trials = list(await asyncio.gather(*(one(task) for task in plan)))
        recorded = await endpoint.record(job.name, job.trials)
        for trial in job.trials:
            trial.records = None if recorded is None else list(recorded.get(trial.name) or [])
        judge(job)
        self.progress.done(job.name, masked=sum(not one.measured for one in job.trials))
        return job

    def close(self) -> None:
        self.progress.close()

    def _agent_config(self, harness: Harness, delivered: Mapping[str, Any]) -> AgentConfig:
        if not harness.agent.strip():
            raise ValueError(
                "name the harness under `[agent] harness`: Harbor reads an unnamed agent "
                "as its oracle, which runs the task's reference solution instead of the policy"
            )
        name, version = _harness_at_version(harness.agent)
        fields: dict[str, Any] = {
            "name": name,
            "env": dict(self.env),
            "kwargs": {**({"version": version} if version else {}), **self.kwargs},
            "override_setup_timeout_sec": self.setup_timeout,
            "override_timeout_sec": self.timeout,
        }
        return AgentConfig(**laid_over(fields, delivered))


def quiet_litellm() -> None:
    import litellm

    litellm.suppress_debug_info = True


def preflight(config: TrialConfig, endpoint: Any, harness: Harness, job: str) -> None:
    """Run Harbor's own check of the agent config a trial would run with."""
    from harbor.agents.factory import AgentFactory

    probe = Trial(job, Path(str(config.task.path)), harness, Path(config.trials_dir) / "preflight")
    agent = connected(config, endpoint.connection(probe)).agent
    try:
        AgentFactory.run_preflight(agent)
    except ValueError as refused:
        raise ValueError(f"the {harness.name} harness refuses its config: {refused}") from None


async def _run_trial(config: TrialConfig) -> Any:
    """Create and run a Harbor trial."""
    from harbor.trial.trial import Trial as HarborTrial

    quiet_litellm()
    trial = await HarborTrial.create(config)
    return await trial.run()


def _trial_config(task: Path, *, trials_dir: Path, agent: AgentConfig, sandbox: str) -> TrialConfig:
    """Build Harbor's config for one trial."""
    return TrialConfig(
        task=TaskConfig(path=task),
        trials_dir=trials_dir,
        agent=agent,
        environment=EnvironmentConfig(type=sandbox, delete=True),
        verifier=VerifierConfig(disable=False),
    )


def connected(config: TrialConfig, connection: Connection) -> TrialConfig:
    """Return a copy of the trial config that connects to `connection`."""
    update: dict[str, Any] = {
        "model_name": connection.model,
        "env": {**dict(config.agent.env or {}), **connection.env},
        "kwargs": laid_over(config.agent.kwargs or {}, connection.kwargs),
    }
    host = connection.host
    if host and allowlisted(Path(config.task.path)):
        hosts = list(config.agent.extra_allowed_hosts or [])
        update["extra_allowed_hosts"] = hosts + [host] * (host not in hosts)
    return config.model_copy(update={"agent": config.agent.model_copy(update=update)})


def holding(config: TrialConfig, held: Mapping[str, str]) -> TrialConfig:
    """Return a copy of the trial config with `held` in its environment and verifier env."""
    if not held:
        return config
    environment = config.environment.model_copy(update={"env": {**config.environment.env, **held}})
    verifier = config.verifier.model_copy(update={"env": {**config.verifier.env, **held}})
    return config.model_copy(update={"environment": environment, "verifier": verifier})


def laid_over(under: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge `over` onto `under`, with `over` winning where both set a key."""
    merged = dict(under)
    for key, value in over.items():
        below = merged.get(key)
        if isinstance(value, Mapping):
            merged[key] = laid_over(below if isinstance(below, Mapping) else {}, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def network_mode(task: Path) -> NetworkMode | None:
    """Return the network mode the task's agent phase runs in."""
    try:
        config = TaskToml.model_validate_toml((task / "task.toml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - not this module's refusal to make
        return None
    phase = config.agent.explicit_phase_policy()
    policy = phase if phase is not None else config.environment.resolve_baseline()
    return policy.network_mode


def allowlisted(task: Path) -> bool:
    """Return whether the task's agent phase runs in Harbor's allowlist network mode."""
    return network_mode(task) == NetworkMode.ALLOWLIST


def _harness_at_version(harness: str) -> tuple[str, str | None]:
    """Split a harness into Harbor's agent name and its `version` kwarg."""
    name = bare_name(harness)
    version = harness[len(name) + 1 :] if name != harness else ""
    return name, version or None
