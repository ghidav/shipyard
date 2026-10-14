"""Harbor as the rollout engine: a job's trials in plan order, each dialling the
connection the agent's endpoint gives it, under one concurrency limit for the run."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
from shipyard.modules import deliver, merge_fields
from shipyard.proxy.profiles import bare_name
from shipyard.types import Agent, Connection, Harness, RolloutEngine, Settings, Task, Trial

if TYPE_CHECKING:
    from shipyard.config import Rollout
    from shipyard.progress import Reporter

logger = logging.getLogger(__name__)

#: Modal's image builder, and the version that builds a Dockerfile as written: the
#: legacy one injects `python -m pip` into images that have no python.
MODAL_IMAGE_BUILDER = "MODAL_IMAGE_BUILDER_VERSION"
MODAL_IMAGE_BUILDER_VERSION = "2025.06"

#: How one trial is run: Harbor's `Trial`, or a test's stand-in for it.
Runner = Callable[[TrialConfig], Awaitable[Any]]
#: Where a job's modules are delivered: under the job's directory, beside its trials.
DELIVERED = "modules"


class HarborEngine(RolloutEngine):
    """Built from `[rollout]`: the sandbox, the jobs directory, the concurrency limit every
    job shares, and the settings a call runs under unless it brings its own."""

    def __init__(
        self, rollout: Rollout, *, run_trial: Runner | None = None, progress: Reporter | None = None
    ) -> None:
        self.sandbox = rollout.sandbox
        self.jobs_dir = Path(rollout.jobs_dir)
        self.settings = Settings(
            setup_timeout=rollout.setup_timeout,
            timeout=rollout.timeout,
            env=dict(rollout.env),
            kwargs=dict(rollout.kwargs),
        )
        self.run_trial = run_trial
        self.progress = progress
        self._limit = asyncio.Semaphore(rollout.concurrency)

    async def run(
        self,
        job: str,
        tasks: Sequence[Task],
        attempts: int,
        *,
        agent: Agent,
        settings: Settings | None = None,
    ) -> list[Trial]:
        """A trial whose runner raised keeps its slot with no result, which admission
        masks. The agent's modules are delivered once, under the job's directory."""
        if not tasks:
            raise ValueError(f"job {job} names no task")
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1; got {attempts}")
        settings = self.settings if settings is None else settings
        trials_dir = self.jobs_dir / job
        delivered = (
            deliver(agent.modules, trials_dir / DELIVERED) if agent.modules is not None else {}
        )
        configs = [self._agent_config(one, settings, delivered) for one in agent.harness]
        plan = [
            (Path(task), index, attempt)
            for task in tasks
            for index in range(len(agent.harness))
            for attempt in range(attempts)
        ]
        trials_dir.mkdir(parents=True, exist_ok=True)
        if self.sandbox == "modal":
            os.environ.setdefault(MODAL_IMAGE_BUILDER, MODAL_IMAGE_BUILDER_VERSION)
        endpoint = agent.policy.endpoint
        run_one = self.run_trial if self.run_trial is not None else _run_trial

        async def one(task: Path, index: int, attempt: int) -> Trial:
            config = _trial_config(
                task,
                trials_dir=trials_dir,
                agent=configs[index],
                sandbox=self.sandbox,
                verify=settings.verify,
            )
            trial = Trial(job, task, attempt, agent.harness[index], trials_dir / config.trial_name)
            config = connected(config, endpoint.connection(trial))
            async with self._limit:
                try:
                    await run_one(config)
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is None or current.cancelling():
                        raise  # ours: the run is stopping
                    # A cancellation not ours fails only this trial.
                    logger.warning("trial %s ended in a cancellation it did not own", trial.name)
                except Exception:
                    logger.exception("trial %s could not run", trial.name)
            if self.progress is not None:
                self.progress.finished(trial.dir, masked=verdict(trial.dir).mask is not None)
            return trial

        return list(await asyncio.gather(*(one(*planned) for planned in plan)))

    @staticmethod
    def _agent_config(
        harness: Harness, settings: Settings, delivered: Mapping[str, Any]
    ) -> AgentConfig:
        if not harness.agent.strip():
            raise ValueError(
                "name the harness under `[rollout] harness`: Harbor reads an unnamed agent "
                "as its oracle, which runs the task's reference solution instead of the policy"
            )
        name, version = _harness_at_version(harness.agent)
        fields: dict[str, Any] = {
            "name": name,
            "env": dict(settings.env),
            "kwargs": {**({"version": version} if version else {}), **settings.kwargs},
            "override_setup_timeout_sec": settings.setup_timeout,
            "override_timeout_sec": settings.timeout,
        }
        merge_fields(fields, delivered)
        return AgentConfig(**fields)


def quiet_litellm() -> None:
    """Stop litellm printing its provider list to stdout, over the progress bars, for
    every model name it cannot place. Idempotent; a no-op without litellm."""
    try:
        import litellm
    except ImportError:  # pragma: no cover - Harbor depends on it
        return
    litellm.suppress_debug_info = True


async def _run_trial(config: TrialConfig) -> Any:
    """Create and run Harbor's trial. Imported late because it pulls in the agents, the
    backends and the registry."""
    from harbor.trial.trial import Trial

    quiet_litellm()
    trial = await Trial.create(config)
    return await trial.run()


def _trial_config(
    task: Path, *, trials_dir: Path, agent: AgentConfig, sandbox: str, verify: bool
) -> TrialConfig:
    """`trial_name` is left empty so Harbor names it `<task>__<7 chars>`, as `harbor run`
    does. `delete=True` is Harbor's default, set explicitly because a run opens thousands
    of trials."""
    return TrialConfig(
        task=TaskConfig(path=task),
        trials_dir=trials_dir,
        agent=agent,
        environment=EnvironmentConfig(type=sandbox, delete=True),
        verifier=VerifierConfig(disable=not verify),
    )


def connected(config: TrialConfig, connection: Connection) -> TrialConfig:
    """A copy of the trial's config dialling `connection`: its model name, its env over
    the job's, its kwargs laid over the job's, and its host on an allowlisted task. The
    agent config is shared by the job, hence the copy."""
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


def laid_over(under: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `over` onto `under`, table by table, so a blueprint's `opencode_config` keeps
    its own keys beside the profile's. `over` wins where both set one. Values of `over`
    are deep-copied so trials do not share the profile's objects."""
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
    `[agent]` override when it names a mode, else the `[environment]` baseline. False for
    a task whose `task.toml` cannot be read; Harbor says why at trial time."""
    try:
        config = TaskToml.model_validate_toml((task / "task.toml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - not this module's refusal to make
        return False
    phase = config.agent.explicit_phase_policy()
    policy = phase if phase is not None else config.environment.resolve_baseline()
    return policy.network_mode == NetworkMode.ALLOWLIST


def _harness_at_version(harness: str) -> tuple[str, str | None]:
    """Split `pi@0.85.1` into the halves Harbor takes: the agent's name and the version,
    passed as its `version` kwarg. A name with a colon has no version."""
    name = bare_name(harness)
    version = harness[len(name) + 1 :] if name != harness else ""
    return name, version or None
