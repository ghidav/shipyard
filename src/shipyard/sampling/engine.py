"""Harbor rollout engine: runs a job's trials in plan order under one concurrency limit."""

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

from shipyard.harnesses import bare_name
from shipyard.modules import deliver
from shipyard.sampling.admit import verdict
from shipyard.sampling.secrets import Secrets, scrub
from shipyard.types import Agent, Connection, Harness, RolloutEngine, Settings, Task, Trial

if TYPE_CHECKING:
    from shipyard.config import Rollout
    from shipyard.progress import Reporter

logger = logging.getLogger(__name__)

#: Modal's image-builder variable, and the version that builds a Dockerfile as written. The
#: legacy builder injects `python -m pip` into images that have no Python.
MODAL_IMAGE_BUILDER = "MODAL_IMAGE_BUILDER_VERSION"
MODAL_IMAGE_BUILDER_VERSION = "2025.06"

#: Runs one trial: Harbor's `Trial` by default, or a stand-in in tests.
TrialRunner = Callable[[TrialConfig], Awaitable[Any]]
#: Directory under the job's directory, next to its trials, where its modules are delivered.
DELIVERED = "modules"


class HarborEngine(RolloutEngine):
    """Rollout engine that runs trials with Harbor.

    It is configured from `[rollout]` (sandbox, jobs directory, concurrency limit, timeouts)
    and from `[agent]` (the harness's Harbor `env` and `kwargs`). Each trial holds its
    `secrets` for exactly its duration."""

    def __init__(
        self,
        rollout: Rollout,
        *,
        env: Mapping[str, str] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        secrets: Secrets | None = None,
        run_trial: TrialRunner | None = None,
        progress: Reporter | None = None,
    ) -> None:
        self.sandbox = rollout.sandbox
        self.jobs_dir = Path(rollout.jobs_dir)
        self.settings = Settings(
            setup_timeout=rollout.setup_timeout,
            timeout=rollout.timeout,
            env=dict(env or {}),
            kwargs=dict(kwargs or {}),
        )
        self.secrets = Secrets() if secrets is None else secrets
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
        """Run each task `attempts` times in each harness and return the trials in plan order.

        A trial whose runner raises keeps its slot with no result, and admission masks it. Each
        trial holds the secrets it acquires from before Harbor starts it until after its
        teardown, and the secret values are scrubbed from its directory."""
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
                held = await self.secrets.acquire(trial) if settings.secrets else {}
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
                            scrub(trial.dir, held)
                        finally:
                            await self.secrets.release(trial, held)
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
                "name the harness under `[agent] harness`: Harbor reads an unnamed agent "
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
        return AgentConfig(**laid_over(fields, delivered))


def quiet_litellm() -> None:
    """Stop litellm from printing its provider list to stdout over the progress bars.

    litellm prints the list for every model name it does not recognize. Idempotent, and a
    no-op without litellm."""
    try:
        import litellm
    except ImportError:  # pragma: no cover - Harbor depends on it
        return
    litellm.suppress_debug_info = True


async def _run_trial(config: TrialConfig) -> Any:
    """Create and run a Harbor trial.

    Harbor's `Trial` is imported lazily because it pulls in the agents, the backends and the
    registry."""
    from harbor.trial.trial import Trial

    quiet_litellm()
    trial = await Trial.create(config)
    return await trial.run()


def _trial_config(
    task: Path, *, trials_dir: Path, agent: AgentConfig, sandbox: str, verify: bool
) -> TrialConfig:
    """Build Harbor's config for one trial.

    `trial_name` is left empty so Harbor names the trial `<task>__<7 chars>`, as `harbor run`
    does. `delete=True` is Harbor's default, set explicitly because a run opens thousands of
    trials."""
    return TrialConfig(
        task=TaskConfig(path=task),
        trials_dir=trials_dir,
        agent=agent,
        environment=EnvironmentConfig(type=sandbox, delete=True),
        verifier=VerifierConfig(disable=not verify),
    )


def connected(config: TrialConfig, connection: Connection) -> TrialConfig:
    """Return a copy of the trial config that connects to `connection`.

    The connection's env and kwargs override the job's. On a task in allowlist network mode,
    the proxy's host is added to the agent's allowed hosts."""
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
    """Return a copy of the trial config with `held` in its environment and verifier env.

    These values override what the task's `${NAME}` resolves to, because Harbor applies a
    trial's env over the task's, on the sandbox and on every command run in it."""
    if not held:
        return config
    environment = config.environment.model_copy(update={"env": {**config.environment.env, **held}})
    verifier = config.verifier.model_copy(update={"env": {**config.verifier.env, **held}})
    return config.model_copy(update={"environment": environment, "verifier": verifier})


def laid_over(under: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge `over` onto `under`, with `over` winning where both set a key.

    Nested mappings are merged key by key, so a blueprint's `opencode_config` keeps its own
    keys beside the profile's. Values from `over` are deep-copied so trials do not share the
    profile's objects."""
    merged = dict(under)
    for key, value in over.items():
        below = merged.get(key)
        if isinstance(value, Mapping):
            merged[key] = laid_over(below if isinstance(below, Mapping) else {}, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def allowlisted(task: Path) -> bool:
    """Return whether the task's agent phase runs in Harbor's allowlist network mode.

    The mode comes from the `[agent]` override in `task.toml` when it names one, else from the
    `[environment]` baseline. Return False when `task.toml` cannot be read; Harbor reports
    the error when the trial runs."""
    try:
        config = TaskToml.model_validate_toml((task / "task.toml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - not this module's refusal to make
        return False
    phase = config.agent.explicit_phase_policy()
    policy = phase if phase is not None else config.environment.resolve_baseline()
    return policy.network_mode == NetworkMode.ALLOWLIST


def _harness_at_version(harness: str) -> tuple[str, str | None]:
    """Split a harness such as `pi@0.85.1` into Harbor's agent name and its `version` kwarg.

    A name containing a colon (an import path) has no version."""
    name = bare_name(harness)
    version = harness[len(name) + 1 :] if name != harness else ""
    return name, version or None
