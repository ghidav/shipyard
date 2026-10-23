from __future__ import annotations

import asyncio
import copy
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from harbor.models.task.config import NetworkMode
from harbor.models.task.config import TaskConfig as TaskToml
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)

from shipyard.groups import Group, Trial, grouped
from shipyard.jobs.admit import judge
from shipyard.jobs.harnesses import Harness
from shipyard.jobs.hooks import hooks_of
from shipyard.jobs.modules import Modules, deliver
from shipyard.jobs.progress import Bars
from shipyard.jobs.secrets import Secrets, scrub
from shipyard.records import Records

if TYPE_CHECKING:
    from shipyard.config import Rollout
    from shipyard.groups import Connection
    from shipyard.run.logs import Logs

logger = logging.getLogger(__name__)

#: Modal's image-builder variable, and the version that builds a Dockerfile as written. The
#: legacy builder injects `python -m pip` into images that have no Python.
MODAL_IMAGE_BUILDER = "MODAL_IMAGE_BUILDER_VERSION"
MODAL_IMAGE_BUILDER_VERSION = "2025.06"
#: Directory under the job's directory, next to its trials, where its modules are delivered.
DELIVERED = "modules"
#: The sandboxes shipyard brings beside Harbor's own: the environment Harbor imports for each.
ENVIRONMENTS = {"together": "shipyard.jobs.together:TogetherEnvironment"}
#: The package a sandbox's environment starts it with, by sandbox, and the install that brings
#: it: an extra installs it, and the environment imports it only as a trial starts.
PACKAGES = {
    "modal": (
        "modal",
        'uv pip install "harbor[modal]", or uv tool install shipyard --with "harbor[modal]"',
    ),
    "together": (
        "together_sandbox",
        'uv pip install "shipyard[together]", or uv tool install "shipyard[together]"',
    ),
}

#: Runs one trial: Harbor's `Trial` by default, or a stand-in in tests.
TrialRunner = Callable[[TrialConfig], Awaitable[Any]]


class Endpoint(Protocol):
    """What a trial's harness dials, and who keeps its records: the run's proxy, which opens
    each trial and hands its records back when it closes, or a provider, which keeps none."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def open(self, trial: Trial) -> Connection: ...

    def connection(self, trial: Trial) -> Connection: ...

    async def close(self, trial: Trial) -> Records: ...

    def check(self, job: str, trials: Sequence[Trial]) -> None: ...


@dataclass
class Job:
    name: str
    purpose: str = "training"
    step: int | None = None
    modules: Modules | None = None
    #: The harness its trials run, such as `pi@0.85.1`, with its env and kwargs.
    harness: Harness | None = None
    #: The role of the model its trials run, which its spend is filed under.
    role: str = ""
    #: The renderer the proxy renders its trials' calls with; None for the model's own.
    renderer: str | None = None
    #: Grade a trial cut at its timeout by the verifier's reward, rather than mask it.
    grade_timeouts: bool = False
    trials: list[Trial] = field(default_factory=list)


class HarborEngine:
    def __init__(
        self,
        rollout: Rollout,
        *,
        logs: Logs | None = None,
        secrets: Secrets | None = None,
        grade_timeouts: bool = False,
        run_trial: TrialRunner | None = None,
    ) -> None:
        self.sandbox = rollout.sandbox
        self.jobs_dir = Path(rollout.jobs_dir)
        self.setup_timeout, self.timeout = rollout.setup_timeout, rollout.timeout
        #: The run's logs, which name each rollout's job and keep its rows.
        self.logs = logs
        self.secrets = Secrets() if secrets is None else secrets
        #: Grade a trial cut at its timeout by the verifier's reward, rather than mask it.
        self.grade_timeouts = grade_timeouts
        self.run_trial = run_trial
        if run_trial is None:  # Harbor's trials start the sandbox; a stand-in starts none
            installed(self.sandbox)
        self.progress = Bars()
        self._limit = asyncio.Semaphore(rollout.concurrency)

    async def rollout(
        self,
        tasks: Sequence[Path],
        *,
        attempts: int,
        harness: Harness,
        endpoint: Endpoint,
        renderer: str | None = None,
        modules: Modules | None = None,
        purpose: str = "training",
        step: int | None = None,
        role: str = "",
    ) -> list[Group]:
        """Run each task `attempts` times through `harness`, as one job, and return one group
        per task, in the order of `tasks`. The trials run the model `role` names, which
        `endpoint` reaches, its calls rendered with `renderer`, carrying `modules`. The job is
        named, judged and logged before this returns or raises. Raise `ValueError` before any
        trial runs for a rollout no trial could run, and `ProxyGone` or `NothingServed` once
        the job has ended, when the endpoint died or served it nothing."""
        if self.logs is None:
            raise RuntimeError("a rollout writes its job's rows: build the engine with `logs`")
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1; got {attempts}")
        if not tasks:
            return []
        job = Job(
            self.logs.reserve(),
            purpose=purpose,
            step=step,
            modules=modules,
            harness=harness,
            renderer=renderer,
            grade_timeouts=self.grade_timeouts,
            role=role,
        )
        await self.run(job, list(tasks), attempts, endpoint=endpoint)
        self.logs.job(job, endpoint)
        endpoint.check(job.name, job.trials)
        return grouped(job.trials)

    async def run(
        self, job: Job, tasks: Sequence[Path], attempts: int, *, endpoint: Endpoint
    ) -> Job:
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
        agent = self._agent_config(harness, delivered, job)
        plan = [Path(task) for task in tasks for _ in range(attempts)]
        trials_dir.mkdir(parents=True, exist_ok=True)
        if not harness.runs_without_network:
            cut = sorted({t.name for t in plan if network_mode(t) == NetworkMode.NO_NETWORK})
            if cut:
                raise ValueError(
                    f"job {job.name}: {', '.join(cut)} run their agent with no network, so "
                    f"{harness.bare}, which calls the model from the container, cannot reach it"
                )
        first = _trial_config(plan[0], trials_dir=trials_dir, agent=agent, sandbox=self.sandbox)
        preflight(first, endpoint, harness, job.name)
        if self.sandbox == "modal":
            os.environ.setdefault(MODAL_IMAGE_BUILDER, MODAL_IMAGE_BUILDER_VERSION)
        run_one = self.run_trial if self.run_trial is not None else _run_trial

        async def one(task: Path) -> Trial:
            config = _trial_config(task, trials_dir=trials_dir, agent=agent, sandbox=self.sandbox)
            trial = Trial(
                job.name,
                task,
                harness,
                trials_dir / config.trial_name,
                renderer=job.renderer,
                role=job.role,
            )
            async with self._limit:
                held = await self.secrets.acquire(trial)
                for name in held:  # Harbor fills the task's `${NAME}` from here first
                    os.environ.setdefault(name, "")
                try:
                    # An open that fails stops the run: the proxy is gone, or cannot serve
                    # the weights.
                    dialled = connected(config, await endpoint.open(trial))
                    try:
                        await run_one(holding(dialled, held))
                    except asyncio.CancelledError:
                        current = asyncio.current_task()
                        if current is None or current.cancelling():
                            raise  # ours: the run is stopping, and its proxy with it
                        # A cancellation not ours fails only this trial.
                        logger.warning(
                            "trial %s ended in a cancellation it did not own", trial.name
                        )
                    except Exception:
                        logger.exception("trial %s could not run", trial.name)
                    trial.records = await endpoint.close(trial)
                finally:
                    if held:
                        try:
                            await asyncio.to_thread(scrub, trial.dir, held)
                        finally:
                            await self.secrets.release(trial, held)
            self.progress.finished(job.name)
            return trial

        self.progress.started(job.name, trials=len(plan))
        running = [asyncio.ensure_future(one(task)) for task in plan]
        try:
            job.trials = list(await asyncio.gather(*running))
        except BaseException:
            # A trial that could not open stops the run, so the trials beside it stop too,
            # rather than run on for a run that is ending.
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise
        judge(job)
        self.progress.done(job.name, masked=sum(not one.graded for one in job.trials))
        return job

    def close(self) -> None:
        self.progress.close()

    def _agent_config(
        self, harness: Harness, delivered: Mapping[str, Any], job: Job
    ) -> AgentConfig:
        version = harness.version
        fields: dict[str, Any] = {
            "name": harness.bare,
            "env": dict(harness.env),
            "kwargs": {**({"version": version} if version else {}), **harness.kwargs},
            "override_setup_timeout_sec": self.setup_timeout,
            "override_timeout_sec": self.timeout,
        }
        return AgentConfig(**laid_over(fields, delivered))


def quiet_litellm() -> None:
    import litellm

    litellm.suppress_debug_info = True


def installed(sandbox: str) -> None:
    """Raise `ValueError` when the package `sandbox` is started with is not installed, which
    Harbor would find out only as each trial starts, once the run has opened its models."""
    if sandbox not in PACKAGES:
        return
    package, install = PACKAGES[sandbox]
    if find_spec(package) is None:
        raise ValueError(f'[rollout] sandbox = "{sandbox}" needs the {package} package: {install}')


def preflight(config: TrialConfig, endpoint: Endpoint, harness: Harness, job: str) -> None:
    """Run Harbor's own check of the agent config a trial would run with."""
    from harbor.agents.factory import AgentFactory

    probe = Trial(job, Path(str(config.task.path)), harness, Path(config.trials_dir) / "preflight")
    agent = connected(config, endpoint.connection(probe)).agent
    try:
        AgentFactory.run_preflight(agent)
    except ValueError as refused:
        raise ValueError(f"the {harness.bare} harness refuses its config: {refused}") from None


async def _run_trial(config: TrialConfig) -> Any:
    """Create and run a Harbor trial, with its task's hooks."""
    from harbor.trial.trial import Trial as HarborTrial

    quiet_litellm()
    trial = await HarborTrial.create(config)
    for event, hook in hooks_of(Path(str(config.task.path))).items():
        trial.add_hook(event, hook)
    return await trial.run()


def _trial_config(task: Path, *, trials_dir: Path, agent: AgentConfig, sandbox: str) -> TrialConfig:
    """Build Harbor's config for one trial, in Harbor's sandbox of that name or shipyard's."""
    imported = ENVIRONMENTS.get(sandbox)
    where = {"type": sandbox} if imported is None else {"import_path": imported}
    return TrialConfig(
        task=TaskConfig(path=task),
        trials_dir=trials_dir,
        agent=agent,
        environment=EnvironmentConfig(**where, delete=True),
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
