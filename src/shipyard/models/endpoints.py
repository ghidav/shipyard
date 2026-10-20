from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from harness_proxy.client import Proxy
from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, SAMPLER_FAILED
from shipyard.jobs.harnesses import slug_of
from shipyard.models.clients.tinker import backend, tagged

if TYPE_CHECKING:
    from harness_proxy.wire import Record
    from shipyard.config import Blueprint, Rollout
    from shipyard.groups import Trial

logger = logging.getLogger(__name__)

#: Sandboxes that run on this machine, whose containers reach a local proxy by its Docker name.
LOCAL_SANDBOXES = frozenset({"docker", "podman", "apple-container"})
#: What the proxy and its tunnel print, kept in the run directory.
PROXY_LOG = "proxy.log"


class ProxyGone(RuntimeError):
    """Raised when the run's proxy is no longer reachable."""


class NothingServed(RuntimeError):
    """Raised when the proxy served a job nothing."""


@dataclass
class Connection:
    """How a trial's harness reaches the model. For a provider's model, only `model` is set."""

    model: str
    env: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: The proxy's host name, added to the allowed hosts of a task in Harbor's allowlist
    #: network mode.
    host: str | None = None


def placement(rollout: Rollout) -> str:
    """Return where the proxy runs: `remote`, `local` or `tunnel`."""
    if rollout.endpoint_url.strip():
        return "remote"
    return "local" if rollout.sandbox in LOCAL_SANDBOXES else "tunnel"


def settings(config: Blueprint, run: str) -> dict[str, Any]:
    """Return the settings a proxy the run starts is given."""
    rollout = config.rollout
    return {
        "model": config.agent.model,
        "renderer": list(rollout.renderer) or None,
        "bind": rollout.bind,
        "bind_port": rollout.bind_port,
        "temperature": rollout.temperature,
        "top_p": rollout.top_p,
        "top_k": rollout.top_k,
        "max_tokens": rollout.max_tokens,
        "max_context": rollout.max_context or None,
        "fill_context": rollout.fill_context,
        "metadata": tagged(run),
    }


@dataclass
class ProxyEndpoint:
    """The run's proxy, which the agent's harnesses call."""

    config: Blueprint
    directory: Path
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ, repr=False)
    proxy: Proxy | None = None
    #: The weights the run last pointed the proxy at; None for the base model.
    told: str | None = None
    #: The proxy's backend settings beyond Tinker's: for Together, the backend and the
    #: session, which the agent's client opens before the proxy starts.
    backend: dict[str, str] = field(default_factory=dict)
    _logs: list[IO[str]] = field(default_factory=list, repr=False)

    @property
    def budget(self) -> int | None:
        """Most tokens one rollout may sample, as the proxy reports it."""
        return self.proxy.budget if self.proxy is not None else None

    @property
    def provider(self) -> str:
        return self.backend.get("backend") or backend()

    @property
    def model(self) -> str:
        return self.config.agent.model

    async def open(self) -> None:
        """Start the proxy, or connect to the remote one, and wait until it answers."""
        rollout = self.config.rollout
        where = placement(rollout)
        if where == "remote":
            proxy = Proxy.remote(
                rollout.endpoint_url,
                self.environ.get(PROXY_TOKEN_ENV) or "",
                self.environ.get(CONTROL_TOKEN_ENV) or None,
            )
            if proxy.control_token is None:
                raise RuntimeError(
                    f"the proxy at {rollout.endpoint_url} takes its control token to read "
                    f"records and point weights, and {CONTROL_TOKEN_ENV} is unset: export the "
                    "token it was started with"
                )
            self.proxy = proxy
        else:
            given = {**settings(self.config, self.directory.name), **self.backend}
            log = self._log(PROXY_LOG)
            if where == "local":
                self.proxy = await Proxy.local(
                    given, host=rollout.host, environ=self.environ, log=log
                )
            else:
                self.proxy = await Proxy.tunnelled(given, environ=self.environ, log=log)
        self.told = None
        served = await self.proxy.ready()
        renders_each(self.proxy, rollout.renderer)
        logger.info("the proxy at %s answers (%s), serving %s", self.proxy.origin, where, served)

    def alive(self) -> None:
        """Raise `ProxyGone` if the proxy is no longer reachable."""
        why = self.proxy.gone() if self.proxy is not None else None
        if why:
            raise ProxyGone(f"the proxy is no longer reachable: {why}")

    async def serve(self, path: str | None) -> None:
        """Point the proxy at the sampler weights at `path`, or at the base model for None."""
        self.alive()
        await self.proxy.swap(path)
        self.told = path

    def connection(self, trial: Trial) -> Connection:
        """Return what a trial's harness dials: its address, naming its renderer, and the token."""
        profile = trial.harness.profile
        address = self.proxy.address_for(trial.name, trial.renderer)
        if profile.in_process is not None:
            # Its calls leave from this machine, not the container: the proxy's own address.
            address = self.proxy.loopback + address.removeprefix(self.proxy.origin)
        if profile.strip_v1:
            address = address.removesuffix("/v1")
        slug = slug_of(profile)
        env = {
            **(dict(profile.env) if self.config.rollout.fill_context else {}),
            f"{slug.upper()}_BASE_URL": address,
            f"{slug.upper()}_API_KEY": self.proxy.token,
        }
        kwargs = dict(profile.kwargs)
        if profile.in_process is not None:
            kwargs |= profile.in_process(address, self.proxy.token)
        return Connection(
            model=f"{slug}/{self.config.agent.model}",
            env=env,
            kwargs=kwargs,
            host=self.proxy.host or None,
        )

    async def record(self, job: str, trials: Sequence[Trial]) -> dict[str, list[Record]]:
        """Drain the trials' records from the proxy, by trial name, and check who served them."""
        proxy = self.proxy
        taken: dict[str, list[Record]] = {}
        for trial in trials:
            try:
                taken[trial.name] = await proxy.records(trial.name)
            except Exception:
                self.alive()  # a fetch that failed because the proxy died says so
                raise
        answered_by_ours(job, taken, self.told)
        return taken

    def check(self, job: str, trials: Sequence[Trial]) -> None:
        """Raise if the proxy died during the job or served it nothing."""
        self.alive()
        why = served_nothing(trials)
        if why:
            raise NothingServed(f"job {job}: {why}")

    def _log(self, name: str) -> IO[str]:
        opened = (self.directory / name).open("a", encoding="utf-8")
        self._logs.append(opened)
        return opened

    def close(self) -> None:
        proxy, self.proxy = self.proxy, None
        if proxy is not None:
            proxy.close()
        for opened in self._logs:
            opened.close()
        self._logs.clear()


def served_nothing(trials: Sequence[Trial]) -> str | None:
    """Return why the proxy served the trials nothing, or None if it served them."""
    if not trials:
        return None
    made = [item for one in trials for item in one.records or ()]
    if not made:
        return "no trial reached the proxy; the endpoint is dead or unreachable"
    failed = sorted({item.error or "" for item in made})
    if not all(error.startswith(SAMPLER_FAILED) for error in failed):
        return None
    return (
        f"every request failed at the sampler ({', '.join(failed)}); the weights it serves "
        "are gone or the backend refuses them"
    )


def answered_by_ours(job: str, taken: Mapping[str, Sequence[Record]], told: str | None) -> None:
    """Raise `ValueError` unless every record of `job` was served from `told`."""
    answered = {item.served for records in taken.values() for item in records if item.served}
    if answered - {told}:
        raise ValueError(
            f"job {job} was served from {sorted(answered)}, but this run pointed its proxy "
            f"at {told}. Something re-pointed it while the batch was in flight."
        )


def renders_each(proxy: Any, named: Sequence[str]) -> None:
    """Raise `RuntimeError` unless the proxy renders with each renderer `named`."""
    known = proxy.renderers
    missing = [one for one in named if one not in (known or ())]
    if not missing:
        return
    has = (
        "does not say which renderers it renders with"
        if known is None
        else f"renders with {', '.join(known) or 'its model default alone'}"
    )
    raise RuntimeError(
        f"the proxy at {proxy.origin} {has}, and [rollout] renderer names {', '.join(missing)}: "
        f"start it with a --renderer for each of {', '.join(named)}"
    )


#: Calls to a provider in flight at once, and LiteLLM's retries of one that failed.
PROVIDER_CALLS, PROVIDER_RETRIES = 8, 4


@dataclass(frozen=True)
class ProviderEndpoint:
    """A model its provider hosts, which a harness calls with its own credentials."""

    model: str
    provider: str
    #: Calls `sample` has in flight at once, so a search's burst meets no rate limit at once.
    _calls: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(PROVIDER_CALLS), compare=False, repr=False
    )

    @property
    def budget(self) -> None:
        """None: the provider's limits apply, not a budget of the run's."""
        return None

    @property
    def routed(self) -> str:
        """The model as Harbor's agents and LiteLLM name it: `provider/model`."""
        named = self.model.startswith(f"{self.provider}/")
        return self.model if named else f"{self.provider}/{self.model}"

    def connection(self, trial: Trial) -> Connection:
        """Return the model under its provider, for a harness to call with its own credentials."""
        return Connection(model=self.routed)

    async def record(self, job: str, trials: Sequence[Trial]) -> None:
        """Return None: the provider keeps no records for the run."""
        return None

    def check(self, job: str, trials: Sequence[Trial]) -> None:
        """Do nothing: the run keeps no proxy for a provider's model, so none can fail it."""
        return None

    async def sample(
        self, messages: Sequence[Mapping[str, Any]], **params: Any
    ) -> tuple[str, dict[str, int]]:
        """Return the model's reply to `messages`, through LiteLLM, and the tokens it took."""
        import litellm

        asked = {"num_retries": PROVIDER_RETRIES, **params}
        async with self._calls:
            reply = await litellm.acompletion(model=self.routed, messages=list(messages), **asked)
        usage = reply.usage
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cache = min(int(getattr(details, "cached_tokens", 0) or 0), prompt)
        took = {
            "prefill": prompt - cache,
            "cache": cache,
            "sample": int(getattr(usage, "completion_tokens", 0) or 0),
        }
        return reply.choices[0].message.content or "", took
