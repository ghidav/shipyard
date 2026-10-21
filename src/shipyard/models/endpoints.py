from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from harness_proxy.client import Proxy
from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, Served
from shipyard.groups import Connection, Where
from shipyard.models.clients.tinker import tagged
from shipyard.records import Records, served_nothing

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


def placement(rollout: Rollout) -> str:
    """Return where the proxy runs: `remote`, `local` or `tunnel`."""
    if rollout.endpoint_url.strip():
        return "remote"
    return "local" if rollout.sandbox in LOCAL_SANDBOXES else "tunnel"


def sampling_params(config: Blueprint, run: str) -> dict[str, Any]:
    """Return the sampling settings `[rollout]` gives each lane of the proxy the run starts."""
    rollout = config.rollout
    return {
        "temperature": rollout.temperature,
        "top_p": rollout.top_p,
        "top_k": rollout.top_k,
        "max_tokens": rollout.max_tokens,
        "fill_context": rollout.fill_context,
        "metadata": tagged(run),
    }


@dataclass
class ProxyEndpoint:
    """The run's proxy. It serves each model on Tinker or Together in a lane of its own, which
    `lane` makes and a trial's address names. It starts once every model's session has opened,
    and stops with the run."""

    config: Blueprint
    directory: Path
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ, repr=False)
    proxy: Proxy | None = None
    #: The models it serves, by the name of each one's lane.
    lanes: dict[str, LaneEndpoint] = field(default_factory=dict)
    _logs: list[IO[str]] = field(default_factory=list, repr=False)

    def lane(
        self,
        name: str,
        load: str,
        *,
        opened: Any = None,
        renderers: Sequence[str] = (),
        max_context: int | None = None,
    ) -> LaneEndpoint:
        """Make the lane `name`, which serves `load`, a model's name or a `tinker://` path, and
        return what a trial's harness dials to reach it. `opened`, a model the run opens,
        knows a path's base model once open; `renderers` are the ones its calls may name, and
        `max_context` caps what one trial may sample, its model's context if None."""
        if name in self.lanes:
            raise ValueError(f"two models would serve in the proxy's lane {name!r}: rename one")
        made = LaneEndpoint(self, name, load, opened, list(renderers), max_context)
        self.lanes[name] = made
        return made

    async def __aenter__(self) -> ProxyEndpoint:
        try:
            await self.open()
        except BaseException:
            self.close()
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()

    async def open(self) -> None:
        """Start the proxy with every lane, or connect to the remote one, and wait until it
        answers. Raise `RuntimeError` unless it serves each lane as the run made it."""
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
            given = {
                "bind": rollout.bind,
                "bind_port": rollout.bind_port,
                "lanes": [one.document for one in self.lanes.values()],
            }
            log = self._log(PROXY_LOG)
            if where == "local":
                self.proxy = await Proxy.local(
                    given, host=rollout.host, environ=self.environ, log=log
                )
            else:
                self.proxy = await Proxy.tunnelled(given, environ=self.environ, log=log)
        served = await self.proxy.ready()
        for one in self.lanes.values():
            serves(self.proxy.origin, one, served.get(one.name))
        logger.info("the proxy at %s answers (%s)", self.proxy.origin, where)
        for name, one in served.items():
            logger.info("lane %s serves %s", name, one.serving)

    def client(self) -> Proxy:
        """Return the proxy's client; `RuntimeError` before the proxy opens."""
        if self.proxy is None:
            raise RuntimeError("the run's proxy is not open: it opens when the run starts")
        return self.proxy

    def alive(self) -> None:
        """Raise `ProxyGone` if the proxy is no longer reachable."""
        why = self.proxy.gone() if self.proxy is not None else None
        if why:
            raise ProxyGone(f"the proxy is no longer reachable: {why}")

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


@dataclass
class LaneEndpoint:
    """A model the run's proxy serves, in a lane of its own: what a trial's harness dials when
    `rollout` runs it, and where the run points the model's weights."""

    proxy: ProxyEndpoint
    #: The lane's name, which a trial's address gives.
    name: str
    #: What it loads: a model's name, or a `tinker://` path.
    load: str
    #: The model the run opened for it, the agent's session or a model on Tinker: it knows a
    #: path's base model once open, and who the lane's tokens are billed to.
    opened: Any = None
    #: The renderers its calls may name; none for its model's own.
    renderers: list[str] = field(default_factory=list)
    #: Most tokens one trial may sample; None for its model's context.
    max_context: int | None = None
    #: The proxy's backend settings for the lane beyond Tinker's: for Together, the backend
    #: and the session, which the model's client opens before the proxy starts.
    backend: dict[str, str] = field(default_factory=dict)
    #: The weights the run last pointed the lane at; None for the base model.
    told: str | None = None

    def __post_init__(self) -> None:
        """Serve a `tinker://` path from the start."""
        if self.told is None and self.load.startswith("tinker://"):
            self.told = self.load

    @property
    def provider(self) -> str | None:
        """Who its tokens are billed to: the model it serves', once that is opened."""
        return getattr(self.opened, "provider", None)

    @property
    def kind(self) -> str | None:
        """The backend it reaches, as a job's row names it: the model it serves'."""
        return getattr(self.opened, "kind", None)

    @property
    def model(self) -> str:
        """The base model it serves, which its tokens are priced as."""
        return str(getattr(self.opened, "model", None) or self.load)

    @property
    def document(self) -> dict[str, Any]:
        """The lane, as the proxy is started with it: its name, model and weights, its own
        renderers and context, the run's sampling, and its backend beyond Tinker's."""
        own = {"weights": self.told, "renderer": self.renderers, "max_context": self.max_context}
        return {
            **sampling_params(self.proxy.config, self.proxy.directory.name),
            "name": self.name,
            "model": self.model,
            **{key: value for key, value in own.items() if value},
            **self.backend,
        }

    @property
    def budget(self) -> int | None:
        """Most tokens one trial may sample, as the proxy reports it for the lane."""
        proxy = self.proxy.proxy
        served = proxy.lanes.get(self.name) if proxy is not None else None
        return served.budget if served is not None else None

    async def serve(self, path: str | None) -> None:
        """Point the lane at the sampler weights at `path`, or at its base model for None."""
        self.proxy.alive()
        await self.proxy.client().swap(path, lane=self.name)
        self.told = path

    def connection(self, trial: Trial) -> Connection:
        """Return what a trial's harness dials: the lane's address, naming the trial's
        renderer, and the token, as the harness reaches them."""
        proxy = self.proxy.client()
        address = proxy.address_for(trial.name, trial.renderer, lane=self.name)
        where = Where(
            address,
            # The same address from this machine, for a harness whose calls leave from here.
            local=proxy.loopback + address.removeprefix(proxy.origin),
            token=proxy.token,
            host=proxy.host or None,
        )
        return trial.harness.dial(
            where, self.model, fill_context=self.proxy.config.rollout.fill_context
        )

    async def record(self, job: str, trials: Sequence[Trial]) -> dict[str, list[Record]]:
        """Drain the trials' records from the proxy, by trial name, and check that the lane's
        weights served them."""
        proxy = self.proxy.client()
        taken: dict[str, list[Record]] = {}
        for trial in trials:
            try:
                taken[trial.name] = await proxy.records(trial.name)
            except Exception:
                self.proxy.alive()  # a fetch that failed because the proxy died says so
                raise
        answered_by_ours(job, taken, self.told, lane=self.name)
        return taken

    def check(self, job: str, trials: Sequence[Trial]) -> None:
        """Raise if the proxy died during the job or served it nothing."""
        self.proxy.alive()
        why = served_nothing(one.records for one in trials)
        if why:
            raise NothingServed(f"job {job}: {why}")


def answered_by_ours(
    job: str, taken: Mapping[str, Sequence[Record]], told: str | None, *, lane: str
) -> None:
    """Raise `ValueError` unless every record of `job` was served from `told`, the weights the
    run pointed `lane` at."""
    answered = {name for one in taken.values() for name in Records(one).weights}
    if answered - {told}:
        raise ValueError(
            f"job {job} was served from {sorted(answered)} in lane {lane}, and the run pointed "
            f"that lane at {told}: something re-pointed it while the batch was in flight, or a "
            "remote proxy was started on other weights"
        )


def serves(origin: str, lane: LaneEndpoint, served: Served | None) -> None:
    """Raise `RuntimeError` unless the proxy at `origin`, which reports `served` for `lane`,
    serves it as the run made it: its model, rendered with each of its renderers."""
    if served is None:
        raise RuntimeError(
            f"the proxy at {origin} serves no lane for {lane.name}: start it with a lane for "
            "each model on Tinker the trainer holds"
        )
    if served.model != lane.model:
        raise RuntimeError(
            f"the proxy at {origin} serves {served.model} in lane {lane.name}, and the run "
            f"serves {lane.model} there: start it with that model"
        )
    missing = [one for one in lane.renderers if one not in served.renderers]
    if missing:
        has = ", ".join(served.renderers) or "its model's default alone"
        raise RuntimeError(
            f"the proxy at {origin} renders lane {lane.name} with {has}, and [rollout] renderer "
            f"names {', '.join(missing)}: start it with each of {', '.join(lane.renderers)}"
        )


@dataclass(frozen=True)
class ProviderEndpoint:
    """A model its provider hosts, which a harness calls with its own credentials."""

    model: str
    provider: str

    @property
    def kind(self) -> str:
        """The backend it reaches, as a job's row names it: the provider's API."""
        return "api"

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
