from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from harness_proxy.client import Proxy
from harness_proxy.wire import CONTROL_TOKEN_ENV, Served
from shipyard.groups import Connection, Where
from shipyard.models.clients.tinker import tagged
from shipyard.records import Records, served_nothing

if TYPE_CHECKING:
    from shipyard.config import Config, Rollout
    from shipyard.groups import Trial

logger = logging.getLogger(__name__)

#: Sandboxes that run on this machine, whose containers reach a local proxy by its Docker name.
LOCAL_SANDBOXES = frozenset({"docker", "podman", "apple-container"})
#: What the proxy and its tunnel print, kept in the run directory.
PROXY_LOG = "proxy.log"
#: The token a preflight's harness is given: it makes no call, so no trial is opened for it.
PREFLIGHT_TOKEN = "preflight"
#: The proxy's backend that samples a session's weights published last, so serves one version.
ONE_VERSION = "together"


class ProxyGone(RuntimeError):
    """Raised when the run's proxy is no longer reachable."""


class NothingServed(RuntimeError):
    """Raised when the proxy served a job nothing."""


def placement(rollout: Rollout) -> str:
    """Return where the proxy runs: `remote`, `local` or `tunnel`."""
    if rollout.endpoint_url.strip():
        return "remote"
    return "local" if rollout.sandbox in LOCAL_SANDBOXES else "tunnel"


@dataclass
class ProxyEndpoint:
    """The run's proxy. It serves each model on Tinker or Together in a lane of its own, which
    `lane` makes and each trial is opened on. It starts once every model's session has opened,
    and stops with the run."""

    config: Config
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
        sampling: Mapping[str, Any] | None = None,
    ) -> LaneEndpoint:
        """Make the lane `name`, which serves `load`, a model's name or a `tinker://` path, and
        return what a trial's harness dials to reach it. `opened`, a model the run opens,
        knows a path's base model once open; `renderers` are the ones its calls may name,
        `max_context` caps what one trial may sample, its model's context if None, and
        `sampling` is what every call is sampled with, over the harness's own."""
        if name in self.lanes:
            raise ValueError(f"two models would serve in the proxy's lane {name!r}: rename one")
        made = LaneEndpoint(
            self, name, load, opened, list(renderers), max_context, dict(sampling or {})
        )
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
            self.proxy = Proxy.remote(
                rollout.endpoint_url, self.environ.get(CONTROL_TOKEN_ENV) or ""
            )
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
            logger.info("lane %s serves %s", name, one.model)

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
            proxy.stop()
        for opened in self._logs:
            opened.close()
        self._logs.clear()


@dataclass
class LaneEndpoint:
    """A model the run's proxy serves, in a lane of its own: it opens each trial `rollout` runs
    of the model, on the weights the model serves then, and closes it for its records."""

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
    #: What the proxy samples every call with, over the harness's own: temperature, top_p,
    #: top_k, max_tokens and fill_context.
    sampling: dict[str, Any] = field(default_factory=dict)
    #: The proxy's backend settings for the lane beyond Tinker's: for Together, the backend
    #: and the session, which the model's client opens before the proxy starts.
    backend: dict[str, str] = field(default_factory=dict)
    #: The weights the model serves now, which each trial opened from now on samples; None
    #: for the base model.
    told: str | None = None
    #: The trials open on the lane, by name.
    trials: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Open trials on a `tinker://` path from the start."""
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
        """The lane, as the proxy is started with it: its name and model, its own renderers,
        context and sampling, and its backend beyond Tinker's. Each trial names its weights."""
        own = {"renderer": self.renderers, "max_context": self.max_context}
        return {
            **self.sampling,
            "metadata": tagged(self.proxy.directory.name),
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

    def publishing(self) -> None:
        """Raise `RuntimeError` if the model cannot publish new weights now: Together serves
        the weights a session published last, so a trial open on the old ones would be
        answered by the new."""
        if self.backend.get("backend") == ONE_VERSION and self.trials:
            raise RuntimeError(
                f"[models.{self.name}] serves on Together, which samples the weights published "
                f"last, and {len(self.trials)} of its trials are open on the ones before: "
                "publish once they close"
            )

    def serve(self, path: str | None) -> None:
        """Open the lane's trials from now on at the sampler weights at `path`, or at its base
        model for None; a trial open already keeps the weights it was opened on."""
        self.told = path

    async def open(self, trial: Trial) -> Connection:
        """Open `trial` on the proxy, on the weights the model serves now and rendered with its
        renderer, and return what its harness dials."""
        self.proxy.alive()
        try:
            token = await self.proxy.client().open(
                trial.name, lane=self.name, renderer=trial.renderer, weights=self.told
            )
        except Exception:
            self.proxy.alive()  # an open that failed because the proxy died says so
            raise
        self.trials.add(trial.name)
        return self.connection(trial, token)

    def connection(self, trial: Trial, token: str = PREFLIGHT_TOKEN) -> Connection:
        """Return what a trial's harness dials: the proxy's address, as the harness reaches
        it, with `token`, the one its trial was opened with."""
        proxy = self.proxy.client()
        where = Where(
            proxy.address, local=proxy.local_address, token=token, host=proxy.host or None
        )
        return trial.harness.dial(
            where, self.model, fill_context=bool(self.sampling.get("fill_context"))
        )

    async def close(self, trial: Trial) -> Records:
        """Close `trial` on the proxy and return its records."""
        try:
            return Records(await self.proxy.client().close(trial.name))
        except Exception:
            self.proxy.alive()  # a close that failed because the proxy died says so
            raise
        finally:
            self.trials.discard(trial.name)

    def check(self, job: str, trials: Sequence[Trial]) -> None:
        """Raise if the proxy died during the job or served it nothing."""
        self.proxy.alive()
        why = served_nothing(one.records for one in trials)
        if why:
            raise NothingServed(f"job {job}: {why}")


def serves(origin: str, lane: LaneEndpoint, served: Served | None) -> None:
    """Raise `RuntimeError` unless the proxy at `origin`, which reports `served` for `lane`,
    serves it as the run made it: its model, rendered with each of its renderers."""
    if served is None:
        raise RuntimeError(
            f"the proxy at {origin} serves no lane for {lane.name}: start it with a lane for "
            "each model on Tinker or Together that run.toml names"
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
            f"the proxy at {origin} renders lane {lane.name} with {has}, and [models.{lane.name}] "
            f"renderers names {', '.join(missing)}: start it with each of "
            f"{', '.join(lane.renderers)}"
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

    async def open(self, trial: Trial) -> Connection:
        """Return what `trial`'s harness dials: the model under its provider. Nothing opens."""
        return self.connection(trial)

    def connection(self, trial: Trial, token: str = PREFLIGHT_TOKEN) -> Connection:
        """Return the model under its provider, for a harness to call with its own credentials."""
        return Connection(model=self.routed)

    async def close(self, trial: Trial) -> Records:
        """Return the records of a trial its provider served: none, as it keeps none for the run."""
        return Records.unrecorded()

    def check(self, job: str, trials: Sequence[Trial]) -> None:
        """Do nothing: the run keeps no proxy for a provider's model, so none can fail it."""
        return None
