"""The proxy over a model this run serves, and the provider case, which records nothing."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any
from urllib.parse import urlsplit

from shipyard import record
from shipyard.config import TINKER
from shipyard.cost import cached
from shipyard.harnesses import slug_of
from shipyard.proxy.client import Proxy
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, SAMPLER_FAILED
from shipyard.types import Connection, Endpoint, Records, Trial

if TYPE_CHECKING:
    from shipyard.config import Blueprint, Rollout
    from shipyard.proxy.wire import Record
    from shipyard.rollout import Rollouts

logger = logging.getLogger(__name__)

#: Sandboxes on this machine, whose containers reach a proxy here by its Docker name.
LOCAL_SANDBOXES = frozenset({"docker", "podman", "apple-container"})
PROXY_LOG = "proxy.log"
TUNNEL_LOG = "tunnel.log"


class ProxyGone(RuntimeError):
    """The proxy this run started has exited."""


class NothingServed(RuntimeError):
    """The proxy served a job nothing: it is dead, or its sampler refuses every request."""

    def __init__(self, message: str, rolled: Rollouts | None = None) -> None:
        super().__init__(message)
        self.rolled = rolled


def placement(rollout: Rollout) -> str:
    """`remote` for a named `endpoint_url`, `local` for a sandbox on this machine, else `tunnel`."""
    if rollout.endpoint_url.strip():
        return "remote"
    return "local" if rollout.sandbox in LOCAL_SANDBOXES else "tunnel"


def backend(environ: Mapping[str, str] | None = None) -> str:
    """The party sampling is billed to: `tinker`, or `tinker@<host>` under a `TINKER_BASE_URL`."""
    environ = os.environ if environ is None else environ
    host = urlsplit(environ.get("TINKER_BASE_URL") or "").hostname
    return f"{TINKER}@{host}" if host else TINKER


def tinker_base_url(environ: Mapping[str, str] | None = None) -> str:
    from tinker.lib.base_url import DEFAULT_BASE_URL

    environ = os.environ if environ is None else environ
    return (environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def start_weights(cfg: Blueprint) -> str | None:
    """The sampler path the proxy starts on; None for a state path, which cannot be sampled."""
    return None if cfg.trains else (cfg.model.from_checkpoint or None)


def endpoint_settings(cfg: Blueprint, *, run: str | None = None) -> dict[str, Any]:
    """The settings passed to `shipyard serve`; with `run`, its Tinker session is tagged."""
    rollout = cfg.rollout
    tagged = {"metadata": {"shipyard_run": run, "shipyard_recipe": cfg.recipe.kind}} if run else {}
    return {
        "model": cfg.model.name,
        "weights": start_weights(cfg),
        "renderer": rollout.renderer or None,
        "bind": rollout.bind,
        "bind_port": rollout.bind_port,
        "temperature": rollout.temperature,
        "top_p": rollout.top_p,
        "top_k": rollout.top_k,
        "max_tokens": rollout.max_tokens,
        "max_context": rollout.max_context or None,
        "fill_context": rollout.fill_context,
        **tagged,
    }


@dataclass(frozen=True)
class ProviderEndpoint(Endpoint):
    """A model served elsewhere: the harness dials it with its own credentials."""

    model: str | None

    async def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def connection(self, trial: Trial) -> Connection:
        return Connection(model=self.model)

    async def serve(self, path: str | None) -> None:
        raise RuntimeError(f"{self.model} is served by its provider; there are no weights to point")

    async def record(self, job: str, trials: Sequence[Trial]) -> None:
        return None


def endpoint_for(cfg: Blueprint, directory: Path) -> Endpoint:
    """The proxy over the run's weights, or the provider the harness dials itself."""
    return ProxyEndpoint(cfg, directory) if cfg.model.served else ProviderEndpoint(cfg.model.name)


@dataclass
class ProxyEndpoint(Endpoint):
    """The proxy over a Tinker sampler: started when the policy opens, stopped when it closes."""

    config: Blueprint
    directory: Path
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ, repr=False)
    proxy: Proxy | None = None
    told: str | None = None
    placement: str | None = None
    origin: str | None = None
    _logs: list[IO[str]] = field(default_factory=list, repr=False)

    @property
    def budget(self) -> int | None:  # type: ignore[override]
        return self.proxy.budget if self.proxy is not None else None

    @property
    def party(self) -> str:  # type: ignore[override]
        return backend(self.environ)

    async def open(self) -> None:
        """Start the proxy once and wait for `/healthz`; after that, check it is still up."""
        if self.proxy is not None:
            self.alive()
            return
        cfg, weights = self.config, start_weights(self.config)
        where = self.placement = placement(cfg.rollout)
        if where == "remote":
            self.proxy = Proxy.remote(
                cfg.rollout.endpoint_url,
                self.environ.get(PROXY_TOKEN_ENV) or "",
                self.environ.get(CONTROL_TOKEN_ENV) or None,
            )
            self.origin = self.proxy.origin
            await self.proxy.ready()
            self._note()
            if self.proxy.control_token is not None:
                await self.serve(weights)
            elif weights is not None:
                raise RuntimeError(
                    f"[model] from_checkpoint names {weights}, and the proxy at "
                    f"{cfg.rollout.endpoint_url} opened no control route to this run: set "
                    f"{CONTROL_TOKEN_ENV}, or start it with --weights"
                )
            else:
                self.told = None
            return
        log = self._log(PROXY_LOG)
        settings = endpoint_settings(cfg, run=self.directory.name)
        if where == "local":
            self.proxy = await Proxy.local(
                settings, host=cfg.rollout.host, environ=self.environ, log=log
            )
        else:
            self.proxy = await Proxy.tunnelled(
                settings, environ=self.environ, log=log, tunnel_log=self._log(TUNNEL_LOG)
            )
        self.origin = self.proxy.origin
        self.told = weights
        served = await self.proxy.ready()
        self._note()
        logger.info("the proxy at %s answers (%s), serving %s", self.origin, where, served)

    def _note(self) -> None:
        """Note the proxy's placement and origin on the run's process note."""
        path = self.directory / record.PROCESS
        note = record.read_json(path)
        note["proxy"] = {"placement": self.placement, "origin": self.origin}
        record.write_json(path, note)

    def alive(self) -> None:
        why = self.proxy.gone() if self.proxy is not None else None
        if why:
            raise ProxyGone(f"the proxy is no longer reachable: {why}")

    async def serve(self, path: str | None) -> None:
        """Point the proxy at `path`, None for the base model, and note it as this run's."""
        if self.proxy is None:
            raise RuntimeError("the proxy is not started; there is nothing to point")
        self.alive()
        await self.proxy.swap(path)
        self.told = path

    def connection(self, trial: Trial) -> Connection:
        if self.proxy is None:
            raise RuntimeError("the proxy is not started; nothing can be rolled out yet")
        profile = trial.harness.profile
        address = self.proxy.address_for(trial.name)
        if profile.strip_v1:
            address = address.removesuffix("/v1")
        slug = slug_of(profile)
        env = {
            **(dict(profile.env) if self.config.rollout.fill_context else {}),
            f"{slug.upper()}_BASE_URL": address,
            f"{slug.upper()}_API_KEY": self.proxy.token,
        }
        return Connection(
            model=f"{slug}/{self.config.model.name}",
            address=address,
            token=self.proxy.token,
            env=env,
            kwargs=dict(profile.kwargs),
            host=self.proxy.host or None,
        )

    async def record(self, job: str, trials: Sequence[Trial]) -> Records:
        """Drain the trials' records, a `requests.jsonl` row each; refuse a path not this run's."""
        proxy = self.proxy
        if proxy is None:
            raise RuntimeError("nothing was served; there are no records to take")
        taken: Records = {}
        for trial in trials:
            try:
                taken[trial.name] = records = await proxy.records(trial.name)
            except Exception:
                self.alive()  # a fetch that failed because the proxy died says so
                raise
            for item in records:
                record.append(self.directory / record.REQUESTS, _row(job, trial.name, item))
        answered_by_ours(job, taken, self.told)
        return taken

    def check(self, job: str, records: Records) -> None:
        """`ProxyGone` if the proxy died under the batch, `NothingServed` if it served nothing."""
        self.alive()
        why = served_nothing(records)
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


def served_nothing(taken: Mapping[str, Sequence[Record]]) -> str | None:
    """Why the proxy served a job nothing, or None when one request was answered or refused."""
    if not taken:
        return None
    made = [item for records in taken.values() for item in records]
    if not made:
        return "no trial reached the proxy; the endpoint is dead or unreachable"
    failed = sorted({item.error or "" for item in made})
    if not all(error.startswith(SAMPLER_FAILED) for error in failed):
        return None
    return (
        f"every request failed at the sampler ({', '.join(failed)}); the weights it serves "
        "are gone or the backend refuses them"
    )


def _row(job: str, trial: str, item: Record) -> dict[str, Any]:
    return {
        "job": job,
        "trial": trial,
        "seq": item.seq,
        "prompt_tokens": len(item.prompt_token_ids),
        "cached_tokens": cached(item),
        "completion_tokens": len(item.completion_token_ids),
        "stop_reason": item.stop_reason,
        "sample_ms": round(float(item.sample_ms), 1),
        "served": item.served,
        "request_id": item.request_id,
        "bridged": item.bridged,
        "error": item.error,
    }


def answered_by_ours(job: str, taken: Mapping[str, Sequence[Record]], told: str | None) -> None:
    """Raise unless every record of `job` was served from the path this run pointed at."""
    answered = {item.served for records in taken.values() for item in records if item.served}
    if answered - {told}:
        raise ValueError(
            f"job {job} was served from {sorted(answered)}, but this run pointed its proxy "
            f"at {told}. Something re-pointed it while the batch was in flight."
        )
