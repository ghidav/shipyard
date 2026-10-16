"""Rollout endpoints: a proxy over the weights this run serves, or a provider's model."""

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
    from shipyard.types import Job

logger = logging.getLogger(__name__)

#: Sandboxes that run on this machine, whose containers reach a local proxy by its Docker name.
LOCAL_SANDBOXES = frozenset({"docker", "podman", "apple-container"})
PROXY_LOG = "proxy.log"
TUNNEL_LOG = "tunnel.log"


class ProxyGone(RuntimeError):
    """Raised when the proxy this run uses is no longer reachable."""


class NothingServed(RuntimeError):
    """Raised when the proxy served a job nothing.

    Either no trial reached the proxy, or its sampler refused every request."""

    def __init__(self, message: str, job: Job | None = None) -> None:
        super().__init__(message)
        self.job = job


def placement(rollout: Rollout) -> str:
    """Return where the proxy runs: `remote`, `local` or `tunnel`.

    `remote` when `endpoint_url` is set, `local` for a sandbox on this machine, else `tunnel`."""
    if rollout.endpoint_url.strip():
        return "remote"
    return "local" if rollout.sandbox in LOCAL_SANDBOXES else "tunnel"


def backend(environ: Mapping[str, str] | None = None) -> str:
    """Return the billing party: `tinker`, or `tinker@<host>` when `TINKER_BASE_URL` is set."""
    environ = os.environ if environ is None else environ
    host = urlsplit(environ.get("TINKER_BASE_URL") or "").hostname
    return f"{TINKER}@{host}" if host else TINKER


def tinker_base_url(environ: Mapping[str, str] | None = None) -> str:
    from tinker.lib.base_url import DEFAULT_BASE_URL

    environ = os.environ if environ is None else environ
    return (environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def start_weights(cfg: Blueprint) -> str | None:
    """Return the sampler path the proxy starts on, or None for the base model.

    A training run starts on None: its `from_checkpoint` is a state path, which cannot be
    sampled."""
    return None if cfg.trains else (cfg.agent.from_checkpoint or None)


def endpoint_settings(cfg: Blueprint, *, run: str | None = None) -> dict[str, Any]:
    """Return the settings passed to `shipyard serve`.

    With `run`, the Tinker session is tagged with the run name and recipe."""
    rollout = cfg.rollout
    tagged = {"metadata": {"shipyard_run": run, "shipyard_recipe": cfg.recipe.kind}} if run else {}
    return {
        "model": cfg.agent.model,
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
    """A model hosted by its provider, which the harness calls with its own credentials.

    No calls are recorded."""

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
    """Return a `ProxyEndpoint` when the run serves its own model, else a `ProviderEndpoint`."""
    return ProxyEndpoint(cfg, directory) if cfg.agent.served else ProviderEndpoint(cfg.agent.model)


@dataclass
class ProxyEndpoint(Endpoint):
    """The proxy in front of a Tinker sampler, opened and closed with the policy."""

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
        """Start or connect to the proxy and wait for its `/healthz`.

        Later calls only check that the proxy is still reachable. Raise `RuntimeError` if the
        run starts on a checkpoint and a remote proxy offers no control route to point it."""
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
                    f"[agent] from_checkpoint names {weights}, and the proxy at "
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
        """Record the proxy's placement and origin in the run's `process.json`."""
        path = self.directory / record.PROCESS
        note = record.read_json(path)
        note["proxy"] = {"placement": self.placement, "origin": self.origin}
        record.write_json(path, note)

    def alive(self) -> None:
        why = self.proxy.gone() if self.proxy is not None else None
        if why:
            raise ProxyGone(f"the proxy is no longer reachable: {why}")

    async def serve(self, path: str | None) -> None:
        """Point the proxy at the sampler weights at `path`, or at the base model for None.

        The path is remembered so `record` can check that every record was served from it."""
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
            model=f"{slug}/{self.config.agent.model}",
            address=address,
            token=self.proxy.token,
            env=env,
            kwargs=dict(profile.kwargs),
            host=self.proxy.host or None,
        )

    async def record(self, job: str, trials: Sequence[Trial]) -> Records:
        """Drain the trials' records from the proxy and append one `requests.jsonl` row each.

        Raise `ValueError` if any record was served from weights other than the ones this run
        pointed the proxy at."""
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
        """Check that the proxy is still up and served this job.

        Raise `ProxyGone` if it died during the job, or `NothingServed` if it served nothing."""
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
    """Return why the proxy served a job nothing, or None if it served the job.

    A job was served nothing when no trial reached the proxy or every request failed at the
    sampler. A request the proxy itself refused counts as served."""
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
    """Raise `ValueError` unless every record of `job` was served from `told`.

    `told` is the path this run pointed the proxy at."""
    answered = {item.served for records in taken.values() for item in records if item.served}
    if answered - {told}:
        raise ValueError(
            f"job {job} was served from {sorted(answered)}, but this run pointed its proxy "
            f"at {told}. Something re-pointed it while the batch was in flight."
        )
