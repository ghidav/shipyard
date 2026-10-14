"""The endpoints: the proxy this run starts and drains for a model it serves, and
the provider case, which records nothing."""

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
from shipyard.proxy.client import Proxy
from shipyard.proxy.profiles import bare_name, slug_of
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, SAMPLER_FAILED
from shipyard.types import Connection, Endpoint, Harvest, Trial

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
    """A served job the proxy served nothing for: the endpoint is dead, or its sampler
    refuses every request."""

    def __init__(self, message: str, rolled: Rollouts | None = None) -> None:
        super().__init__(message)
        self.rolled = rolled


def placement(rollout: Rollout) -> str:
    """`remote` for a named `endpoint_url`, `local` for a sandbox on this machine, else
    `tunnel`."""
    if rollout.endpoint_url.strip():
        return "remote"
    return "local" if rollout.sandbox in LOCAL_SANDBOXES else "tunnel"


def backend(environ: Mapping[str, str] | None = None) -> str:
    """The party sampling is billed to: `tinker`, or `tinker@<host>` under a
    `TINKER_BASE_URL`."""
    environ = os.environ if environ is None else environ
    host = urlsplit(environ.get("TINKER_BASE_URL") or "").hostname
    return f"{TINKER}@{host}" if host else TINKER


def tinker_base_url(environ: Mapping[str, str] | None = None) -> str:
    from tinker.lib.base_url import DEFAULT_BASE_URL

    environ = os.environ if environ is None else environ
    return (environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def start_weights(cfg: Blueprint) -> str | None:
    """The checkpoint the proxy starts on. A training run's `from_checkpoint` is a state
    path Tinker cannot sample, so it starts on none and the first step points it."""
    return None if cfg.trains else (cfg.model.from_checkpoint or None)


def endpoint_settings(cfg: Blueprint, *, run: str | None = None) -> dict[str, Any]:
    """The settings passed to `shipyard serve`. With `run`, the proxy's Tinker session is
    tagged with it."""
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


@dataclass
class ProxyEndpoint(Endpoint):
    """The proxy over a Tinker sampler. The first job starts it, the run stops it, and it
    remembers the path it was pointed at, so a batch answered by other weights is refused."""

    config: Blueprint
    directory: Path
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ, repr=False)
    proxy: Proxy | None = None
    told: str | None = None
    pointed: bool = False
    placement: str | None = None
    origin: str | None = None
    _logs: list[IO[str]] = field(default_factory=list, repr=False)

    @property
    def budget(self) -> int | None:  # type: ignore[override]
        return self.proxy.budget if self.proxy is not None else None

    @property
    def party(self) -> str:
        return backend(self.environ)

    async def open(self) -> None:
        """Start the proxy once for this placement and wait until `/healthz` answers at the
        address a sandbox dials. A remote proxy is pointed through its control route, or
        left as it is. Once started, a call checks the proxy is still up."""
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
            if self.proxy.control_token is not None:
                await self.serve(weights)
            elif weights is not None:
                raise RuntimeError(
                    f"[model] from_checkpoint names {weights}, and the proxy at "
                    f"{cfg.rollout.endpoint_url} opened no control route to this run: set "
                    f"{CONTROL_TOKEN_ENV}, or start it with --weights"
                )
            else:
                self.told, self.pointed = None, True
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
        self.told, self.pointed = weights, True
        served = await self.proxy.ready()
        logger.info("the proxy at %s answers (%s), serving %s", self.origin, where, served)

    def alive(self) -> None:
        why = self.proxy.gone() if self.proxy is not None else None
        if why:
            raise ProxyGone(f"the proxy is no longer reachable: {why}")

    async def serve(self, path: str | None) -> None:
        """Point the proxy at `path`, None for the base model, and note it as what this run
        told it."""
        if self.proxy is None:
            raise RuntimeError("the proxy is not started; there is nothing to point")
        self.alive()
        await self.proxy.swap(path)
        self.told, self.pointed = path, True

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

    async def record(self, job: str, trials: Sequence[Trial]) -> Harvest:
        """Drain every trial's records from the proxy, one `requests.jsonl` row per
        record, and check the served path."""
        proxy = self.proxy
        if proxy is None:
            raise RuntimeError("nothing was served; there are no records to harvest")
        counted = dict(trials=0, input_tokens=0, cache_tokens=0, output_tokens=0)
        noted = dict(turned_away=0, bridged=0)
        taken: dict[str, list[Record]] = {}
        asked: dict[str, int] = {}
        failed: set[str] = set()
        for trial in trials:
            name = trial.name
            try:
                taken[name] = records = await proxy.records(name)
            except Exception:
                self.alive()  # a fetch that failed because the proxy died says so
                raise
            counts = proxy.counters.pop(name, {})
            noted["turned_away"] += int(counts.get("turned_away", 0))
            for item in records:
                hit = min(int(item.cached_tokens or 0), len(item.prompt_token_ids))
                counted["input_tokens"] += len(item.prompt_token_ids)
                counted["cache_tokens"] += hit
                counted["output_tokens"] += len(item.completion_token_ids)
                noted["bridged"] += int(item.bridged)
                record.append(self.directory / record.REQUESTS, _row(job, name, item, hit))
            profile = trial.harness.profile
            if self.config.rollout.check_turns and profile.turns is not None:
                turns = turns_asked(trial.dir, trial.harness.name, profile.turns)
                if turns == 0 and records:
                    logger.warning(
                        "%s: the %s turn counter found no turns beside %d record(s); its log "
                        "format may have moved, so this trial is not checked",
                        name,
                        trial.harness.name,
                        len(records),
                    )
                if turns is not None:
                    asked[name] = turns
            reader = profile.failed_last
            text = harness_log(trial.dir, trial.harness.name) if reader else None
            if reader is not None and text is not None and reader(text):
                failed.add(name)
        counted["trials"] = sum(1 for trial in trials if trial.result is not None)
        if self.pointed:
            answered_by_ours(job, taken, self.told)
        return Harvest(taken, asked, failed, counted, noted, self.party, self._stopped(job, taken))

    def _stopped(self, job: str, taken: Mapping[str, Sequence[Record]]) -> RuntimeError | None:
        """The stop to raise once the job row is written: the proxy died under the batch,
        or served it nothing."""
        why = self.proxy.gone() if self.proxy is not None else None
        if why:
            return ProxyGone(f"the proxy is no longer reachable: {why}")
        why = served_nothing(taken)
        return NothingServed(f"job {job}: {why}") if why else None

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
    """Why the proxy served a job nothing, or None: no trial reached it, or every request
    failed at the sampler. A job with one answered or refused request is not this case."""
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


def _row(job: str, trial: str, item: Record, hit: int) -> dict[str, Any]:
    return {
        "job": job,
        "trial": trial,
        "seq": item.seq,
        "prompt_tokens": len(item.prompt_token_ids),
        "cached_tokens": hit,
        "completion_tokens": len(item.completion_token_ids),
        "stop_reason": item.stop_reason,
        "sample_ms": round(float(item.sample_ms), 1),
        "served": item.served,
        "request_id": item.request_id,
        "bridged": item.bridged,
        "error": item.error,
    }


def turns_asked(trial: Path, harness: str, counter: Any) -> int | None:
    """How many turns the harness asked for, off `agent/<harness>.txt`; None without the
    log."""
    text = harness_log(trial, harness)
    return None if text is None else int(counter(text))


def harness_log(trial: Path, harness: str) -> str | None:
    try:
        return (trial / "agent" / f"{bare_name(harness)}.txt").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None


def answered_by_ours(job: str, taken: Mapping[str, Sequence[Record]], told: str | None) -> None:
    """Raise unless every record of `job` was served from the path this run told its
    proxy."""
    answered = {item.served for records in taken.values() for item in records if item.served}
    if answered - {told}:
        raise ValueError(
            f"job {job} was served from {sorted(answered)}, but this run pointed its proxy "
            f"at {told}. Something re-pointed it while the batch was in flight."
        )
