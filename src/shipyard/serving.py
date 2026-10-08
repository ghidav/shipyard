"""The run's serving half: where the proxy goes for this sandbox, when it starts and
stops, what each job's records cost, and whose weights answered them. `Run` holds one
of these and delegates; nothing here opens a sandbox or takes a gradient."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any
from urllib.parse import urlsplit

from shipyard import record
from shipyard.admit import results
from shipyard.config import TINKER
from shipyard.proxy.client import Proxy
from shipyard.proxy.profiles import Profile, bare_name, profile_for, slug_of
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, SAMPLER_FAILED
from shipyard.rollout import Rollouts, Served

if TYPE_CHECKING:
    from shipyard.config import Blueprint, Rollout
    from shipyard.proxy.wire import Record

logger = logging.getLogger(__name__)

#: Sandboxes on this machine, whose containers reach a proxy here by its Docker name.
LOCAL_SANDBOXES = frozenset({"docker", "podman", "apple-container"})
#: Where the proxy's and the tunnel's output go, under the run directory.
PROXY_LOG = "proxy.log"
TUNNEL_LOG = "tunnel.log"


class ProxyGone(RuntimeError):
    """The proxy this run started exited under it: the run stops rather than send the next
    batch to an address nothing answers, which would come back as a batch of masks."""


class NothingServed(RuntimeError):
    """A served job the proxy served nothing for: an endpoint that is dead, unreachable or
    refused by its sampler, not a batch of failed rollouts to mask and carry on from."""

    def __init__(self, message: str, rolled: Rollouts | None = None) -> None:
        super().__init__(message)
        #: The job as harvested and recorded, for a report of what did come back.
        self.rolled = rolled


def placement(rollout: Rollout) -> str:
    """`remote` for a named `endpoint_url`, `local` for a sandbox on this machine, else
    `tunnel`: a sandbox elsewhere has no route back to a proxy in this process."""
    if rollout.endpoint_url.strip():
        return "remote"
    return "local" if rollout.sandbox in LOCAL_SANDBOXES else "tunnel"


def backend(environ: Mapping[str, str] | None = None) -> str:
    """The party sampling is billed to: `tinker`, or `tinker@<host>` under a
    `TINKER_BASE_URL`, so a run on another backend is not filed under Tinker's bill."""
    environ = os.environ if environ is None else environ
    host = urlsplit(environ.get("TINKER_BASE_URL") or "").hostname
    return f"{TINKER}@{host}" if host else TINKER


def tinker_base_url(environ: Mapping[str, str] | None = None) -> str:
    """The backend the proxy will sample from, as the SDK resolves it."""
    from tinker.lib.base_url import DEFAULT_BASE_URL

    environ = os.environ if environ is None else environ
    return (environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def start_weights(cfg: Blueprint) -> str | None:
    """The checkpoint the proxy starts on: a sampling run's `from_checkpoint` (a sampler
    path); none for a training run, whose `from_checkpoint` is a state path Tinker cannot
    sample and whose first step points the proxy at the weights it publishes."""
    return None if cfg.trains else (cfg.model.from_checkpoint or None)


def endpoint_settings(cfg: Blueprint, *, run: str | None = None) -> dict[str, Any]:
    """What `shipyard serve` is handed: the model and its checkpoint, the bind, the
    renderer, and every `[rollout]` knob the endpoint pins; volatile lines unless the
    run keeps them. Named, the run tags the proxy's Tinker session, as the trainer's is
    tagged."""
    rollout, profile = cfg.rollout, profile_for(cfg.rollout.harness)
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
        "volatile": list(volatile_for(rollout, profile)),
        **tagged,
    }


def volatile_for(rollout: Rollout, profile: Profile) -> tuple[str, ...]:
    """The profile's known lines under `cut_volatile`; nothing otherwise."""
    return tuple(profile.volatile) if rollout.cut_volatile else ()


@dataclass(frozen=True)
class Harvest:
    """One job's records attached to its rollouts, and what they add up to."""

    rollouts: Rollouts
    counted: dict[str, int]
    noted: dict[str, int]


@dataclass
class Serving:
    """One run's proxy: started by the first job, stopped with the run; the served path it
    was told, so a batch answered by other weights is refused rather than measured."""

    config: Blueprint
    directory: Path
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ, repr=False)
    proxy: Proxy | None = None
    told: str | None = None
    pointed: bool = False
    #: Where the proxy stood and the origin sandboxes dialled, kept once it is stopped.
    placement: str | None = None
    origin: str | None = None
    _logs: list[IO[str]] = field(default_factory=list, repr=False)

    @property
    def profile(self) -> Profile:
        return profile_for(self.config.rollout.harness)

    @property
    def party(self) -> str:
        return backend(self.environ)

    @property
    def budget(self) -> int | None:
        """The tokens one trial may sample, as the proxy said when it answered; None before
        it started or when it enforces no budget."""
        return self.proxy.budget if self.proxy is not None else None

    def model_name(self) -> str:
        """`<slug>/<model>`: the provider pose the harness's wire expects."""
        return f"{slug_of(self.profile)}/{self.config.model.name}"

    def served(self) -> Served:
        if self.proxy is None:
            raise RuntimeError("the proxy is not started; nothing can be rolled out yet")
        return Served(self.proxy, self.profile, fill_context=self.config.rollout.fill_context)

    async def start(self) -> Proxy:
        """The proxy for this placement, once, with `/healthz` answering where a sandbox
        will dial it (through the tunnel): local or tunnelled with the checkpoint on its
        command line, remote pointed through its control route or left as it stands."""
        if self.proxy is not None:
            return self.proxy
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
                await self.point(weights)
            elif weights is not None:
                raise RuntimeError(
                    f"[model] from_checkpoint names {weights}, and the proxy at "
                    f"{cfg.rollout.endpoint_url} opened no control route to this run: set "
                    f"{CONTROL_TOKEN_ENV}, or start it with --weights"
                )
            else:
                # Unpointed, it must still serve what this run measures: the base model.
                self.told, self.pointed = None, True
            return self.proxy
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
        return self.proxy

    def alive(self) -> None:
        """Raise `ProxyGone` when the tunnel or the serve process this run started has
        exited: asked before every job after the first, and before every re-point."""
        why = self.proxy.gone() if self.proxy is not None else None
        if why:
            raise ProxyGone(f"the proxy is no longer reachable: {why}")

    async def point(self, model_path: str | None) -> None:
        """Re-point the proxy at `model_path` (None for the base) and note it as what this
        run told it, so the next harvest checks the records against that path."""
        if self.proxy is None:
            raise RuntimeError("the proxy is not started; there is nothing to point")
        self.alive()
        await self.proxy.swap(model_path)
        self.told, self.pointed = model_path, True

    def _log(self, name: str) -> IO[str]:
        opened = (self.directory / name).open("a", encoding="utf-8")
        self._logs.append(opened)
        return opened

    async def harvest(self, rolled: Rollouts) -> Harvest:
        """Every trial's records off the proxy, one `requests.jsonl` row each, the turn
        counts off the harness's log when asked, and the served path checked."""
        proxy = self.proxy
        if proxy is None:
            raise RuntimeError("nothing was served; there are no records to harvest")
        counted = dict(trials=0, input_tokens=0, cache_tokens=0, output_tokens=0)
        noted = dict(turned_away=0, cut=0, bridged=0)
        taken: dict[str, list[Record]] = {}
        asked: dict[str, int] = {}
        failed: set[str] = set()
        counter = self.profile.turns if self.config.rollout.check_turns else None
        reader = self.profile.failed_last
        for trial in rolled.trials:
            name = Path(trial).name
            try:
                taken[name] = records = await proxy.records(name)
            except Exception:
                self.alive()  # a fetch that failed because the proxy died says so
                raise
            counts = proxy.counters.pop(name, {})
            noted["turned_away"] += int(counts.get("turned_away", 0))
            noted["cut"] += int(counts.get("cut", 0))
            for item in records:
                hit = min(int(item.cached_tokens or 0), len(item.prompt_token_ids))
                counted["input_tokens"] += len(item.prompt_token_ids)
                counted["cache_tokens"] += hit
                counted["output_tokens"] += len(item.completion_token_ids)
                noted["bridged"] += int(item.bridged)
                record.append(self.directory / record.REQUESTS, _row(rolled.job, name, item, hit))
            if counter is not None:
                turns = turns_asked(Path(trial), self.config.rollout.harness, counter)
                if turns == 0 and records:
                    logger.warning(
                        "%s: the %s turn counter found no turns beside %d record(s); its log "
                        "format may have moved, so this trial is not checked",
                        name,
                        bare_name(self.config.rollout.harness),
                        len(records),
                    )
                if turns is not None:
                    asked[name] = turns
            text = harness_log(Path(trial), self.config.rollout.harness) if reader else None
            if reader is not None and text is not None and reader(text):
                failed.add(name)
        counted["trials"] = sum(1 for _ in results(rolled))
        if self.pointed:
            answered_by_ours(rolled.job, taken, self.told)
        return Harvest(replace(rolled, records=taken, asked=asked, failed=failed), counted, noted)

    def close(self) -> None:
        """Stop the proxy and the tunnel; synchronous, for a run closing outside its loop."""
        proxy, self.proxy = self.proxy, None
        if proxy is not None:
            proxy.close()
        for opened in self._logs:
            opened.close()
        self._logs.clear()


def served_nothing(rolled: Rollouts) -> str | None:
    """Why the proxy served a harvested job nothing, or None: no trial reached it, or every
    request that did failed at the sampler (weights expired, a poisoned client). A job with
    one answered or refused request is not this; its silent trials are masked one by one."""
    taken = rolled.records
    if taken is None or not rolled.trials:
        return None
    made = [item for trial in rolled.trials for item in taken.get(Path(trial).name) or []]
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
    """How many turns the harness asked for, counted off `agent/<harness>.txt`; None when
    it left no such log, which is not a claim that it asked for nothing."""
    text = harness_log(trial, harness)
    return None if text is None else int(counter(text))


def harness_log(trial: Path, harness: str) -> str | None:
    """The harness's own log, `agent/<harness>.txt`, or None when it left none."""
    try:
        return (trial / "agent" / f"{bare_name(harness)}.txt").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None


def answered_by_ours(job: str, taken: Mapping[str, Sequence[Record]], told: str | None) -> None:
    """Raise unless every record of `job` was served from the path this run told its
    proxy: the records carry it so a proxy re-pointed under a batch is caught here."""
    answered = {item.served for records in taken.values() for item in records if item.served}
    if answered - {told}:
        raise ValueError(
            f"job {job} was served from {sorted(answered)}, but this run pointed its proxy "
            f"at {told}: something re-pointed it while the batch was in flight, so those "
            "turns did not come from this run's weights"
        )
