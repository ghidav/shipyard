"""The run's side of the proxy, faked: records, a `Proxy` that notes what it was asked, and a
blueprint to open an endpoint on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from shipyard.config import Blueprint
from shipyard.proxy.wire import Picture, Record
from tests.proxies import MODEL, ids


def served(**rollout: Any) -> Blueprint:
    """A served evaluate blueprint: `from_checkpoint` goes to `[model]`, the rest to `[rollout]`."""
    model = {"name": MODEL}
    if "from_checkpoint" in rollout:
        model["from_checkpoint"] = rollout.pop("from_checkpoint")
    return Blueprint.model_validate(
        {
            "model": model,
            "data": {"dataset": "d", "batch_size": 1},
            "rollout": {"harness": "pi@0.85.1", **rollout},
            "recipe": {"kind": "evaluate"},
        }
    )


def made(
    prompt: str | tuple[int, ...] = "go",
    completion: str | tuple[int, ...] = "ok",
    *,
    seq: int = 1,
    error: str | None = None,
    served: str | None = None,
    cached: int = 0,
    bridged: bool = False,
    request_id: str | None = None,
    images: tuple[Picture, ...] = (),
) -> Record:
    """A record as the proxy would have kept it, with text rendered by character."""
    prompt_ids = ids(prompt) if isinstance(prompt, str) else tuple(prompt)
    completion_ids = (
        () if error else (ids(completion) if isinstance(completion, str) else tuple(completion))
    )
    return Record(
        seq=seq,
        at=0.0,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        inference_logprobs=(-0.5,) * len(completion_ids),
        stop_reason="" if error else "stop",
        sampling_params={"temperature": 1.0},
        requested_model="m",
        sample_ms=0.0 if error else 12.0,
        served=served,
        cached_tokens=cached,
        request_id=request_id,
        bridged=bridged,
        prompt_digests=(),
        reply_digest=None,
        error=error,
        images=images,
    )


@dataclass
class FakeProxy:
    """`Proxy` as `ProxyEndpoint` drives it; answers each trial's records by its task name."""

    origin: str = "http://host.docker.internal:8000"
    token: str = "harness-token"
    control_token: str | None = "control-token"
    answers: dict[str, list[Record]] = field(default_factory=dict)
    default: list[Record] | None = None
    failing: set[str] = field(default_factory=set)
    placed: list[tuple[str, Any, dict[str, Any]]] = field(default_factory=list)
    addressed: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    swapped: list[str | None] = field(default_factory=list)
    #: The token budget `/healthz` reports; None: the proxy enforces none.
    budget: int | None = None
    closed: int = 0
    #: How often `/healthz` was probed, and why it is gone (None: alive).
    probed: int = 0
    exited: str | None = None

    @property
    def host(self) -> str:
        return urlsplit(self.origin).hostname or ""

    async def local(self, settings: Any, *, host: str, **named: Any) -> FakeProxy:
        self.placed.append(("local", dict(settings), {"host": host, **named}))
        return self

    async def tunnelled(self, settings: Any, **named: Any) -> FakeProxy:
        self.placed.append(("tunnel", dict(settings), named))
        self.origin = "https://fake-words.trycloudflare.com"
        return self

    def remote(self, url: str, token: str, control_token: str | None = None) -> FakeProxy:
        if not token:
            raise ValueError("SHIPYARD_PROXY_TOKEN is unset")
        self.placed.append(("remote", url, {"token": token, "control_token": control_token}))
        self.origin, self.token, self.control_token = url.rstrip("/"), token, control_token
        return self

    def address_for(self, trial: str) -> str:
        self.addressed.append(trial)
        return f"{self.origin}/r/trial/{trial}/v1"

    async def swap(self, model_path: str | None) -> None:
        if self.control_token is None:
            raise RuntimeError("no control route")
        self.swapped.append(model_path)

    async def records(self, trial: str) -> list[Record]:
        self.fetched.append(trial)
        task = trial.split("__")[0]
        if task in self.failing:
            raise httpx.ConnectError("the proxy went away")
        found = self.answers.get(task, self.default)
        return list(found if found is not None else [made()])

    async def ready(self, seconds: float = 30.0) -> str:
        self.probed += 1
        return MODEL

    def gone(self) -> str | None:
        return self.exited

    def close(self) -> None:
        self.closed += 1
