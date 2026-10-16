"""Sampling client wrapper that records every model call as token ids, per trial.

It also applies the run's pinned sampling parameters, enforces the token budget and the
context length, and replays SDK retries."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any

from shipyard.proxy import cookbook
from shipyard.proxy.bridge import Index, Reply
from shipyard.proxy.exchange import exchange
from shipyard.proxy.wire import (
    BUDGET,
    CONTEXT,
    SAMPLER_FAILED,
    Picture,
    Record,
    ids_of,
    pictured,
    pictures_of,
)


class Recorder:
    """Sampling client given to the cookbook app, holding the records of every trial.

    A request with no trial in its address is served and recorded under the trial name ""."""

    def __init__(
        self,
        client: Any,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        max_context: int | None = None,
        budget: int | None = None,
        fill_context: bool = False,
        served: str | None = None,
    ) -> None:
        self.client = client
        self.served = served
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        #: Maximum tokens a turn may sample. A request asking for more is capped to this.
        self.max_tokens = max_tokens
        self.max_context = max_context
        #: Total tokens a trial may sample. None for no limit.
        self.budget = budget
        #: Whether a turn that would overrun the context gets the remaining context as its
        #: limit instead of being refused.
        self.fill_context = fill_context
        self.records: dict[str, list[Record]] = {}
        #: Tokens sampled so far, per trial.
        self.spoke: dict[str, int] = {}
        self._seq: dict[str, int] = {}
        self._replies: dict[tuple[Any, ...], asyncio.Future[Any]] = {}
        #: Sampled replies the bridge may extend, per trial.
        self.indexes: dict[str, Index] = {}
        #: Trials whose records were taken. No later fetch reads them, so records of
        #: calls that finish afterwards are dropped.
        self.taken: set[str] = set()

    def index_for(self, trial: str) -> Index:
        return self.indexes.setdefault(trial, Index())

    def pin(self, sampling_params: Any) -> Any:
        """Override the harness's sampling parameters with the run's.

        `max_tokens` is only capped, because it bounds a turn's length and does not change which
        token is drawn."""
        update: dict[str, Any] = {}
        if self.temperature is not None:
            update["temperature"] = float(self.temperature)
        if self.top_p is not None:
            update["top_p"] = float(self.top_p)
        if self.top_k is not None:
            update["top_k"] = int(self.top_k)
        asked = int(getattr(sampling_params, "max_tokens", None) or 0)
        if self.max_tokens is not None and (not asked or asked > self.max_tokens):
            update["max_tokens"] = int(self.max_tokens)
        return sampling_params.model_copy(update=update) if update else sampling_params

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        found = exchange.get()
        trial = found.trial if found is not None else ""
        try:
            return await self._sample(found, trial, prompt, num_samples, sampling_params)
        finally:
            if trial in self.taken:
                # A call that outlived its trial's fetch (Harbor's timeout can cut one)
                # leaves records nobody reads, so drop them.
                self._forget(trial)

    async def _sample(
        self, found: Any, trial: str, prompt: Any, num_samples: int, sampling_params: Any
    ) -> Any:
        prompt_ids = ids_of(prompt)
        pictures = pictures_of(prompt)
        pinned = self.pin(sampling_params)
        # Keyed on the params as pinned, before the budget cut, so a retry arriving after
        # the trial's count moved still matches its reply.
        asked = json.dumps(_params_dict(pinned), sort_keys=True, default=str)
        keys: list[tuple[Any, ...]] = [(trial, prompt_ids, int(num_samples), asked)]
        if found is not None and found.idempotency_key:
            keys.append((trial, found.idempotency_key))
        # A retry replays by either key. A fresh request with an idempotency key replays
        # by that key only, since the same prompt under a new key is a resample.
        lookups = keys if found is not None and found.retry else keys[1:]
        for key in lookups:
            stored = self._replies.get(key)
            answered = await stored if stored is not None else None
            if answered is not None:
                return answered  # the SDK sent this before: one reply, one record
        # Fit after the replay, so a retry of the turn that spent the budget gets its reply.
        pinned = self._fitting(trial, prompt, prompt_ids, pinned, int(num_samples))
        params = _params_dict(pinned)
        pending: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        for key in keys:
            self._replies[key] = pending
        # Read together before the await, so a swap in flight does not stamp the record
        # with weights that did not answer it.
        client, served = self.client, self.served
        started = time.perf_counter()
        try:
            response = await client.sample_async(
                prompt=prompt, num_samples=num_samples, sampling_params=pinned
            )
        except BaseException as failed:
            for key in keys:
                if self._replies.get(key) is pending:
                    del self._replies[key]
            if not pending.done():
                pending.set_result(None)  # a retry waiting on it samples for itself
            failure = f"{SAMPLER_FAILED}{type(failed).__name__}"
            self._add(trial, prompt_ids, params, served, error=failure, images=pictures)
            raise
        if not pending.done():
            pending.set_result(response)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # The cache hit describes the prompt the sequences share, so only the first carries it.
        cache_hit = int(getattr(response, "prompt_cache_hit_tokens", 0) or 0)
        for sequence in list(getattr(response, "sequences", None) or []):
            tokens = tuple(int(token) for token in sequence.tokens)
            self._add(
                trial,
                prompt_ids,
                params,
                served,
                completion=tokens,
                logprobs=tuple(float(x) for x in (getattr(sequence, "logprobs", None) or [])),
                stop_reason=str(getattr(sequence, "stop_reason", "") or ""),
                sample_ms=elapsed_ms,
                cached_tokens=cache_hit,
                images=pictures,
            )
            cache_hit = 0
            self.spoke[trial] = self.spoke.get(trial, 0) + len(tokens)
        return response

    def _fitting(
        self, trial: str, prompt: Any, prompt_ids: tuple[int, ...], pinned: Any, num_samples: int
    ) -> Any:
        """Return the sampling params to serve the request with, or raise a refusal.

        A trial that has spent its budget gets the cookbook's 400. A turn that would exceed the
        budget has `max_tokens` cut to its per-sample share of what is left. A prompt with no room
        for its reply is refused as a context overflow, unless `fill_context` lets the turn use
        the remaining context."""
        if self.budget:
            spent = self.spoke.get(trial, 0)
            share = (self.budget - spent) // max(1, num_samples)
            if share <= 0:
                self._refuse(trial, prompt_ids, pinned, BUDGET)
                raise cookbook.refused(
                    f"This rollout has spent its budget: {spent} sampled tokens of "
                    f"{self.budget}, the model's context length. No further requests are served."
                )
            if int(getattr(pinned, "max_tokens", 0) or 0) > share:
                pinned = pinned.model_copy(update={"max_tokens": int(share)})
        if not self.max_context:
            return pinned
        asked, wants = int(prompt.length), int(getattr(pinned, "max_tokens", 0) or 0)
        if asked + wants <= self.max_context:
            return pinned
        if self.fill_context and asked < self.max_context:
            return pinned.model_copy(update={"max_tokens": self.max_context - asked})
        self._refuse(trial, prompt_ids, pinned, CONTEXT)
        raise cookbook.too_long(
            "Prompt length plus max_tokens exceeds the model's context window: "
            f"{asked} prompt tokens + {wants} max_tokens > {self.max_context}."
        )

    def _refuse(self, trial: str, prompt_ids: tuple[int, ...], pinned: Any, why: str) -> None:
        self._add(trial, prompt_ids, _params_dict(pinned), self.served, error=why)

    def _add(
        self,
        trial: str,
        prompt_ids: tuple[int, ...],
        params: dict[str, Any],
        served: str | None,
        *,
        completion: tuple[int, ...] = (),
        logprobs: tuple[float, ...] = (),
        stop_reason: str = "",
        sample_ms: float = 0.0,
        cached_tokens: int = 0,
        error: str | None = None,
        images: tuple[Picture, ...] = (),
    ) -> Record:
        """Append the next record of `trial`, numbered from 1 within the trial.

        Refused and failed calls are recorded too, so the records show every call the harness
        made."""
        found = exchange.get()
        self._seq[trial] = seq = self._seq.get(trial, 0) + 1
        record = Record(
            seq=seq,
            at=time.time(),
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion,
            inference_logprobs=logprobs,
            stop_reason=stop_reason,
            sampling_params=params,
            requested_model=cookbook.scope_value("requested_model"),
            sample_ms=sample_ms,
            served=served,
            cached_tokens=cached_tokens,
            request_id=(found.idempotency_key or found.request_id) if found else None,
            bridged=found.bridged if found else False,
            prompt_digests=found.prompt_digests if found else (),
            reply_digest=None,
            error=error,
            images=images,
        )
        self.records.setdefault(trial, []).append(record)
        if found is not None and error is None:
            found.seq = seq
        return record

    def reply(
        self,
        trial: str,
        seq: int,
        reply_digest: str,
        *,
        ended_with_stop: bool,
        rendered: tuple[int, ...] | None = None,
        call_ids: tuple[str, ...] = (),
    ) -> None:
        """Set the parsed reply's digest on record `seq` and index the reply for the bridge.

        The reply is indexed with the ids of its tool calls. A reply whose prompt has an image the
        record did not keep is not indexed, because nothing can rebuild that prompt."""
        records = self.records.get(trial, [])
        for at in range(len(records) - 1, -1, -1):
            if records[at].seq != seq:
                continue
            record = records[at] = replace(records[at], reply_digest=reply_digest)
            if pictured(record):
                reply = Reply(
                    prompt_ids=record.prompt_token_ids,
                    completion_ids=record.completion_token_ids,
                    ended_with_stop=ended_with_stop,
                    rendered=record.prompt_token_ids if rendered is None else rendered,
                    pictures=record.images,
                )
                self.index_for(trial).add(record.prompt_digests, reply_digest, reply, call_ids)
            return

    def records_for(self, trial: str) -> list[Record]:
        return list(self.records.get(trial, []))

    def take(self, trial: str) -> list[Record]:
        """Remove and return the trial's records, and drop its counters and stored replies.

        Each trial is read once, so the proxy does not keep every token of a long run."""
        if trial:
            self.taken.add(trial)
        return self._forget(trial)

    def _forget(self, trial: str) -> list[Record]:
        for key in [key for key in self._replies if key[0] == trial]:
            del self._replies[key]
        for counter in (self.spoke, self._seq, self.indexes):
            counter.pop(trial, None)
        return self.records.pop(trial, [])


def _params_dict(sampling_params: Any) -> dict[str, Any]:
    dump = getattr(sampling_params, "model_dump", None)
    if not callable(dump):
        return {}
    try:
        return {key: value for key, value in dump().items() if value is not None}
    except Exception:  # noqa: BLE001 - a failed annotation must not fail a rollout
        return {}
