from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from harness_proxy.bridge import Index, Reply
from harness_proxy.chat import Caller, Refused
from harness_proxy.wire import (
    BUDGET,
    CONTEXT,
    SAMPLER_FAILED,
    Picture,
    Record,
    ids_of,
    pictured,
    pictures_of,
)

if TYPE_CHECKING:
    from harness_proxy.rendering import Prompt


@dataclass(frozen=True)
class Sampled:
    """One sampled reply: its tokens, why sampling stopped, and the number of the record it
    made; None for a replayed send, which made none."""

    tokens: tuple[int, ...]
    stop_reason: str
    seq: int | None


class Recorder:
    """What every call of a lane samples through: it pins the run's settings, replays a retried
    call's reply, fits the budget and the context, and keeps the records of every trial."""

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
        """Override the harness's sampling parameters with the run's, `max_tokens` only capped."""
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

    async def sample(
        self,
        caller: Caller,
        prompt: Prompt,
        sampling_params: Any,
        *,
        requested_model: str | None = None,
    ) -> Sampled:
        """Sample one reply to `prompt` for `caller`'s trial and record it, or replay the reply an
        earlier send of the call got; `Refused`, once recorded, for a trial past its budget or a
        prompt past the context."""
        trial = caller.trial
        stamp = {
            "requested_model": requested_model,
            "request_id": caller.idempotency_key or caller.request_id,
            "bridged": prompt.bridged,
            "prompt_digests": prompt.digests,
        }
        try:
            return await self._sample(caller, prompt, sampling_params, stamp)
        finally:
            if trial in self.taken:
                # A call that outlived its trial's fetch (Harbor's timeout can cut one)
                # leaves records nobody reads, so drop them.
                self._forget(trial)

    async def _sample(
        self, caller: Caller, prompt: Prompt, sampling_params: Any, stamp: dict[str, Any]
    ) -> Sampled:
        trial = caller.trial
        prompt_ids = ids_of(prompt.model_input)
        pictures = pictures_of(prompt.model_input)
        pinned = self.pin(sampling_params)
        # Keyed on the params as pinned, before the budget cut, so a retry arriving after
        # the trial's count moved still matches its reply.
        asked = json.dumps(_params_dict(pinned), sort_keys=True, default=str)
        keys: list[tuple[Any, ...]] = [(trial, prompt_ids, asked)]
        if caller.idempotency_key:
            keys.append((trial, caller.idempotency_key))
        # A retry replays by either key. A fresh request with an idempotency key replays
        # by that key only, since the same prompt under a new key is a resample.
        lookups = keys if caller.retry else keys[1:]
        for key in lookups:
            stored = self._replies.get(key)
            answered = await stored if stored is not None else None
            if answered is not None:
                return replace(answered, seq=None)  # the SDK sent this before: one record
        # Fit after the replay, so a retry of the turn that spent the budget gets its reply.
        wanted = int(getattr(sampling_params, "max_tokens", None) or 0)
        pinned = self._fitting(trial, prompt, prompt_ids, pinned, wanted, stamp)
        params = _params_dict(pinned)
        pending: asyncio.Future[Sampled | None] = asyncio.get_running_loop().create_future()
        for key in keys:
            self._replies[key] = pending
        # Read together before the await, so a swap in flight does not stamp the record
        # with weights that did not answer it.
        client, served = self.client, self.served
        started = time.perf_counter()
        try:
            response = await client.sample_async(
                prompt=prompt.model_input, num_samples=1, sampling_params=pinned
            )
            sequence = response.sequences[0]
        except BaseException as failed:
            for key in keys:
                if self._replies.get(key) is pending:
                    del self._replies[key]
            if not pending.done():
                pending.set_result(None)  # a retry waiting on it samples for itself
            failure = f"{SAMPLER_FAILED}{type(failed).__name__}"
            self._add(trial, prompt_ids, params, served, **stamp, error=failure, images=pictures)
            raise
        tokens = tuple(int(token) for token in sequence.tokens)
        stop_reason = str(getattr(sequence, "stop_reason", "") or "")
        record = self._add(
            trial,
            prompt_ids,
            params,
            served,
            **stamp,
            completion=tokens,
            logprobs=tuple(float(x) for x in (getattr(sequence, "logprobs", None) or [])),
            stop_reason=stop_reason,
            sample_ms=(time.perf_counter() - started) * 1000.0,
            cached_tokens=int(getattr(response, "prompt_cache_hit_tokens", 0) or 0),
            images=pictures,
        )
        self.spoke[trial] = self.spoke.get(trial, 0) + len(tokens)
        sampled = Sampled(tokens, stop_reason, record.seq)
        if not pending.done():
            pending.set_result(sampled)
        return sampled

    def _fitting(
        self,
        trial: str,
        prompt: Prompt,
        prompt_ids: tuple[int, ...],
        pinned: Any,
        wanted: int,
        stamp: dict[str, Any],
    ) -> Any:
        """Return the sampling params that fit the budget and the context, or record a refusal
        and raise it."""
        if self.budget:
            spent = self.spoke.get(trial, 0)
            if spent >= self.budget:
                self._refuse(trial, prompt_ids, pinned, BUDGET, stamp)
                raise Refused(
                    f"This rollout has spent its budget: {spent} sampled tokens of "
                    f"{self.budget}, the model's context length. No further requests are served."
                )
            if int(getattr(pinned, "max_tokens", 0) or 0) > self.budget - spent:
                pinned = pinned.model_copy(update={"max_tokens": int(self.budget - spent)})
        if not self.max_context:
            return pinned
        length, wants = int(prompt.model_input.length), int(getattr(pinned, "max_tokens", 0) or 0)
        if length + wants <= self.max_context:
            return pinned
        if self.fill_context and length < self.max_context:
            return pinned.model_copy(update={"max_tokens": self.max_context - length})
        self._refuse(trial, prompt_ids, pinned, CONTEXT, stamp)
        # What a harness compacts its history on.
        raise Refused(
            f"prompt is too long: {length} prompt tokens plus {wanted or wants} max_tokens "
            "exceed the model's context window"
        )

    def _refuse(
        self,
        trial: str,
        prompt_ids: tuple[int, ...],
        pinned: Any,
        why: str,
        stamp: dict[str, Any],
    ) -> None:
        self._add(trial, prompt_ids, _params_dict(pinned), self.served, **stamp, error=why)

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
        requested_model: str | None = None,
        request_id: str | None = None,
        bridged: bool = False,
        prompt_digests: tuple[str, ...] = (),
    ) -> Record:
        """Append the next record of `trial`, numbered from 1, refused and failed calls included."""
        self._seq[trial] = seq = self._seq.get(trial, 0) + 1
        record = Record(
            seq=seq,
            at=time.time(),
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion,
            inference_logprobs=logprobs,
            stop_reason=stop_reason,
            sampling_params=params,
            requested_model=requested_model,
            sample_ms=sample_ms,
            served=served,
            cached_tokens=cached_tokens,
            request_id=request_id,
            bridged=bridged,
            prompt_digests=prompt_digests,
            reply_digest=None,
            error=error,
            images=images,
        )
        self.records.setdefault(trial, []).append(record)
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
        bridgeable: bool = True,
    ) -> None:
        """Set the parsed reply's digest on record `seq` and, unless the harness got the reply
        changed, index it for the bridge."""
        records = self.records.get(trial, [])
        for at in range(len(records) - 1, -1, -1):
            if records[at].seq != seq:
                continue
            record = records[at] = replace(records[at], reply_digest=reply_digest)
            if bridgeable and pictured(record):
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
        """Remove and return the trial's records, and drop its counters and stored replies."""
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
