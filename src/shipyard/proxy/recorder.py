"""What the proxy keeps of every model call: the token ids per trial, with the run's
sampling pins applied, the budget and the context fit enforced, and SDK retries replayed."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any

from shipyard.proxy import cookbook
from shipyard.proxy.bridge import Index, Reply
from shipyard.proxy.exchange import exchange

# The record and the image stand-in live in `wire`, which the run reads without the
# cookbook; re-exported here, where the recorder makes them.
from shipyard.proxy.wire import IMAGE_TOKEN as IMAGE_TOKEN
from shipyard.proxy.wire import SAMPLER_FAILED
from shipyard.proxy.wire import Record as Record


class Recorder:
    """The sampling client the cookbook app calls, and the keeper of every trial's records.
    A request with no trial in its address is served and recorded under "", never dropped
    and never filed under the last trial seen."""

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
        #: The most a turn may run: a request asking more is cut to it.
        self.max_tokens = max_tokens
        self.max_context = max_context
        #: Sampled tokens a trial may write in all; None for no limit.
        self.budget = budget
        #: Whether a turn that would overrun the context is given what is left instead.
        self.fill_context = fill_context
        self.records: dict[str, list[Record]] = {}
        #: Per trial: requests refused (budget or context), volatile lines cut, tokens sampled.
        self.turned_away: dict[str, int] = {}
        self.cut: dict[str, int] = {}
        self.spoke: dict[str, int] = {}
        self._seq: dict[str, int] = {}
        self._replies: dict[tuple[Any, ...], asyncio.Future[Any]] = {}
        #: Per trial, the sampled replies the bridge may extend.
        self.indexes: dict[str, Index] = {}

    def index_for(self, trial: str) -> Index:
        return self.indexes.setdefault(trial, Index())

    def pin(self, sampling_params: Any) -> Any:
        """The run's distribution knobs over the harness's; `max_tokens` only capped, since
        it bounds how long a turn runs and not which token is drawn."""
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
        prompt_ids = ids_of(prompt)
        pinned = self.pin(sampling_params)
        # Keyed on the params as pinned, before the budget cut: a retry arriving after the
        # trial's count moved would otherwise be cut differently and not match its reply.
        asked = json.dumps(_params_dict(pinned), sort_keys=True, default=str)
        keys: list[tuple[Any, ...]] = [(trial, prompt_ids, int(num_samples), asked)]
        if found is not None and found.idempotency_key:
            keys.append((trial, found.idempotency_key))
        # A retry replays by either key; a fresh request with an idempotency key replays
        # by that key alone, since the same prompt under a new key is a resample.
        lookups = keys if found is not None and found.retry else keys[1:]
        for key in lookups:
            stored = self._replies.get(key)
            answered = await stored if stored is not None else None
            if answered is not None:
                return answered  # the SDK sent this before: one reply, one record
        # Fitted after the replay: a retry of the turn that spent the budget gets its reply.
        pinned = self._fitting(trial, prompt, prompt_ids, pinned, int(num_samples))
        params = _params_dict(pinned)
        pending: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        for key in keys:
            self._replies[key] = pending
        # Read together, before the await: a swap while this was in flight must not
        # re-stamp the record with weights that did not answer it.
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
            self._add(trial, prompt_ids, params, served, error=failure)
            raise
        if not pending.done():
            pending.set_result(response)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # The cache hit describes the prompt the sequences share: spent on the first only.
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
            )
            cache_hit = 0
            self.spoke[trial] = self.spoke.get(trial, 0) + len(tokens)
        return response

    def _fitting(
        self, trial: str, prompt: Any, prompt_ids: tuple[int, ...], pinned: Any, num_samples: int
    ) -> Any:
        """The params a request is served with, or the refusal: a trial past its budget is
        refused (the cookbook's 400, not an overflow), a turn beyond the budget is cut to
        its share of what is left per sample, and a prompt with no room for its reply is
        refused as an overflow."""
        if self.budget:
            spent = self.spoke.get(trial, 0)
            share = (self.budget - spent) // max(1, num_samples)
            if share <= 0:
                self._refuse(trial, prompt_ids, pinned, "budget")
                raise cookbook.refused(
                    f"This rollout has spent its budget: {spent} sampled tokens of "
                    f"{self.budget}, the model's context length. Nothing more is served to it."
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
        self._refuse(trial, prompt_ids, pinned, "context")
        raise cookbook.too_long(
            "Prompt length plus max_tokens exceeds the model's context window: "
            f"{asked} prompt tokens + {wants} max_tokens > {self.max_context}."
        )

    def _refuse(self, trial: str, prompt_ids: tuple[int, ...], pinned: Any, why: str) -> None:
        self.turned_away[trial] = self.turned_away.get(trial, 0) + 1
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
    ) -> Record:
        """The next record of the trial, numbered per trial from 1, refused and failed calls
        included so the sequence says what the harness asked, not only what it was given."""
        found = exchange.get()
        if found is not None and found.cut:
            # Counted once per request that left a record: a replayed retry cut nothing new.
            self.cut[trial] = self.cut.get(trial, 0) + found.cut
            found.cut = 0
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
        """Stamp the parsed reply's digest on record `seq` of the trial and index the
        reply for the bridge, with its calls' ids, unless the prompt held an image, which
        no ids rebuild."""
        records = self.records.get(trial, [])
        for at in range(len(records) - 1, -1, -1):
            if records[at].seq != seq:
                continue
            record = records[at] = replace(records[at], reply_digest=reply_digest)
            if IMAGE_TOKEN not in record.prompt_token_ids:
                reply = Reply(
                    prompt_ids=record.prompt_token_ids,
                    completion_ids=record.completion_token_ids,
                    ended_with_stop=ended_with_stop,
                    rendered=record.prompt_token_ids if rendered is None else rendered,
                )
                self.index_for(trial).add(record.prompt_digests, reply_digest, reply, call_ids)
            return

    def records_for(self, trial: str) -> list[Record]:
        return list(self.records.get(trial, []))

    def take(self, trial: str) -> list[Record]:
        """This trial's records, removed with its counters and replies: a trial is read once,
        and a proxy that only accumulated would hold every token of a long run."""
        for key in [key for key in self._replies if key[0] == trial]:
            del self._replies[key]
        for counter in (self.turned_away, self.cut, self.spoke, self._seq, self.indexes):
            counter.pop(trial, None)
        return self.records.pop(trial, [])


def ids_of(prompt: Any) -> tuple[int, ...]:
    """The prompt as ids, every non-text chunk as a run of `IMAGE_TOKEN` of its length:
    `ModelInput.to_ints` refuses an image chunk, and the record still has to say how long."""
    ids: list[int] = []
    for chunk in prompt.chunks:
        tokens = getattr(chunk, "tokens", None)
        if tokens is not None:
            ids.extend(int(token) for token in tokens)
        else:
            ids.extend([IMAGE_TOKEN] * int(getattr(chunk, "length", 0) or 0))
    return tuple(ids)


def _params_dict(sampling_params: Any) -> dict[str, Any]:
    dump = getattr(sampling_params, "model_dump", None)
    if not callable(dump):
        return {}
    try:
        return {key: value for key, value in dump().items() if value is not None}
    except Exception:  # noqa: BLE001 - an annotation never fails a rollout
        return {}
