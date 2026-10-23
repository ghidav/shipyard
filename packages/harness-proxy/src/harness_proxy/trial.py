from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from harness_proxy.bridge import Index, Reply
from harness_proxy.chat import Caller, Refused
from harness_proxy.cookbook import remembering
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


class Trial:
    """A trial the run opened, which its calls sample through until the run closes it: the
    renderer and weights they are answered with, the run's settings it pins on them, the tokens it
    has sampled, its records, the replies an SDK's retry is sent again, and the bridge's index of
    the replies a later call may quote."""

    def __init__(
        self,
        name: str,
        client: Any,
        *,
        renderer: Any = None,
        weights: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        max_context: int | None = None,
        budget: int | None = None,
        fill_context: bool = False,
    ) -> None:
        self.name = name
        #: The sampling client on its weights, which its lane holds while a trial uses them.
        self.client = client
        #: Its own copy of its lane's renderer, over a tokenizer that remembers what it encoded.
        self.renderer = remembering(renderer) if renderer is not None else None
        #: The weights it was opened on, under the backend's prefix; None for the base model.
        self.weights = weights
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        #: Maximum tokens a call may sample. A call asking for more is capped to this.
        self.max_tokens = max_tokens
        self.max_context = max_context
        #: Total tokens the trial may sample. None for no limit.
        self.budget = budget
        #: Whether a call that would overrun the context gets the remaining context as its
        #: limit instead of being turned away.
        self.fill_context = fill_context
        self.records: list[Record] = []
        #: Tokens sampled so far.
        self.spoke = 0
        #: Sampled replies the bridge may extend.
        self.index = Index()
        self._seq = 0
        self._replies: dict[tuple[Any, ...], asyncio.Future[Any]] = {}

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
        """Sample one reply to `prompt` and record it, or replay the reply an earlier send of the
        call got; `Refused`, once recorded, for a call past the budget or a prompt past the
        context."""
        prompt_ids = ids_of(prompt.model_input)
        pictures = pictures_of(prompt.model_input)
        stamp = {
            "requested_model": requested_model,
            "request_id": caller.idempotency_key or caller.request_id,
            "bridged": prompt.bridged,
            "prompt_digests": prompt.digests,
        }
        pinned = self.pin(sampling_params)
        # Keyed on the params as pinned, before the budget cut, so a retry arriving after
        # the trial's count moved still matches its reply.
        asked = json.dumps(_params_dict(pinned), sort_keys=True, default=str)
        keys: list[tuple[Any, ...]] = [(prompt_ids, asked)]
        if caller.idempotency_key:
            keys.append((caller.idempotency_key,))
        # A retry replays by either key. A fresh request with an idempotency key replays
        # by that key only, since the same prompt under a new key is a resample.
        lookups = keys if caller.retry else keys[1:]
        for key in lookups:
            stored = self._replies.get(key)
            answered = await stored if stored is not None else None
            if answered is not None:
                return replace(answered, seq=None)  # the SDK sent this before: one record
        # Fit after the replay, so a retry of the call that spent the budget gets its reply.
        wanted = int(getattr(sampling_params, "max_tokens", None) or 0)
        pinned = self._fitting(prompt, prompt_ids, pinned, wanted, stamp)
        params = _params_dict(pinned)
        pending: asyncio.Future[Sampled | None] = asyncio.get_running_loop().create_future()
        for key in keys:
            self._replies[key] = pending
        started = time.perf_counter()
        try:
            response = await self.client.sample_async(
                prompt=prompt.model_input, num_samples=1, sampling_params=pinned
            )
            sequence = response.sequences[0]
        except BaseException as failed:
            for key in keys:
                if self._replies.get(key) is pending:
                    del self._replies[key]
            if not pending.done():
                pending.set_result(None)  # a retry waiting on it samples for itself
            if isinstance(failed, Refused):
                # The backend refused the request before sampling it, as a renderer refuses an
                # image it cannot take: the request was at fault, and it costs no record.
                raise
            failure = f"{SAMPLER_FAILED}{type(failed).__name__}"
            self._add(prompt_ids, params, **stamp, error=failure, images=pictures)
            raise
        tokens = tuple(int(token) for token in sequence.tokens)
        stop_reason = str(getattr(sequence, "stop_reason", "") or "")
        record = self._add(
            prompt_ids,
            params,
            **stamp,
            completion=tokens,
            logprobs=tuple(float(x) for x in (getattr(sequence, "logprobs", None) or [])),
            stop_reason=stop_reason,
            sample_ms=(time.perf_counter() - started) * 1000.0,
            cached_tokens=int(getattr(response, "prompt_cache_hit_tokens", 0) or 0),
            images=pictures,
        )
        self.spoke += len(tokens)
        sampled = Sampled(tokens, stop_reason, record.seq)
        if not pending.done():
            pending.set_result(sampled)
        return sampled

    def _fitting(
        self,
        prompt: Prompt,
        prompt_ids: tuple[int, ...],
        pinned: Any,
        wanted: int,
        stamp: dict[str, Any],
    ) -> Any:
        """Return the sampling params that fit the budget and the context, or record a refusal
        and raise it."""
        if self.budget:
            if self.spoke >= self.budget:
                self._refuse(prompt_ids, pinned, BUDGET, stamp)
                raise Refused(
                    f"This trial has spent its budget: {self.spoke} sampled tokens of "
                    f"{self.budget}, the model's context length. No further calls are answered."
                )
            if int(getattr(pinned, "max_tokens", 0) or 0) > self.budget - self.spoke:
                pinned = pinned.model_copy(update={"max_tokens": int(self.budget - self.spoke)})
        if not self.max_context:
            return pinned
        length, wants = int(prompt.model_input.length), int(getattr(pinned, "max_tokens", 0) or 0)
        if length + wants <= self.max_context:
            return pinned
        if self.fill_context and length < self.max_context:
            return pinned.model_copy(update={"max_tokens": self.max_context - length})
        self._refuse(prompt_ids, pinned, CONTEXT, stamp)
        # What a harness compacts its history on.
        raise Refused(
            f"prompt is too long: {length} prompt tokens plus {wanted or wants} max_tokens "
            "exceed the model's context window"
        )

    def _refuse(
        self, prompt_ids: tuple[int, ...], pinned: Any, why: str, stamp: dict[str, Any]
    ) -> None:
        self._add(prompt_ids, _params_dict(pinned), **stamp, error=why)

    def _add(
        self,
        prompt_ids: tuple[int, ...],
        params: dict[str, Any],
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
        """Append the trial's next record, numbered from 1, refused and failed calls included."""
        self._seq += 1
        record = Record(
            seq=self._seq,
            at=time.time(),
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion,
            inference_logprobs=logprobs,
            stop_reason=stop_reason,
            sampling_params=params,
            requested_model=requested_model,
            sample_ms=sample_ms,
            weights=self.weights,
            cached_tokens=cached_tokens,
            request_id=request_id,
            bridged=bridged,
            prompt_digests=prompt_digests,
            reply_digest=None,
            error=error,
            images=images,
        )
        self.records.append(record)
        return record

    def reply(
        self,
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
        for at in range(len(self.records) - 1, -1, -1):
            if self.records[at].seq != seq:
                continue
            record = self.records[at] = replace(self.records[at], reply_digest=reply_digest)
            if bridgeable and pictured(record):
                reply = Reply(
                    prompt_ids=record.prompt_token_ids,
                    completion_ids=record.completion_token_ids,
                    ended_with_stop=ended_with_stop,
                    rendered=record.prompt_token_ids if rendered is None else rendered,
                    pictures=record.images,
                )
                self.index.add(record.prompt_digests, reply_digest, reply, call_ids)
            return


def _params_dict(sampling_params: Any) -> dict[str, Any]:
    dump = getattr(sampling_params, "model_dump", None)
    if not callable(dump):
        return {}
    try:
        return {key: value for key, value in dump().items() if value is not None}
    except Exception:  # noqa: BLE001 - a failed annotation must not fail the call
        return {}
