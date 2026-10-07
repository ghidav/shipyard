"""What both ends of the proxy agree on without the cookbook: the record as it is kept
and as it crosses the wire (a prompt extending an earlier record sent as that record's
index and the new tokens, read back whole), the routes, the env names, the token."""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

#: The harness's key and the run's control key: env vars and never config keys, since the
#: config is copied into the run directory and a secret written there is in every copy.
PROXY_TOKEN_ENV = "SHIPYARD_PROXY_TOKEN"
CONTROL_TOKEN_ENV = "SHIPYARD_CONTROL_TOKEN"
RECORDS_PATH = "/records"
CONTROL_PATH = "/control/weights"
#: The one route behind no token: what a run probes before a sandbox dials the proxy.
HEALTH_PATH = "/healthz"
#: How a record of a request the sampler failed begins its `error`: `sampler: <type>`.
SAMPLER_FAILED = "sampler: "
#: Stands in for an image's tokens in a record's prompt ids, one per token of the chunk:
#: the record still says how long the prompt was, and admission sets such a rollout aside.
IMAGE_TOKEN = -1


def new_token() -> str:
    return secrets.token_urlsafe(24)


@dataclass(frozen=True)
class Record:
    """One model call of one trial as token ids; `error` set means it was refused or failed
    and produced nothing, kept so completeness can see the call was made."""

    seq: int
    at: float
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    inference_logprobs: tuple[float, ...]
    stop_reason: str
    sampling_params: dict[str, Any]
    requested_model: str | None
    sample_ms: float
    served: str | None
    cached_tokens: int
    request_id: str | None
    bridged: bool
    prompt_digests: tuple[str, ...]
    reply_digest: str | None
    error: str | None


def delta_encoded(records: Sequence[Record]) -> list[dict[str, Any]]:
    """Each record as a dict; a prompt that begins with an earlier record's prompt and
    completion carries `prompt_from` and only the tokens after them (the longest match)."""
    out: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = asdict(record)
        prompt = record.prompt_token_ids
        best, best_end = None, 0
        for earlier in range(index - 1, -1, -1):
            before = records[earlier]
            head = len(before.prompt_token_ids)
            end = head + len(before.completion_token_ids)
            if end <= best_end or len(prompt) < end:
                continue
            if (
                prompt[head:end] == before.completion_token_ids
                and prompt[:head] == before.prompt_token_ids
            ):
                best, best_end = earlier, end
        if best is not None:
            row["prompt_from"] = best
            row["prompt_token_ids"] = list(prompt[best_end:])
        out.append(row)
    return out


def records_of(items: Sequence[Mapping[str, Any]]) -> list[Record]:
    """A trial's records back off the wire, in either form the route sends them. The
    `prompt_from` link is not kept: the run finds its own prefixes over the ids."""
    made: list[Record] = []
    for item in items:
        record = record_of(item)
        base = item.get("prompt_from")
        if base is not None:
            if not 0 <= int(base) < len(made):
                raise ValueError(
                    f"A record off the wire extends record {base}, and only {len(made)} "
                    "came before it: the response is not one trial's records in order."
                )
            before = made[int(base)]
            record = replace(
                record,
                prompt_token_ids=(
                    *before.prompt_token_ids,
                    *before.completion_token_ids,
                    *record.prompt_token_ids,
                ),
            )
        made.append(record)
    return made


def record_of(payload: Mapping[str, Any]) -> Record:
    """One record off the wire. The ids are required: defaulting them to empty would hand
    training a turn the policy never took; every other field is an annotation."""
    missing = [key for key in ("prompt_token_ids", "completion_token_ids") if key not in payload]
    if missing:
        raise ValueError(
            f"A record off the wire carries no {' or '.join(missing)}, so there is nothing "
            f"in it to train on; it had {sorted(payload)}."
        )
    return Record(
        seq=int(payload.get("seq") or 0),
        at=float(payload.get("at") or 0.0),
        prompt_token_ids=tuple(int(token) for token in payload["prompt_token_ids"]),
        completion_token_ids=tuple(int(token) for token in payload["completion_token_ids"]),
        inference_logprobs=tuple(float(x) for x in payload.get("inference_logprobs") or []),
        stop_reason=str(payload.get("stop_reason") or ""),
        sampling_params=dict(payload.get("sampling_params") or {}),
        requested_model=_text(payload.get("requested_model")),
        sample_ms=float(payload.get("sample_ms") or 0.0),
        served=_text(payload.get("served")),
        cached_tokens=int(payload.get("cached_tokens") or 0),
        request_id=_text(payload.get("request_id")),
        bridged=bool(payload.get("bridged") or False),
        prompt_digests=tuple(str(digest) for digest in payload.get("prompt_digests") or []),
        reply_digest=_text(payload.get("reply_digest")),
        error=_text(payload.get("error")),
    )


def _text(value: Any) -> str | None:
    """None rather than "": an empty string is how a wire that carries nothing arrives."""
    return str(value) if value else None
