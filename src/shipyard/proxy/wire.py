"""Definitions shared by both ends of the proxy, independent of the cookbook: the record
as stored and as sent over the wire, the routes, the env var names and the token. On the
wire, a prompt that extends an earlier record is sent as that record's index plus the new
tokens and is rebuilt in full on read."""

from __future__ import annotations

import base64
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any

#: The harness's key and the run's control key. They are env vars because the config is
#: copied into the run directory, and a secret in the config would be copied with it.
PROXY_TOKEN_ENV = "SHIPYARD_PROXY_TOKEN"
CONTROL_TOKEN_ENV = "SHIPYARD_CONTROL_TOKEN"
RECORDS_PATH = "/records"
CONTROL_PATH = "/control/weights"
#: The route that needs no token. A run probes it before a sandbox dials the proxy.
HEALTH_PATH = "/healthz"
#: Prefix of `error` on a record whose request the sampler failed: `sampler: <type>`.
SAMPLER_FAILED = "sampler: "
#: Placeholder id for an image's tokens in a record's prompt ids, one per token of the
#: chunk, so the record keeps the prompt's length. The image itself is a `Picture` on the
#: record. Admission sets aside a rollout whose record did not keep one.
IMAGE_TOKEN = -1


def new_token() -> str:
    return secrets.token_urlsafe(24)


@dataclass(frozen=True)
class Picture:
    """An image as the model read it: where its tokens sit in the prompt ids, how many
    there are, the format, and the bytes the renderer sent, base64 for the wire."""

    offset: int
    length: int
    format: str
    data: str


@dataclass(frozen=True)
class Record:
    """One model call of one trial as token ids. A record with `error` set was refused or
    failed and produced nothing. It is kept so the completeness check sees the call.
    `images` holds the pictures behind the prompt's `IMAGE_TOKEN` runs, when the proxy had
    their bytes."""

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
    images: tuple[Picture, ...] = ()


def image_runs(ids: Sequence[int]) -> list[tuple[int, int]]:
    """Each run of `IMAGE_TOKEN` in `ids` as (offset, length), in order."""
    runs: list[tuple[int, int]] = []
    at = 0
    while at < len(ids):
        if ids[at] != IMAGE_TOKEN:
            at += 1
            continue
        start = at
        while at < len(ids) and ids[at] == IMAGE_TOKEN:
            at += 1
        runs.append((start, at - start))
    return runs


def pictured(record: Record) -> bool:
    """Whether every image in the record's prompt is kept: each `IMAGE_TOKEN` run has a
    picture of exactly its place and length, and the completion holds none."""
    if IMAGE_TOKEN in record.completion_token_ids:
        return False
    kept = {(one.offset, one.length) for one in record.images}
    return all(run in kept for run in image_runs(record.prompt_token_ids))


def ids_of(prompt: Any) -> tuple[int, ...]:
    """The prompt as ids. Each non-text chunk becomes a run of `IMAGE_TOKEN` of the
    chunk's length, because `ModelInput.to_ints` refuses image chunks and the record
    must still show the length."""
    ids: list[int] = []
    for chunk in prompt.chunks:
        tokens = getattr(chunk, "tokens", None)
        if tokens is not None:
            ids.extend(int(token) for token in tokens)
        else:
            ids.extend([IMAGE_TOKEN] * int(getattr(chunk, "length", 0) or 0))
    return tuple(ids)


def pictures_of(prompt: Any) -> tuple[Picture, ...]:
    """The prompt's images, each at the offset its `IMAGE_TOKEN` run has in `ids_of`. A
    chunk that carries no bytes (an asset pointer) leaves no picture, and admission sets
    the rollout aside."""
    found: list[Picture] = []
    at = 0
    for chunk in prompt.chunks:
        tokens = getattr(chunk, "tokens", None)
        if tokens is not None:
            at += len(tokens)
            continue
        length = int(getattr(chunk, "length", 0) or 0)
        data = getattr(chunk, "data", None)
        if isinstance(data, bytes | bytearray) and length:
            encoded = _encoded(bytes(data))
            found.append(Picture(at, length, str(getattr(chunk, "format", "jpeg")), encoded))
        at += length
    return tuple(found)


@lru_cache(maxsize=256)
def _encoded(data: bytes) -> str:
    """The bytes as base64, one string per distinct image: a harness re-sends the same image
    with every call of a trial, and the records then share it instead of each holding a copy."""
    return base64.b64encode(data).decode("ascii")


def model_input(tokens: Sequence[int], images: Sequence[Picture] = ()) -> Any:
    """The tokens as the model reads them, a `tinker.ModelInput`: text chunks, with an
    image chunk in place of each `IMAGE_TOKEN` run. Without images, `from_ints`. Raises
    when a run has no picture, or a picture no run, since the model would then read a
    prompt the policy never saw."""
    import tinker

    if not images:
        if IMAGE_TOKEN in tokens:
            raise ValueError("a sequence with an image's tokens needs the picture behind them")
        return tinker.ModelInput.from_ints(list(tokens))
    placed = sorted(images, key=lambda one: one.offset)
    runs = image_runs(tokens)
    if [(one.offset, one.length) for one in placed] != runs:
        raise ValueError(
            f"the pictures sit at {[(one.offset, one.length) for one in placed]} but the "
            f"image tokens at {runs}"
        )
    chunks: list[Any] = []
    at = 0
    for picture in placed:
        if picture.offset > at:
            chunks.append(tinker.types.EncodedTextChunk(tokens=list(tokens[at : picture.offset])))
        chunks.append(
            tinker.types.ImageChunk(
                data=base64.b64decode(picture.data),
                format=picture.format,  # type: ignore[arg-type]
                expected_tokens=picture.length,
            )
        )
        at = picture.offset + picture.length
    if at < len(tokens):
        chunks.append(tinker.types.EncodedTextChunk(tokens=list(tokens[at:])))
    return tinker.ModelInput(chunks=chunks)


def delta_encoded(records: Sequence[Record]) -> list[dict[str, Any]]:
    """Each record as a dict. A prompt that begins with an earlier record's prompt and
    completion is sent as `prompt_from` (the longest such record) plus the tokens after
    them."""
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
            # The pictures inside the prefix travel with the record it extends.
            row["images"] = [one for one in row["images"] if int(one["offset"]) >= best_end]
        out.append(row)
    return out


def records_of(items: Sequence[Mapping[str, Any]]) -> list[Record]:
    """A trial's records decoded from the wire, in either form the route sends them. The
    `prompt_from` link is dropped. The run finds prefixes from the ids."""
    made: list[Record] = []
    for item in items:
        record = record_of(item)
        base = item.get("prompt_from")
        if base is not None:
            if not 0 <= int(base) < len(made):
                raise ValueError(
                    f"A record off the wire extends record {base}, but only {len(made)} "
                    "came before it. The response is not one trial's records in order."
                )
            before = made[int(base)]
            record = replace(
                record,
                prompt_token_ids=(
                    *before.prompt_token_ids,
                    *before.completion_token_ids,
                    *record.prompt_token_ids,
                ),
                images=(*before.images, *record.images),
            )
        made.append(record)
    return made


def record_of(payload: Mapping[str, Any]) -> Record:
    """One record off the wire. The ids are required, because empty defaults would give
    training a turn the policy never took. Every other field is an annotation."""
    missing = [key for key in ("prompt_token_ids", "completion_token_ids") if key not in payload]
    if missing:
        raise ValueError(
            f"A record off the wire carries no {' or '.join(missing)}, so there is nothing "
            f"in it to train on. Its keys: {sorted(payload)}."
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
        images=tuple(_picture(item) for item in payload.get("images") or []),
    )


def _picture(payload: Mapping[str, Any]) -> Picture:
    return Picture(
        offset=int(payload["offset"]),
        length=int(payload["length"]),
        format=str(payload.get("format") or "jpeg"),
        data=str(payload.get("data") or ""),
    )


def _text(value: Any) -> str | None:
    """The value as a string, or None when it is empty or missing."""
    return str(value) if value else None
