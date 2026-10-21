from __future__ import annotations

import base64
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any
from urllib.parse import quote, unquote

#: Env vars holding the harness token and the run's control token. Secrets stay out of the
#: config because the config is copied into the run directory.
PROXY_TOKEN_ENV = "HARNESS_PROXY_TOKEN"
CONTROL_TOKEN_ENV = "HARNESS_PROXY_CONTROL_TOKEN"
RECORDS_PATH = "/records"
CONTROL_PATH = "/control/weights"
#: The only route that needs no token. A run probes it before a sandbox dials the proxy.
HEALTH_PATH = "/healthz"
#: Prefix of `error` on a record whose sampling call failed, as `sampler: <type>`.
SAMPLER_FAILED = "sampler: "
#: `error` on a record the proxy refused because of the policy's own output: a spent
#: budget or a full context.
BUDGET = "budget"
CONTEXT = "context"
#: Placeholder id for image tokens in a record's prompt ids, one per token of the image
#: chunk, so the record keeps the prompt's length. The image itself is a `Picture` on
#: the record. Admission sets aside a rollout whose record did not keep one.
IMAGE_TOKEN = -1


def new_token() -> str:
    return secrets.token_urlsafe(24)


@dataclass(frozen=True)
class Picture:
    """An image as the model read it: where its tokens sit in the prompt, and its bytes."""

    offset: int
    length: int
    format: str
    data: str


@dataclass(frozen=True)
class Record:
    """A trial's model call, as token ids, a failed call kept with its `error`."""

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
    """Return each run of `IMAGE_TOKEN` in `ids` as `(offset, length)`, in order."""
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
    """Return whether every image in the record's prompt was kept."""
    if IMAGE_TOKEN in record.completion_token_ids:
        return False
    kept = {(one.offset, one.length) for one in record.images}
    return all(run in kept for run in image_runs(record.prompt_token_ids))


def extends(record: Record, earlier: Record) -> bool:
    """Return whether `record`'s prompt extends `earlier`'s prompt and completion, images too."""
    head = len(earlier.prompt_token_ids)
    end = head + len(earlier.completion_token_ids)
    prompt = record.prompt_token_ids
    return (
        len(prompt) >= end
        and prompt[head:end] == earlier.completion_token_ids
        and prompt[:head] == earlier.prompt_token_ids
        and tuple(one for one in record.images if one.offset < end) == earlier.images
    )


def ids_of(prompt: Any) -> tuple[int, ...]:
    """Return the prompt's token ids, with each non-text chunk as a run of `IMAGE_TOKEN`."""
    ids: list[int] = []
    for chunk in prompt.chunks:
        tokens = getattr(chunk, "tokens", None)
        if tokens is not None:
            ids.extend(int(token) for token in tokens)
        else:
            ids.extend([IMAGE_TOKEN] * int(getattr(chunk, "length", 0) or 0))
    return tuple(ids)


def pictures_of(prompt: Any) -> tuple[Picture, ...]:
    """Return the prompt's images, each at the offset of its `IMAGE_TOKEN` run in `ids_of`."""
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
    """Return the bytes as base64, cached so each distinct image yields one shared string."""
    return base64.b64encode(data).decode("ascii")


def model_input(tokens: Sequence[int], images: Sequence[Picture] = ()) -> Any:
    """Build a `tinker.ModelInput` with an image chunk in place of each `IMAGE_TOKEN` run."""
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
    """Encode records as dicts for the wire, sending prompts as deltas where possible."""
    out: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = asdict(record)
        best, best_end = None, 0
        for earlier in range(index - 1, -1, -1):
            before = records[earlier]
            end = len(before.prompt_token_ids) + len(before.completion_token_ids)
            if end > best_end and extends(record, before):
                best, best_end = earlier, end
        prompt = record.prompt_token_ids
        if best is not None:
            row["prompt_from"] = best
            row["prompt_token_ids"] = list(prompt[best_end:])
            # The pictures inside the prefix travel with the record it extends.
            row["images"] = [one for one in row["images"] if int(one["offset"]) >= best_end]
        out.append(row)
    return out


def records_of(items: Sequence[Mapping[str, Any]]) -> list[Record]:
    """Decode a trial's records from the wire, in either form the records route sends."""
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
    """Decode one record from the wire; its token ids are required."""
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
    return str(value) if value else None


# ---------------------------------------------------------------------- the address

#: The names an address may give, in the order `Address.path` writes them.
ADDRESS_KEYS = ("lane", "trial", "renderer")


@dataclass(frozen=True)
class Address:
    """Where a trial's harness sends its calls: the trial they are recorded under, and the lane
    and renderer that answer them, if it names them."""

    trial: str
    #: None for a proxy's only lane.
    lane: str | None = None
    #: None for the lane's first renderer.
    renderer: str | None = None

    def __post_init__(self) -> None:
        """Refuse a name the path cannot hold."""
        named = (("trial name", self.trial), ("lane", self.lane), ("renderer", self.renderer))
        for what, name in named:
            if name is not None and (not name or "/" in name):
                raise ValueError(
                    f"{name!r} cannot be addressed: a {what} goes in the URL path, "
                    "so it must be non-empty and contain no '/'."
                )

    @property
    def path(self) -> str:
        """The path a harness's base URL ends in, each name percent-encoded:
        `/r/lane/<lane>/trial/<trial>/renderer/<renderer>/v1`, less the names it leaves out."""
        named = {"lane": self.lane, "trial": self.trial, "renderer": self.renderer}
        keys = "".join(
            f"/{key}/{quote(name, safe='')}" for key, name in named.items() if name is not None
        )
        return f"/r{keys}/v1"

    @classmethod
    def of(cls, path: str) -> Address:
        """Read the address a raw request path starts with, as `path` writes it; `ValueError` for
        one that names no trial, or a name no address gives."""
        parts, named, at = path.split("/"), {}, 2
        under = parts[:2] == ["", "r"]
        while under and at + 1 < len(parts) and parts[at] in ADDRESS_KEYS:
            if parts[at] in named:
                break
            named[parts[at]] = unquote(parts[at + 1])
            at += 2
        if not under or parts[at : at + 1] != ["v1"] or "trial" not in named:
            raise ValueError(
                f"{path!r} is no trial's address, which reads "
                "/r/[lane/<lane>/]trial/<trial>/[renderer/<renderer>/]v1"
            )
        return cls(**named)


# --------------------------------------------------------------------------- health


@dataclass(frozen=True)
class Served:
    """A lane, as the proxy's `/healthz` reports it."""

    #: The base model it serves.
    model: str
    #: What it serves: its weights, or its base model.
    serving: str
    #: Tokens one trial may sample; None when no budget is enforced.
    budget: int | None
    #: The renderers a trial's address may name; none for its model's default alone.
    renderers: tuple[str, ...]

    @classmethod
    def of(cls, reported: Mapping[str, Any]) -> Served:
        """Read a lane from `/healthz`, taking a budget that is not a positive count as none."""
        budget = reported.get("budget")
        known = isinstance(budget, int) and not isinstance(budget, bool) and budget > 0
        named = reported.get("renderers")
        return cls(
            model=str(reported.get("model") or ""),
            serving=str(reported.get("serving") or ""),
            budget=budget if known else None,
            renderers=tuple(str(one) for one in named) if isinstance(named, list) else (),
        )


def health(lanes: Mapping[str, Served]) -> dict[str, Any]:
    """Return what `/healthz` answers: that the proxy is up, and each lane as it serves it."""
    return {"status": "ok", "lanes": {name: asdict(one) for name, one in lanes.items()}}


def lanes_in(answer: Mapping[str, Any]) -> dict[str, Served]:
    """Read each lane back from what `/healthz` answered; none if it lists none."""
    lanes = answer.get("lanes")
    if not isinstance(lanes, Mapping):
        return {}
    return {str(name): Served.of(one) for name, one in lanes.items() if isinstance(one, Mapping)}


# ---------------------------------------------------------------------- the weights


@dataclass(frozen=True)
class Swap:
    """The run's request to point a lane at other weights, which `CONTROL_PATH` takes."""

    #: The weights, under the lane's backend's prefix; None for its base model.
    weights: str | None
    #: The lane it points; None for a proxy's only lane.
    lane: str | None = None

    @property
    def body(self) -> dict[str, Any]:
        return {"weights": self.weights, **({"lane": self.lane} if self.lane else {})}

    @classmethod
    def of(cls, body: Any) -> Swap:
        """Read a swap back from a request's JSON body; `ValueError` for one that is not."""
        if not isinstance(body, dict) or "weights" not in body:
            raise ValueError("expected a JSON object with 'weights'")
        weights, lane = body["weights"], body.get("lane")
        if weights is not None and not isinstance(weights, str):
            raise ValueError("'weights' must be a path or null")
        if lane is not None and not isinstance(lane, str):
            raise ValueError("'lane' must be the name of a lane")
        return cls(weights, lane or None)


# -------------------------------------------------------------------------- records


def records_path(trial: str) -> str:
    """The path the run fetches a trial's records at, the trial percent-encoded."""
    return f"{RECORDS_PATH}/{quote(trial, safe='')}"


@dataclass(frozen=True)
class Fetch:
    """How a records fetch asks: for prompts as deltas of the record each extends, and, on a
    retry, for the answer sent last, which the lost fetch had drained."""

    delta: bool = False
    again: bool = False

    @property
    def query(self) -> dict[str, str]:
        return {key: "1" for key, asked in (("delta", self.delta), ("again", self.again)) if asked}

    @classmethod
    def of(cls, query: Mapping[str, str]) -> Fetch:
        return cls(delta=query.get("delta") == "1", again=query.get("again") == "1")


def records_answer(records: Sequence[Record], *, delta: bool) -> dict[str, Any]:
    """Return what the records route answers: the records, their prompts as deltas if asked."""
    return {"records": delta_encoded(records) if delta else [asdict(one) for one in records]}


def records_in(answer: Mapping[str, Any]) -> list[Record]:
    """Read the records back from what the records route answered."""
    return records_of(answer.get("records") or [])


# -------------------------------------------------------------------------- startup

#: The first line serve prints, which names its lanes and the port the kernel chose.
BANNER = re.compile(
    r"^serving (?P<lanes>\S+) at http://(?P<host>[^:/]+):(?P<port>\d+) "
    r"\(control: (?P<control>on|off)\)$"
)
#: The line serve prints once its tunnel is up, after the banner.
TUNNEL_LINE = re.compile(r"^tunnel at (?P<origin>https://\S+)$")


def banner(lanes: Sequence[str], host: str, port: int, *, control: bool) -> str:
    """Return the first line serve prints, which `BANNER` reads."""
    named, opened = ",".join(lanes), "on" if control else "off"
    return f"serving {named} at http://{host}:{port} (control: {opened})"


def tunnel_line(origin: str) -> str:
    """Return the line serve prints once its tunnel is up, which `TUNNEL_LINE` reads."""
    return f"tunnel at {origin}"
