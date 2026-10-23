from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from harness_proxy.wire import BUDGET, CONTEXT, Picture, Record, extends, pictured

#: The `error` of a call the proxy turned away because of the model's own output: a spent
#: budget or a full context. Any other `error` is a call that failed at the sampler.
TURNED_AWAY = (BUDGET, CONTEXT)


@dataclass(frozen=True)
class Chain:
    """One chain of answered calls in which the model wrote, as one sequence of token ids."""

    #: The last call's prompt and completion, which each earlier call's extends.
    tokens: tuple[int, ...]
    #: (start, end) of each completion, in chain order.
    spans: tuple[tuple[int, int], ...]
    #: The sampler's logprobs at the targets, as the proxy recorded them; None if a call kept none.
    sampled_logprobs: tuple[float, ...] | None
    images: tuple[Picture, ...]


class Records:
    """What the proxy recorded of one trial's model calls, in order, or that it recorded none,
    because a provider's model served the trial. Ask `recorded` first: any other question about
    a trial the proxy did not record raises `RuntimeError`."""

    def __init__(self, items: Iterable[Record]) -> None:
        self._items: tuple[Record, ...] | None = tuple(items)

    @classmethod
    def unrecorded(cls) -> Records:
        """Return the records of a trial a provider's model served: the proxy saw none of it."""
        found = cls(())
        found._items = None
        return found

    def __repr__(self) -> str:
        return "Records(unrecorded)" if self._items is None else f"Records(calls={len(self)})"

    @property
    def recorded(self) -> bool:
        """Whether the proxy recorded the trial; False when a provider's model served it."""
        return self._items is not None

    def __iter__(self) -> Iterator[Record]:
        return iter(self._held)

    def __len__(self) -> int:
        return len(self._held)

    @property
    def never_fit(self) -> bool:
        """Whether its first call was turned away for the context and no call was answered."""
        held = self._held
        return bool(held) and held[0].error == CONTEXT and not any(map(_answered, held))

    @property
    def out_of_budget(self) -> bool:
        """Whether a call was turned away because the trial's token budget was spent."""
        return any(one.error == BUDGET for one in self._held)

    @property
    def filled_context(self) -> bool:
        """Whether a call was answered and a call was turned away for the context."""
        held = self._held
        return any(map(_answered, held)) and any(one.error == CONTEXT for one in held)

    @property
    def last_failed(self) -> bool:
        """Whether its last call failed at the sampler: neither answered nor turned away."""
        held = self._held
        return bool(held) and _failed(held[-1])

    @property
    def last_answered(self) -> bool:
        """Whether its last call was answered."""
        held = self._held
        return bool(held) and _answered(held[-1])

    @property
    def lost_an_image(self) -> bool:
        """Whether an answered call's prompt held an image the proxy did not keep."""
        return any(not pictured(one) for one in self._held if _answered(one))

    @property
    def sampled_tokens(self) -> int:
        """Tokens the model wrote in its answered calls."""
        return sum(len(one.completion_token_ids) for one in self._held if _answered(one))

    @property
    def billed(self) -> dict[str, int]:
        """Tokens its calls cost, by kind: a call turned away never reached the sampler."""
        sent = [one for one in self._held if one.error not in TURNED_AWAY]
        return {
            "prefill": sum(len(one.prompt_token_ids) - cached(one) for one in sent),
            "cache": sum(cached(one) for one in sent),
            "sample": sum(len(one.completion_token_ids) for one in sent),
        }

    @property
    def turned_away(self) -> int:
        """How many of its calls the proxy turned away, for the budget or the context."""
        return sum(one.error in TURNED_AWAY for one in self._held)

    @property
    def bridged(self) -> int:
        """How many of its calls' prompts the proxy bridged."""
        return sum(one.bridged for one in self._held)

    def chains(self) -> list[Chain]:
        """Return each chain of its answered calls in which the model wrote, in order."""
        found: list[Chain] = []
        for chain in _linked(self._held):
            last = chain[-1]
            spans = tuple(_span(one) for one in chain if one.completion_token_ids)
            if not spans:
                continue
            logprobs = tuple(value for one in chain for value in one.inference_logprobs)
            complete = all(
                len(one.inference_logprobs) == len(one.completion_token_ids) for one in chain
            )
            tokens = (*last.prompt_token_ids, *last.completion_token_ids)
            found.append(Chain(tokens, spans, logprobs if complete else None, last.images))
        return found

    @property
    def _held(self) -> tuple[Record, ...]:
        if self._items is None:
            raise RuntimeError(
                "a provider's model served this trial, so the proxy recorded none of its calls: "
                "ask `records.recorded` first"
            )
        return self._items


def served_nothing(many: Iterable[Records]) -> str | None:
    """Return why the proxy served these trials nothing, or None if it served them."""
    held = list(many)
    if not held:
        return None
    made = [one for records in held for one in records]
    if not made:
        return "no trial reached the proxy; the endpoint is dead or unreachable"
    if not all(map(_failed, made)):
        return None
    failed = sorted({one.error or "" for one in made})
    return (
        f"every request failed at the sampler ({', '.join(failed)}); the weights it serves "
        "are gone or the backend refuses them"
    )


def cached(record: Record) -> int:
    """Return a record's cached prompt tokens, capped at the prompt's length."""
    return min(int(record.cached_tokens or 0), len(record.prompt_token_ids))


def _answered(record: Record) -> bool:
    return record.error is None


def _failed(record: Record) -> bool:
    return record.error is not None and record.error not in TURNED_AWAY


def _linked(records: Iterable[Record]) -> list[list[Record]]:
    """Link the answered records whose images were kept into chains by the token-prefix rule:
    each joins the longest chain whose last call its prompt extends."""
    found: list[list[Record]] = []
    for record in records:
        if not _answered(record) or not pictured(record):
            continue
        home: list[Record] | None = None
        longest = -1
        for chain in found:
            last = chain[-1]
            end = len(last.prompt_token_ids) + len(last.completion_token_ids)
            if end > longest and extends(record, last):
                home, longest = chain, end
        if home is None:
            found.append([record])
        else:
            home.append(record)
    return found


def _span(record: Record) -> tuple[int, int]:
    """Return where a record's completion sits in the chain: after its prompt."""
    head = len(record.prompt_token_ids)
    return head, head + len(record.completion_token_ids)
