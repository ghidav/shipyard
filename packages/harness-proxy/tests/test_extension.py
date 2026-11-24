"""Tests for `Extension`, the form a trial keeps a prompt in when it extends an earlier call's:
that it reads as the tuple of all its tokens, that the wire and the question of what extends
what are the ones flat prompts gave (the flat code is kept here as the reference), and that a
trial holds a conversation's tokens once, in the proxy while it is open and in the run after."""

from __future__ import annotations

import copy
import gc
import json
import random
import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict, fields
from typing import Any

import pytest
import tinker

from harness_proxy.chat import Caller
from harness_proxy.rendering import Prompt
from harness_proxy.trial import Trial
from harness_proxy.wire import (
    Extension,
    Record,
    delta_encoded,
    extended,
    extending,
    extends,
    pictured,
    records_answer,
    records_in,
    records_of,
)
from proxies import FakeResponse, FakeSequence, made

# ----------------------------------------------------------------- it reads as a tuple


def _stacked() -> tuple[Extension, tuple[int, ...]]:
    """Return an extension three calls deep and the tuple it reads as."""
    first = Extension((1, 2, 3), (4, 5), (6,))
    second = Extension(first, (7,), ())
    third = Extension(second, (), (8, 9, 10))
    return third, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def test_an_extension_reads_as_the_tuple_of_all_its_tokens() -> None:
    held, whole = _stacked()
    assert len(held) == len(whole) and tuple(held) == whole and list(held) == list(whole)
    for at in range(-len(whole), len(whole)):
        assert held[at] == whole[at]
    for at in (len(whole), -len(whole) - 1):
        with pytest.raises(IndexError):
            held[at]
    edges = [None, *range(-len(whole) - 2, len(whole) + 3)]
    for start in edges:
        for stop in edges:
            assert held[start:stop] == whole[start:stop], (start, stop)
            assert isinstance(held[start:stop], tuple)
    assert held[::2] == whole[::2] and held[::-1] == whole[::-1] and held[8:1:-3] == whole[8:1:-3]
    assert 7 in held and 1 in held and 10 in held and 11 not in held
    assert held.index(6) == 5 and held.count(4) == 1


def test_an_extension_equals_the_tuple_it_reads_as_and_hashes_with_it() -> None:
    held, whole = _stacked()
    again, _ = _stacked()
    assert held == whole and whole == held and held == again and hash(held) == hash(whole)
    assert held != whole[:-1] and whole[:-1] != held and held != (*whole[:-1], 0)
    assert held != list(whole), "it is a tuple to compare, as a record's prompt was"
    beside = Extension(held.base, held.completion, (8, 9, 11))
    assert held != beside and Extension(held.base, held.completion, (8, 9, 10)) == held
    assert {held: 1}[whole] == 1
    assert copy.copy(held) is held and copy.deepcopy(held) is held
    assert "10 tokens, 3 of them its own" in repr(held)


def test_a_conversation_of_thousands_of_calls_is_read_without_recursion() -> None:
    held: Sequence[int] = (0,)
    for call in range(1, 5001):
        held = Extension(held, (call,), (-call,))
    assert len(held) == 10001 and held[0] == 0 and held[-1] == -5000 and held[1:3] == (1, -1)
    whole = tuple(held)
    assert whole[9997:] == (4999, -4999, 5000, -5000) and held[4000:4004] == whole[4000:4004]
    assert -1234 in held and 5001 not in held and held == whole


# ------------------------------------------------- the same records, wire and extensions


def _flat_extends(record: Record, earlier: Record) -> bool:
    """`extends`, as it compared whole prompts."""
    head = len(earlier.prompt_token_ids)
    end = head + len(earlier.completion_token_ids)
    prompt = record.prompt_token_ids
    return (
        len(prompt) >= end
        and prompt[head:end] == earlier.completion_token_ids
        and prompt[:head] == earlier.prompt_token_ids
        and tuple(one for one in record.images if one.offset < end) == earlier.images
    )


def _flat_delta_encoded(records: Sequence[Record]) -> list[dict[str, Any]]:
    """`delta_encoded`, as it searched whole prompts."""
    out: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = {one.name: getattr(record, one.name) for one in fields(Record)}
        row["images"] = [asdict(one) for one in record.images]
        best, best_end = None, 0
        for earlier in range(index - 1, -1, -1):
            before = records[earlier]
            end = len(before.prompt_token_ids) + len(before.completion_token_ids)
            if end > best_end and _flat_extends(record, before):
                best, best_end = earlier, end
        prompt = record.prompt_token_ids
        if best is not None:
            row["prompt_from"] = best
            row["prompt_token_ids"] = list(prompt[best_end:])
            row["images"] = [one for one in row["images"] if int(one["offset"]) >= best_end]
        out.append(row)
    return out


def _trial(rng: random.Random, calls: int) -> list[Record]:
    """Return a trial's records with whole prompts: conversations that go on, fork, restart,
    are sampled twice and fail, over so few token ids that prompts match by chance too."""

    def some(most: int, least: int = 0) -> tuple[int, ...]:
        return tuple(rng.randrange(3) for _ in range(rng.randint(least, most)))

    records: list[Record] = []
    for seq in range(1, calls + 1):
        kind = rng.random()
        if records and kind < 0.6:
            before = rng.choice(records[-3:]) if rng.random() < 0.7 else rng.choice(records)
            prompt = (*before.prompt_token_ids, *before.completion_token_ids, *some(4))
        elif records and kind < 0.75:
            prompt = tuple(rng.choice(records).prompt_token_ids)  # the same prompt, sampled again
        else:
            prompt = some(5, least=1)
        failed = rng.random() < 0.1
        records.append(
            made(prompt, some(3), seq=seq, error="sampler: ConnectError" if failed else None)
        )
    return records


def _kept(records: Sequence[Record]) -> list[Record]:
    """Return the records as a trial keeps them: each added to those before it."""
    kept: list[Record] = []
    for record in records:
        kept.append(extending(record, kept))
    return kept


@pytest.mark.parametrize("seed", range(40))
def test_extensions_change_no_record_no_row_of_the_wire_and_no_answer_to_what_extends(
    seed: int,
) -> None:
    rng = random.Random(seed)
    whole = _trial(rng, calls=rng.randint(1, 30))
    kept = _kept(whole)
    rows = _flat_delta_encoded(whole)
    assert kept == whole, "a record kept as an extension is the record"
    assert delta_encoded(kept) == rows == delta_encoded(whole)
    back = records_of(json.loads(json.dumps(rows)))
    assert back == whole and delta_encoded(back) == rows
    for held in (kept, back):
        for index, record in enumerate(held):
            sent_as_a_tail = "prompt_from" in rows[index]
            assert isinstance(record.prompt_token_ids, Extension) == sent_as_a_tail
            assert pictured(record)
            for earlier in range(len(held)):
                assert extends(record, held[earlier]) == _flat_extends(
                    whole[index], whole[earlier]
                ), (index, earlier)
            found = extended(record, held[:index])
            assert found == rows[index].get("prompt_from")


def test_an_extension_whose_record_is_not_sent_with_it_crosses_whole() -> None:
    first, second = made("ab", "c"), made("abcd", "e", seq=2)
    (_, kept) = _kept([first, second])
    assert isinstance(kept.prompt_token_ids, Extension)
    (row,) = delta_encoded([kept])
    assert "prompt_from" not in row and row["prompt_token_ids"] == list(second.prompt_token_ids)
    assert json.loads(json.dumps(row))["prompt_token_ids"] == list(second.prompt_token_ids)
    assert records_of([row]) == [second]


# ---------------------------------------------------- a conversation's tokens, held once

CALLS, ADDED, WRITTEN = 200, 50, 10


def _live(build: Any) -> tuple[int, Any]:
    """Return the bytes still allocated by `build`, and what it returned to keep them."""
    gc.collect()
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        held = build()
        gc.collect()
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return after - before, held


class _Forgetful:
    """A sampler that answers each call with the next reply and keeps nothing of the call."""

    def __init__(self, replies: list[list[int]]) -> None:
        self.replies = iter(replies)

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        tokens = next(self.replies)
        return FakeResponse([FakeSequence(tokens, [-0.5] * len(tokens))])


async def _talked() -> Trial:
    """Return a trial after one conversation of `CALLS` calls, each prompt the one before it,
    its reply and `ADDED` new tokens, every token id its own integer."""
    fresh = iter(range(1000, 10**9))
    replies = [[next(fresh) for _ in range(WRITTEN)] for _ in range(CALLS)]
    trial = Trial("t", _Forgetful([list(reply) for reply in replies]))
    prompt: list[int] = []
    for call in range(CALLS):
        prompt = [
            *prompt,
            *(replies[call - 1] if call else ()),
            *(next(fresh) for _ in range(ADDED)),
        ]
        await trial.sample(
            Caller(trial="t"),
            Prompt(tinker.ModelInput.from_ints(list(prompt))),
            tinker.SamplingParams(max_tokens=WRITTEN),
        )
    return trial


async def test_an_open_trial_and_a_closed_one_hold_a_conversation_once() -> None:
    """Whole prompts cost a conversation its length squared: here 1.2M prompt tokens, 9.6 MB of
    tuple slots alone, for the 12 thousand tokens the conversation has. A trial of long
    conversations so cost the proxy and the run gigabytes."""
    tokens = CALLS * (ADDED + WRITTEN)
    summed = sum(call * (ADDED + WRITTEN) + ADDED for call in range(CALLS))
    assert summed > 1_000_000 and summed > 90 * tokens
    budget = 150 * tokens  # an integer, its slot, a logprob and its slot, and a record's own
    assert budget < summed * 8 / 5, "the budget is far under what whole prompts cost"

    tracemalloc.start()
    try:
        gc.collect()
        before, _ = tracemalloc.get_traced_memory()
        trial = await _talked()
        gc.collect()
        opened, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(trial.records) == CALLS
    assert sum(len(one.prompt_token_ids) for one in trial.records) == summed
    assert all(isinstance(one.prompt_token_ids, Extension) for one in trial.records[1:])
    assert opened - before < budget, (
        f"an open trial holds {(opened - before) / tokens:.0f} B a token"
    )

    body = json.dumps(records_answer(trial.records))
    closed, records = _live(lambda: records_in(json.loads(body)))
    assert records == trial.records and len(body) < 40 * tokens
    assert closed < budget, f"a closed trial holds {closed / tokens:.0f} B a token"
