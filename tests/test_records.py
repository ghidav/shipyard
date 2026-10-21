"""Tests for `shipyard.records`: what the proxy recorded of a trial's model calls, read in one
place: the facts a verdict reads, the chains a trial trains on, what its calls cost, and whether
the proxy served a job anything."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from harness_proxy.wire import BUDGET, CONTEXT, IMAGE_TOKEN, Picture, Record
from shipyard.records import Chain, Records, served_nothing
from tests.fakes import made

FAILED = "sampler: ConnectError"
DEAD = "no trial reached the proxy; the endpoint is dead or unreachable"
P1, C1, T1, C2, T2, C3 = (1, 2, 3), (10, 11), (20,), (12,), (21, 22), (13, 14)
PNG = base64.b64encode(b"png-bytes").decode()
#: An image that spans three prompt tokens starting at offset 1.
SHOT = Picture(1, 3, "png", PNG)


def _record(prompt: tuple[int, ...], completion: tuple[int, ...], seq: int, **named) -> Record:
    found = made(prompt, completion, seq=seq, **named)
    scored = tuple(-float(token) / 10 for token in completion)
    return replace(found, inference_logprobs=scored if found.error is None else ())


def _tool_loop() -> list[Record]:
    """Return three tool-loop records, each prompt extending the previous prompt and completion."""
    first = _record(P1, C1, 1)
    second = _record((*P1, *C1, *T1), C2, 2)
    third = _record((*P1, *C1, *T1, *C2, *T2), C3, 3)
    return [first, second, third]


def _chains(*records: Record) -> list[Chain]:
    return Records(records).chains()


def _tokens(*records: Record) -> list[tuple[int, ...]]:
    """Return each chain's tokens: its last call's prompt and completion."""
    return [one.tokens for one in _chains(*records)]


def _written(chain: Chain) -> list[int]:
    return [chain.tokens[at] for start, end in chain.spans for at in range(start, end)]


# ------------------------------------------------------------------ recorded or not


def test_a_trial_a_provider_served_has_no_records_to_ask_about() -> None:
    held = Records.unrecorded()
    assert not held.recorded and repr(held) == "Records(unrecorded)"
    asked: list[Callable[[], Any]] = [
        lambda: len(held),
        lambda: list(held),
        lambda: held.never_fit,
        lambda: held.sampled_tokens,
        lambda: held.billed,
        lambda: held.weights,
        held.chains,
        lambda: served_nothing([held]),
    ]
    for ask in asked:
        with pytest.raises(RuntimeError, match="ask `records.recorded` first"):
            ask()
    empty = Records([])
    assert empty.recorded and len(empty) == 0 and list(empty) == [], "the proxy saw no call"
    assert repr(Records([made()])) == "Records(calls=1)"


# --------------------------------------------------------------- what a verdict reads


def test_a_first_call_that_never_fit_with_nothing_answered() -> None:
    assert Records([made(error=CONTEXT)]).never_fit
    assert not Records([made(error=CONTEXT), made(seq=2)]).never_fit, "a later one was answered"
    assert not Records([made(), made(seq=2, error=CONTEXT)]).never_fit
    assert not Records([]).never_fit


def test_a_spent_budget_and_a_context_the_model_filled() -> None:
    cut = Records([made(seq=1), made(seq=2, error=BUDGET)])
    assert cut.out_of_budget and not cut.filled_context
    filled = Records([made(seq=1), made(seq=2, error=CONTEXT)])
    assert filled.filled_context and not filled.out_of_budget
    assert not Records([made(error=CONTEXT)]).filled_context, "nothing was answered"


def test_a_last_call_that_failed_at_the_sampler_or_was_answered() -> None:
    failed = Records([made(seq=1), made(seq=2, error=FAILED)])
    assert failed.last_failed and not failed.last_answered
    recovered = Records([*failed, made(seq=3)])
    assert recovered.last_answered and not recovered.last_failed
    refused = Records([made(seq=1), made(seq=2, error=CONTEXT)])
    assert not refused.last_failed and not refused.last_answered, "a refusal is neither"
    assert not Records([]).last_failed and not Records([]).last_answered


def test_an_answered_call_that_lost_an_image() -> None:
    lost = made((1, IMAGE_TOKEN, IMAGE_TOKEN, 2), seq=2)
    assert Records([made(seq=1), lost]).lost_an_image
    kept = replace(lost, images=(Picture(1, 2, "png", "cG5nIQ=="),))
    assert not Records([made(seq=1), kept]).lost_an_image, "a kept image"
    assert not Records([replace(lost, error=FAILED)]).lost_an_image, "only an answered call's"


def test_the_model_wrote_only_in_its_answered_calls() -> None:
    held = Records(
        [
            made("go", "ok", seq=1),
            made("go", error=CONTEXT, seq=2),
            made(seq=3, error=FAILED),
            made("goo", "fine", seq=4),
        ]
    )
    assert held.sampled_tokens == 2 + 4


# --------------------------------------------------------- the chains it trains on


def test_a_tool_loop_chains_into_one_sequence_with_three_target_spans() -> None:
    [found] = _chains(*_tool_loop())
    assert found.tokens == (*P1, *C1, *T1, *C2, *T2, *C3)
    assert found.spans == ((3, 5), (6, 7), (9, 11))
    assert _written(found) == [*C1, *C2, *C3]
    assert found.sampled_logprobs == (-1.0, -1.1, -1.2, -1.3, -1.4)


def test_a_prompt_that_does_not_extend_opens_a_second_chain() -> None:
    """The fourth call re-sends the history with thinking stripped, so its prompt shares a
    prefix with the chain and then diverges. It starts a new chain, and each completion is
    still a target exactly once."""
    stripped = _record((*P1, 10, 99, *T1, *C2, *T2, *C3, 23), (15,), 4)
    found = _chains(*_tool_loop(), stripped)
    assert len(found) == 2
    assert found[1].tokens == (*stripped.prompt_token_ids, 15)
    assert found[1].spans == ((len(stripped.prompt_token_ids), len(stripped.prompt_token_ids) + 1),)
    assert sorted(token for one in found for token in _written(one)) == sorted([*C1, *C2, *C3, 15])
    assert sum(end - start for one in found for start, end in one.spans) == 6


def test_a_side_question_is_filed_under_its_own_chain_and_the_longest_match_wins() -> None:
    main1 = _record((1, 2), (3,), 1)
    main2 = _record((1, 2, 3, 4), (5,), 2)
    side = _record((7, 8), (9,), 3)
    main3 = _record((1, 2, 3, 4, 5, 6), (10,), 4)
    side2 = _record((7, 8, 9, 11), (12,), 5)
    main, aside = _chains(main1, main2, side, main3, side2)
    assert (main.tokens, aside.tokens) == ((1, 2, 3, 4, 5, 6, 10), (7, 8, 9, 11, 12))
    assert main.spans == ((2, 3), (4, 5), (6, 7)) and aside.spans == ((2, 3), (4, 5))
    first = _record((1, 2), (3,), 1)
    resampled = _record((1, 2), (4,), 2)
    onward = _record((1, 2, 4, 5), (6,), 3)
    summary = _record((20, 21), (22,), 4)
    found = _chains(first, resampled, onward, summary)
    assert [one.tokens for one in found] == [(1, 2, 3), (1, 2, 4, 5, 6), (20, 21, 22)]
    assert found[1].spans == ((2, 3), (4, 5)), "onward extends the resample, not the first"
    assert _chains() == []
    shorter = _record((1, 2, 3), (4, 5), 1), _record((1, 2, 3, 4), (6,), 2)
    assert _tokens(*shorter) == [(1, 2, 3, 4, 5), (1, 2, 3, 4, 6)]


def test_refused_and_failed_calls_and_images_without_their_picture_never_chain() -> None:
    first, second, third = _tool_loop()
    refused = _record((*P1, *C1, *T1), (), 9, error=CONTEXT)
    failed = _record((*P1, *C1, *T1), (), 11, error=FAILED)
    unkept = _record((IMAGE_TOKEN, IMAGE_TOKEN, 5), (6,), 10)
    assert _chains(refused) == _chains(failed) == _chains(unkept) == []
    assert _tokens(first) == [(*P1, *C1)]
    [found] = _chains(first, refused, second, failed, unkept, third)
    assert found.spans == ((3, 5), (6, 7), (9, 11))


def test_a_kept_image_chains_and_the_chain_carries_the_picture() -> None:
    seen = _record((1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 5), (6, 7), 1, images=(SHOT,))
    assert _tokens(seen) == [(1, -1, -1, -1, 5, 6, 7)]
    again = _record((*seen.prompt_token_ids, 6, 7, 8), (9,), 2, images=(SHOT,))
    [found] = _chains(seen, again)
    assert found.tokens == (1, -1, -1, -1, 5, 6, 7, 8, 9) and found.images == (SHOT,)
    assert found.spans == ((5, 7), (8, 9))
    shorter = replace(seen, images=(Picture(1, 2, "png", PNG),))
    assert _chains(shorter) == [], "a picture of another length is not this image"


def test_another_picture_in_the_prefix_starts_its_own_chain() -> None:
    """Two images of one size are the same run of `IMAGE_TOKEN`, so the ids alone would chain
    them, and the chain would train the first call on the second call's image."""
    seen = _record((1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 5), (6, 7), 1, images=(SHOT,))
    other = Picture(1, 3, "png", "b3RoZXI=")
    swapped = _record((*seen.prompt_token_ids, 6, 7, 8), (9,), 2, images=(other,))
    assert len(_chains(seen, swapped)) == 2
    kept = _record((*seen.prompt_token_ids, 6, 7, 8), (9,), 2, images=(SHOT,))
    assert len(_chains(seen, kept)) == 1


def test_a_call_without_logprobs_leaves_the_chain_with_none() -> None:
    first, second, third = _tool_loop()
    bare = replace(second, inference_logprobs=())
    [found] = _chains(first, bare, third)
    assert found.sampled_logprobs is None and found.spans == ((3, 5), (6, 7), (9, 11))
    silent = replace(first, completion_token_ids=(), inference_logprobs=())
    assert _chains(silent) == [], "a chain in which the model wrote nothing"


# --------------------------------------------------------------------- what it cost


def test_a_call_turned_away_costs_nothing_and_a_failed_one_its_prompt() -> None:
    held = Records(
        [
            made("one", "ok", seq=1),
            made("oneoktwo", "fine", seq=2, cached=5, bridged=True),
            made("go", error=CONTEXT, seq=3),
            made("go", error=BUDGET, seq=4, cached=9),
            made(seq=5, error=FAILED),
        ]
    )
    # 13 prompt tokens reached the sampler, 5 of them cached; the two refusals never did.
    assert held.billed == {"prefill": 8, "cache": 5, "sample": 6}
    assert (held.turned_away, held.bridged) == (2, 1), "a sampler failure is not a refusal"


def test_a_cache_hit_bills_no_more_than_its_prompt() -> None:
    assert Records([made("go", cached=99)]).billed == {"prefill": 0, "cache": 2, "sample": 2}


def test_the_weights_that_served_its_calls() -> None:
    held = Records(
        [
            made(served="tinker://w/step-2"),
            made(seq=2, served="tinker://w/step-2", error=BUDGET),
            made(seq=3),
        ]
    )
    assert held.weights == {"tinker://w/step-2"}
    assert Records([made(served=None)]).weights == frozenset(), "the base model names none"


# ----------------------------------------------------------- what a job was served


def test_served_nothing_is_about_records_not_rewards() -> None:
    assert served_nothing([Records([]), Records([])]) == DEAD
    failed = [made(error=FAILED), made(error="sampler: BadRequestError")]
    assert served_nothing([Records(failed)]) == (
        "every request failed at the sampler (sampler: BadRequestError, sampler: ConnectError); "
        "the weights it serves are gone or the backend refuses them"
    )
    assert served_nothing([Records([made(error=CONTEXT)])]) is None, "a refusal is an answer"
    assert served_nothing([Records(failed), Records([made()])]) is None, "one was served"
    assert served_nothing([]) is None, "no trials: nothing to serve"
