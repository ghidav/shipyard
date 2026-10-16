"""Tests for packing proxy records into sequences and datums, checked against the cookbook."""

from __future__ import annotations

import base64
from dataclasses import replace
from typing import Any

import pytest
import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets as cookbook_shift,
)

from shipyard.proxy.wire import IMAGE_TOKEN, Picture, Record, model_input
from shipyard.training.pack import chains, datum, packable, sequences, shifted
from shipyard.types import Sequence
from tests.records import made

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


def _written(sequence: Sequence) -> list[int]:
    return [sequence.tokens[at] for at in sequence.targets]


# ---------------------------------------------------------------- the prefix rule


def test_a_tool_loop_packs_into_one_sequence_with_three_target_spans() -> None:
    [found] = sequences(_tool_loop())
    assert found.tokens == (*P1, *C1, *T1, *C2, *T2, *C3)
    assert found.spans == ((3, 5), (6, 7), (9, 11))
    assert found.targets == (3, 4, 6, 9, 10)
    assert _written(found) == [*C1, *C2, *C3]
    assert found.mu == (-1.0, -1.1, -1.2, -1.3, -1.4)


def test_a_prompt_that_does_not_extend_opens_a_second_sequence() -> None:
    """The fourth call re-sends the history with thinking stripped, so its prompt shares a
    prefix with the chain and then diverges. It starts a new sequence, and each completion
    is still a target exactly once."""
    stripped = _record((*P1, 10, 99, *T1, *C2, *T2, *C3, 23), (15,), 4)
    found = sequences([*_tool_loop(), stripped])
    assert len(found) == 2
    assert found[1].tokens == (*stripped.prompt_token_ids, 15)
    assert found[1].spans == ((len(stripped.prompt_token_ids), len(stripped.prompt_token_ids) + 1),)
    assert sorted(token for one in found for token in _written(one)) == sorted([*C1, *C2, *C3, 15])
    assert sum(len(one.targets) for one in found) == 6


def test_a_side_question_is_filed_under_its_own_chain_and_the_longest_match_wins() -> None:
    main1 = _record((1, 2), (3,), 1)
    main2 = _record((1, 2, 3, 4), (5,), 2)
    side = _record((7, 8), (9,), 3)
    main3 = _record((1, 2, 3, 4, 5, 6), (10,), 4)
    side2 = _record((7, 8, 9, 11), (12,), 5)
    assert chains([main1, main2, side, main3, side2]) == [[main1, main2, main3], [side, side2]]
    first = _record((1, 2), (3,), 1)
    resampled = _record((1, 2), (4,), 2)
    onward = _record((1, 2, 4, 5), (6,), 3)
    summary = _record((20, 21), (22,), 4)
    assert chains([first, resampled, onward, summary]) == [[first], [resampled, onward], [summary]]
    assert chains([]) == [] and sequences([]) == []
    shorter = _record((1, 2, 3), (4, 5), 1), _record((1, 2, 3, 4), (6,), 2)
    assert chains(list(shorter)) == [[shorter[0]], [shorter[1]]]


def test_error_records_and_images_without_their_picture_never_pack() -> None:
    first, second, third = _tool_loop()
    refused = _record((*P1, *C1, *T1), (), 9, error="context")
    unkept = _record((IMAGE_TOKEN, IMAGE_TOKEN, 5), (6,), 10)
    assert not packable(refused) and not packable(unkept) and packable(first)
    found = sequences([first, refused, second, unkept, third])
    assert len(found) == 1 and found[0].spans == ((3, 5), (6, 7), (9, 11))
    assert sequences([refused, unkept]) == []


def test_a_kept_image_packs_and_the_sequence_carries_the_picture() -> None:
    seen = _record((1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 5), (6, 7), 1, images=(SHOT,))
    assert packable(seen)
    again = _record((*seen.prompt_token_ids, 6, 7, 8), (9,), 2, images=(SHOT,))
    [found] = sequences([seen, again])
    assert found.tokens == (1, -1, -1, -1, 5, 6, 7, 8, 9) and found.images == (SHOT,)
    assert found.spans == ((5, 7), (8, 9))
    shorter = replace(seen, images=(Picture(1, 2, "png", PNG),))
    assert not packable(shorter), "a picture of another length is not this image"


def _chunks(found: tinker.ModelInput) -> list[tuple[str, Any]]:
    return [
        (
            type(chunk).__name__,
            list(chunk.tokens)
            if hasattr(chunk, "tokens")
            else (chunk.length, chunk.data, chunk.format),
        )
        for chunk in found.chunks
    ]


def test_a_pictured_datum_is_the_cookbooks_chunks_and_shift() -> None:
    seen = _record((1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 5), (6, 7), 1, images=(SHOT,))
    [found] = sequences([seen])
    ours = datum(found, logprobs=found.mu, advantages=[0.5, 0.5])
    whole = [
        tinker.types.EncodedTextChunk(tokens=[1]),
        tinker.types.ImageChunk(data=b"png-bytes", format="png", expected_tokens=3),
        tinker.types.EncodedTextChunk(tokens=[5, 6, 7]),
    ]
    theirs, targets = cookbook_shift(whole)
    assert _chunks(ours.model_input) == _chunks(theirs)
    assert _chunks(ours.model_input) == [
        ("EncodedTextChunk", [1]),
        ("ImageChunk", (3, b"png-bytes", "png")),
        ("EncodedTextChunk", [5, 6]),
    ]
    assert ours.model_input.length == 6
    assert ours.loss_fn_inputs["target_tokens"].tolist() == targets == [0, 0, 0, 5, 6, 7]
    assert ours.loss_fn_inputs["mask"].tolist() == [0, 0, 0, 0, 1, 1]
    assert ours.loss_fn_inputs["logprobs"].tolist() == pytest.approx([0, 0, 0, 0, -0.6, -0.7])
    assert ours.loss_fn_inputs["advantages"].tolist() == pytest.approx([0, 0, 0, 0, 0.5, 0.5])


def test_model_input_refuses_an_image_without_its_picture_or_out_of_place() -> None:
    with pytest.raises(ValueError, match="picture"):
        model_input((1, IMAGE_TOKEN, 5))
    with pytest.raises(ValueError, match="image tokens"):
        model_input((1, IMAGE_TOKEN, 5), (Picture(0, 1, "png", PNG),))
    with pytest.raises(ValueError, match="end on an image"):
        shifted((1, IMAGE_TOKEN), (Picture(1, 1, "png", PNG),))
    assert model_input((1, 2, 3)).to_ints() == [1, 2, 3]


def test_a_record_without_logprobs_leaves_the_sequence_with_no_mu() -> None:
    first, second, third = _tool_loop()
    bare = replace(second, inference_logprobs=())
    [found] = sequences([first, bare, third])
    assert found.mu is None and found.spans == ((3, 5), (6, 7), (9, 11))
    silent = replace(first, completion_token_ids=(), inference_logprobs=())
    assert sequences([silent]) == []


# ------------------------------------------------------------ the datum convention


def _cookbook(records: list[Record], advantage: float) -> list[tinker.Datum]:
    """Pack `records` the way the cookbook does, with one transition per record."""
    transitions = [
        Transition(
            ob=tinker.ModelInput.from_ints(list(record.prompt_token_ids)),
            ac=TokensWithLogprobs(
                tokens=list(record.completion_token_ids),
                maybe_logprobs=list(record.inference_logprobs),
            ),
            reward=0.0,
            episode_done=record is records[-1],
            metrics={},
        )
        for record in records
    ]
    return trajectory_to_data(
        Trajectory(transitions=transitions, final_ob=tinker.ModelInput.from_ints([])), advantage
    )


def _fields(found: tinker.Datum) -> dict[str, list]:
    return {
        "input": found.model_input.to_ints(),
        **{key: value.tolist() for key, value in found.loss_fn_inputs.items()},
    }


def test_the_datum_matches_the_cookbooks_shift_by_one_token_for_token() -> None:
    records = _tool_loop()
    [found] = sequences(records)
    ours = datum(found, logprobs=found.mu, advantages=[0.75] * len(found.targets))
    [theirs] = _cookbook(records, 0.75)
    assert set(ours.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages", "mask"}
    assert _fields(ours) == _fields(theirs)
    assert ours.model_input.to_ints() == list(found.tokens[:-1])
    assert ours.loss_fn_inputs["target_tokens"].tolist() == list(found.tokens[1:])
    assert ours.loss_fn_inputs["mask"].tolist() == [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    assert ours.loss_fn_inputs["logprobs"].tolist() == pytest.approx(
        [0, 0, -1.0, -1.1, 0, -1.2, 0, 0, -1.3, -1.4]
    )
    assert ours.loss_fn_inputs["advantages"].tolist() == pytest.approx(
        [0, 0, 0.75, 0.75, 0, 0.75, 0, 0, 0.75, 0.75]
    )
    for key in ("target_tokens", "logprobs", "advantages", "mask"):
        assert ours.loss_fn_inputs[key].dtype == theirs.loss_fn_inputs[key].dtype, key


def test_a_broken_chain_gives_the_cookbooks_two_datums() -> None:
    first, second, _ = _tool_loop()
    apart = _record((30, 31), (32, 33), 3)
    records = [first, second, apart]
    found = sequences(records)
    ours = [datum(one, logprobs=one.mu, advantages=[-1.0] * len(one.targets)) for one in found]
    theirs = _cookbook(records, -1.0)
    assert len(ours) == len(theirs) == 2
    assert [_fields(one) for one in ours] == [_fields(one) for one in theirs]


def test_a_datum_refuses_misaligned_inputs_and_a_sequence_too_short_to_shift() -> None:
    [found] = sequences(_tool_loop())
    with pytest.raises(ValueError, match="target"):
        datum(found, logprobs=[0.0], advantages=[1.0] * 5)
    with pytest.raises(ValueError, match="two tokens"):
        datum(Sequence((1,), ((0, 1),), None), logprobs=[0.0], advantages=[1.0])
    # A completion starting at position 0 has no prediction to credit: the cookbook drops it.
    headless = Sequence((5, 6), ((0, 2),), (-0.1, -0.2))
    made_one = datum(headless, logprobs=headless.mu, advantages=[1.0, 1.0])
    assert made_one.loss_fn_inputs["mask"].tolist() == [1] and made_one.model_input.to_ints() == [5]
