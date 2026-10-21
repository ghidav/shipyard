"""Tests for the sequences `grouped` gives a trial (`shipyard.groups`) and the Tinker datums
made of them (`shipyard.models.clients.tinker`), checked against the cookbook."""

from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets as cookbook_shift,
)

from harness_proxy.wire import IMAGE_TOKEN, Picture, Record, model_input
from shipyard.groups import Sequence, Trial, grouped
from shipyard.jobs.admit import ENV_ERROR, Verdict
from shipyard.jobs.harnesses import harness
from shipyard.models.clients.tinker import datum, shifted
from shipyard.records import Records
from tests.fakes import made

P1, C1, T1, C2, T2, C3 = (1, 2, 3), (10, 11), (20,), (12,), (21, 22), (13, 14)
PNG = base64.b64encode(b"png-bytes").decode()
#: An image that spans three prompt tokens starting at offset 1.
SHOT = Picture(1, 3, "png", PNG)


def _record(prompt: tuple[int, ...], completion: tuple[int, ...], seq: int, **named) -> Record:
    found = made(prompt, completion, seq=seq, **named)
    scored = tuple(-float(token) / 10 for token in completion)
    return replace(found, inference_logprobs=scored if found.error is None else ())


def _trial(*records: Record) -> Trial:
    """A graded trial holding `records`."""
    graded = Verdict(1.0, None, None)
    return Trial("j", Path("t"), harness("pi"), Path("t"), Records(records), graded)


def _packed(records: list[Record]) -> list[Sequence]:
    """The sequences a graded trial's records pack into, as `grouped` gives them."""
    trial = _trial(*records)
    grouped([trial])
    return trial.sequences


def _tool_loop() -> list[Record]:
    """Return three tool-loop records, each prompt extending the previous prompt and completion."""
    first = _record(P1, C1, 1)
    second = _record((*P1, *C1, *T1), C2, 2)
    third = _record((*P1, *C1, *T1, *C2, *T2), C3, 3)
    return [first, second, third]


def _written(sequence: Sequence) -> list[int]:
    return [sequence.tokens[at] for at in sequence.targets]


# ------------------------------------------------------------- what a trial keeps


def test_grouped_gives_a_measured_trial_its_chains_and_any_other_trial_none() -> None:
    graded = _trial(*_tool_loop())
    masked = replace(_trial(*_tool_loop()), verdict=Verdict(None, ENV_ERROR, None))
    provided = Trial("j", Path("t"), harness("pi"), Path("t"), verdict=Verdict(1.0, None, None))
    grouped([graded, masked, provided])
    [found] = graded.sequences
    assert found.trial is graded and found.targets == (3, 4, 6, 9, 10)
    assert _written(found) == [*C1, *C2, *C3]
    assert masked.sequences == [], "a masked trial keeps no sequences"
    assert provided.sequences == [], "a provider's trial: the proxy recorded nothing to train on"


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
    [found] = _packed([seen])
    assert found.sampled_logprobs is not None
    ours = datum(found, logprobs=found.sampled_logprobs, advantages=[0.5, 0.5])
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
    with pytest.raises(ValueError, match="last chunk must be a text chunk"):
        shifted((1, IMAGE_TOKEN), (Picture(1, 1, "png", PNG),))
    assert model_input((1, 2, 3)).to_ints() == [1, 2, 3]


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
    [found] = _packed(records)
    assert found.sampled_logprobs is not None
    ours = datum(found, logprobs=found.sampled_logprobs, advantages=[0.75] * len(found.targets))
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
    found = _packed(records)
    ours = [
        datum(one, logprobs=one.sampled_logprobs or (), advantages=[-1.0] * len(one.targets))
        for one in found
    ]
    theirs = _cookbook(records, -1.0)
    assert len(ours) == len(theirs) == 2
    assert [_fields(one) for one in ours] == [_fields(one) for one in theirs]


def test_a_datum_refuses_misaligned_inputs_and_a_sequence_too_short_to_shift() -> None:
    [found] = _packed(_tool_loop())
    with pytest.raises(ValueError, match="target"):
        datum(found, logprobs=[0.0], advantages=[1.0] * 5)
    with pytest.raises(ValueError, match="at least 2 tokens"):
        datum(Sequence((1,), ((0, 1),), None, (), _trial()), logprobs=[0.0], advantages=[1.0])
    # A completion starting at position 0 has no prediction to credit: the cookbook drops it.
    headless = Sequence((5, 6), ((0, 2),), (-0.1, -0.2), (), _trial())
    made_one = datum(headless, logprobs=(-0.1, -0.2), advantages=[1.0, 1.0])
    assert made_one.loss_fn_inputs["mask"].tolist() == [1] and made_one.model_input.to_ints() == [5]
