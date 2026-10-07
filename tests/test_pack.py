"""Records into sequences: the token-prefix rule, the target spans, the datum convention
checked token for token against the cookbook's own, and the groups a job falls into."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition

from shipyard.pack import Sequence, chains, datum, group, packable, sequences
from shipyard.proxy.wire import IMAGE_TOKEN, Record
from shipyard.rollout import Rollouts
from tests.records import made
from tests.trials import write_result

P1, C1, T1, C2, T2, C3 = (1, 2, 3), (10, 11), (20,), (12,), (21, 22), (13, 14)


def _record(prompt: tuple[int, ...], completion: tuple[int, ...], seq: int, **named) -> Record:
    found = made(prompt, completion, seq=seq, **named)
    scored = tuple(-float(token) / 10 for token in completion)
    return replace(found, inference_logprobs=scored if found.error is None else ())


def _tool_loop() -> list[Record]:
    """Three calls whose prompts each extend the last prompt and completion."""
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
    """The fourth call re-sends the history with the thinking stripped, so its prompt
    begins like the chain and then differs: a new sequence, every completion once."""
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


def test_error_and_image_records_never_pack() -> None:
    first, second, third = _tool_loop()
    refused = _record((*P1, *C1, *T1), (), 9, error="context")
    pictured = _record((IMAGE_TOKEN, IMAGE_TOKEN, 5), (6,), 10)
    assert not packable(refused) and not packable(pictured) and packable(first)
    found = sequences([first, refused, second, pictured, third])
    assert len(found) == 1 and found[0].spans == ((3, 5), (6, 7), (9, 11))
    assert sequences([refused, pictured]) == []


def test_a_record_without_logprobs_leaves_the_sequence_with_no_mu() -> None:
    first, second, third = _tool_loop()
    bare = replace(second, inference_logprobs=())
    [found] = sequences([first, bare, third])
    assert found.mu is None and found.spans == ((3, 5), (6, 7), (9, 11))
    silent = replace(first, completion_token_ids=(), inference_logprobs=())
    assert sequences([silent]) == []


# ------------------------------------------------------------ the datum convention


def _cookbook(records: list[Record], advantage: float) -> list[tinker.Datum]:
    """The same records as the cookbook packs them: one transition per record."""
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


# ----------------------------------------------------------------------- groups


def _job(tmp_path: Path, records: dict[str, list[Record] | None], outcomes: dict) -> Rollouts:
    """Two tasks, two rollouts each, in plan order, with a result and records per trial."""
    trials, plan = [], []
    for task in ("a", "b"):
        for index in range(2):
            name = f"{task}__{index}"
            outcome = outcomes.get(name)
            trial = tmp_path / "jobs" / "j" / name
            if outcome is not None:
                write_result(trial, **outcome)
            else:
                trial.mkdir(parents=True)
            trials.append(trial)
            plan.append((tmp_path / "tasks" / task, index))
    taken = {name: list(found) for name, found in records.items() if found is not None}
    return Rollouts(job="j", trials=trials, plan=plan, records=taken, updates=3)


def test_group_cuts_the_plan_into_tasks_with_verdicts_and_sequences(tmp_path: Path) -> None:
    first, second, third = _tool_loop()
    failed = _record((1, 2), (), 1, error="sampler: Boom")
    rolled = _job(
        tmp_path,
        {"a__0": [first, second, third], "a__1": [], "b__0": [first, failed], "b__1": [first]},
        {
            "a__0": {"reward": 1.0},
            "a__1": {"reward": 0.0},
            "b__0": {"reward": 1.0},
            "b__1": {"reward": 0.0, "exception": "AgentTimeoutError"},
        },
    )
    found = group(rolled, 2)
    assert [one.task.name for one in found] == ["a", "b"]
    assert all(one.updates == 3 for one in found)
    alpha, beta = found
    assert [member.verdict.reward for member in alpha.members] == [1.0, None]
    assert [member.verdict.mask for member in alpha.members] == [None, "env_error"]
    assert len(alpha.members[0].sequences) == 1 and alpha.members[0].sampled_tokens == 5
    assert alpha.members[1].sequences == () and alpha.members[1].sampled_tokens == 0
    assert [member.verdict.mask for member in beta.members] == ["api_error", "timeout"]
    assert all(member.sequences == () for member in beta.members)
    assert beta.members[0].sampled_tokens == 2


def test_group_refuses_a_job_with_no_records_or_a_size_that_does_not_divide(
    tmp_path: Path,
) -> None:
    rolled = _job(tmp_path, {}, {"a__0": {"reward": 1.0}})
    with pytest.raises(ValueError, match="do not divide"):
        group(rolled, 3)
    with pytest.raises(ValueError, match="at least 1"):
        group(rolled, 0)
    with pytest.raises(ValueError, match="recorded no model calls"):
        group(replace(rolled, records=None), 2)
    crossed = replace(rolled, plan=[(tmp_path / "tasks" / "a", 0)] * 3 + [(tmp_path / "x", 0)])
    with pytest.raises(ValueError, match="2 tasks, not one group"):
        group(crossed, 2)
