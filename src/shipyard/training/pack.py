"""Packing of proxy records into token sequences, Tinker datums and groups.

A harness's calls are linked into chains by the token-prefix rule alone, and each chain
becomes one sequence whose completions are its target spans."""

from __future__ import annotations

from collections.abc import Sequence as Values
from typing import TYPE_CHECKING

import tinker
import torch
from tinker import TensorData

from shipyard import types
from shipyard.proxy.wire import IMAGE_TOKEN, Picture, Record, model_input, pictured
from shipyard.types import Sequence

if TYPE_CHECKING:
    from shipyard.types import Agent, Job

__all__ = [
    "chains",
    "datum",
    "groups",
    "packable",
    "sequences",
    "shifted",
]


def packable(record: Record) -> bool:
    """Return whether a record can be packed into a sequence.

    A refused or failed call has no completion. A record with image placeholder tokens packs
    only if it kept the image; one that did not is masked by admission anyway."""
    return record.error is None and pictured(record)


def chains(records: Values[Record]) -> list[list[Record]]:
    """Link one rollout's packable records into chains by the token-prefix rule.

    A record joins the chain whose last record's prompt plus completion is a prefix of its
    prompt, choosing the longest such chain; otherwise it starts a new chain. Records are
    taken in order, since a call can only continue one that had already finished."""
    found: list[list[Record]] = []
    for record in records:
        if not packable(record):
            continue
        prompt = record.prompt_token_ids
        home: list[Record] | None = None
        longest = -1
        for chain in found:
            last = chain[-1]
            head = len(last.prompt_token_ids)
            end = head + len(last.completion_token_ids)
            if end <= longest or len(prompt) < end:
                continue
            if (
                prompt[head:end] == last.completion_token_ids
                and prompt[:head] == last.prompt_token_ids
            ):
                home, longest = chain, end
        if home is None:
            found.append([record])
        else:
            home.append(record)
    return found


def sequences(records: Values[Record]) -> list[Sequence]:
    """Return one `Sequence` per chain, skipping chains in which the policy generated nothing.

    `sampled_logprobs` is None unless every record carries a logprob for each completion
    token."""
    out: list[Sequence] = []
    for chain in chains(records):
        last = chain[-1]
        tokens = (*last.prompt_token_ids, *last.completion_token_ids)
        spans = tuple(_span(record) for record in chain if record.completion_token_ids)
        if not spans:
            continue
        sampled: list[float] | None = []
        for record in chain:
            if len(record.inference_logprobs) != len(record.completion_token_ids):
                sampled = None
                break
            sampled.extend(record.inference_logprobs)
        out.append(
            Sequence(tokens, spans, None if sampled is None else tuple(sampled), last.images)
        )
    return out


def _span(record: Record) -> tuple[int, int]:
    head = len(record.prompt_token_ids)
    return head, head + len(record.completion_token_ids)


def shifted(
    tokens: Values[int], images: Values[Picture] = ()
) -> tuple[tinker.ModelInput, TensorData]:
    """Shift a sequence by one into the model input and target tokens, as the cookbook does.

    The model reads `tokens[:-1]` and predicts `tokens[1:]`, so output position `j` scores
    `tokens[j + 1]`, and a target at `p` sits at `p - 1`. Image positions are never targets:
    like the cookbook, their target token is 0. Raise `ValueError` for fewer than two tokens
    or a sequence that ends on an image."""
    if len(tokens) < 2:
        raise ValueError("a sequence needs two tokens for the model to read one and predict one")
    if tokens[-1] == IMAGE_TOKEN:
        raise ValueError("a sequence cannot end on an image: nothing is predicted after it")
    targets = [0 if token == IMAGE_TOKEN else token for token in tokens[1:]]
    return (
        model_input(tokens[:-1], images),
        TensorData.from_torch(torch.tensor(targets)),
    )


def datum(
    sequence: Sequence, *, logprobs: Values[float], advantages: Values[float]
) -> tinker.Datum:
    """Build a Tinker datum from a sequence, as the cookbook's `trajectory_to_data` does.

    Reproduced here from tinker-cookbook, rl/data_processing.py (Apache-2.0). The input and
    targets are the `shifted` tokens. `mask`, `logprobs` and `advantages` are set at each
    target's index - 1 and are zero elsewhere."""
    read, target_tokens = shifted(sequence.tokens, sequence.images)
    positions = sequence.targets
    if len(logprobs) != len(positions) or len(advantages) != len(positions):
        raise ValueError(
            f"{len(positions)} target(s) take {len(positions)} logprobs and advantages; got "
            f"{len(logprobs)} and {len(advantages)}"
        )
    width = read.length
    mask, logprob_at, advantage_at = [0.0] * width, [0.0] * width, [0.0] * width
    for at, position in enumerate(positions):
        if position == 0:
            continue  # the first token is read but not predicted; the cookbook drops it too
        mask[position - 1] = 1.0
        logprob_at[position - 1] = float(logprobs[at])
        advantage_at[position - 1] = float(advantages[at])
    return tinker.Datum(
        model_input=read,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": TensorData.from_torch(torch.tensor(logprob_at)),
            "advantages": TensorData.from_torch(torch.tensor(advantage_at)),
            "mask": TensorData.from_torch(torch.tensor(mask)),
        },
    )


def groups(job: Job, agent: Agent) -> list[types.Group]:
    """Group a judged job's trials by task and harness, in plan order.

    Each trial becomes a member with its verdict and, unless it is masked, its sequences."""
    records = job.records or {}
    out: list[types.Group] = []
    for trial, verdict in zip(job.trials, job.verdicts, strict=True):
        taken = list(records.get(trial.name) or [])
        member = types.Member(
            trial,
            verdict,
            tuple(sequences(taken)) if verdict.mask is None else (),
            sum(len(one.completion_token_ids) for one in taken if one.error is None),
        )
        if out and out[-1].task == trial.task and out[-1].harness == trial.harness:
            out[-1].members.append(member)
        else:
            out.append(types.Group(trial.task, agent, trial.harness, [member]))
    return out
