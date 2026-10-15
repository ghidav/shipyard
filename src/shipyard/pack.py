"""Records into token sequences. A harness's calls form chains by the token-prefix rule,
and each chain becomes one sequence with the completions as target spans. This module also
holds the datum convention the trainer reads and the groups a batch's rollouts fall into.
It builds no graph."""

from __future__ import annotations

from collections.abc import Sequence as Values
from typing import TYPE_CHECKING

import tinker
import torch
from tinker import TensorData

from shipyard import types
from shipyard.admit import verdicts
from shipyard.proxy.wire import IMAGE_TOKEN, Picture, Record, model_input, pictured
from shipyard.types import Sequence

if TYPE_CHECKING:
    from collections.abc import Sequence as Values

    from shipyard.rollout import Rollouts
    from shipyard.types import Agent, Trial

__all__ = [
    "Sequence",
    "chains",
    "datum",
    "groups",
    "model_input",
    "packable",
    "sequences",
    "shifted",
]


def packable(record: Record) -> bool:
    """A refused or failed call has no completion. An image's stand-in tokens pack only
    with the picture behind them; a record that kept none is masked by admission anyway."""
    return record.error is None and pictured(record)


def chains(records: Values[Record]) -> list[list[Record]]:
    """The token-prefix rule. A record joins the open chain whose last record's prompt plus
    completion is a prefix of its prompt, the longest such, and otherwise opens a chain.
    Only the order within one rollout moves; a call continues one that had finished."""
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
    """One Sequence per chain. A chain in which the policy wrote nothing is left out."""
    out: list[Sequence] = []
    for chain in chains(records):
        last = chain[-1]
        tokens = (*last.prompt_token_ids, *last.completion_token_ids)
        spans = tuple(_span(record) for record in chain if record.completion_token_ids)
        if not spans:
            continue
        mu: list[float] | None = []
        for record in chain:
            if len(record.inference_logprobs) != len(record.completion_token_ids):
                mu = None
                break
            mu.extend(record.inference_logprobs)
        out.append(Sequence(tokens, spans, None if mu is None else tuple(mu), last.images))
    return out


def _span(record: Record) -> tuple[int, int]:
    """Where the record's completion sits in a chain it closes: after its own prompt."""
    head = len(record.prompt_token_ids)
    return head, head + len(record.completion_token_ids)


def shifted(
    tokens: Values[int], images: Values[Picture] = ()
) -> tuple[tinker.ModelInput, TensorData]:
    """The cookbook's shift by one, defined here. The model reads `tokens[:-1]` and predicts
    `tokens[1:]`, so position `j` of what it scores is about `tokens[j + 1]`, and a target
    at `p` sits at `p - 1`. An image's positions are never targets: the cookbook writes 0
    there, and so does this."""
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
    """The datum as the cookbook's `trajectory_to_data` builds it (tinker-cookbook,
    rl/data_processing.py, Apache-2.0, reproduced here): the `shifted` tokens, with mask,
    logprobs and advantages at each target's index - 1 and zeros elsewhere."""
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


def groups(trials: Values[Trial], rolled: Rollouts, agent: Agent) -> list[types.Group]:
    """One group per task and harness in plan order, each trial a member."""
    judged = verdicts(rolled)
    records = rolled.records or {}
    out: list[types.Group] = []
    for trial, verdict in zip(trials, judged, strict=True):
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
