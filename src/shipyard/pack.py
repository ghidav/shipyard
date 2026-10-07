"""Records into token sequences: the chains a harness's calls form by the token-prefix
rule, one sequence per chain with the completions as target spans, the datum convention
the trainer reads, and the groups a batch's rollouts fall into. No graph."""

from __future__ import annotations

from collections.abc import Sequence as Values
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import tinker
import torch
from tinker import TensorData

from shipyard.admit import Verdict, verdicts
from shipyard.proxy.wire import IMAGE_TOKEN, Record

if TYPE_CHECKING:
    from shipyard.rollout import Rollouts


@dataclass(frozen=True)
class Sequence:
    """One chain as the trainer reads it: the last record's prompt and completion, which
    hold every earlier prompt and completion as a prefix; `spans` where each completion
    sits, in chain order; the sampler's logprobs there, or None when a record had none."""

    tokens: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    mu: tuple[float, ...] | None

    @property
    def targets(self) -> tuple[int, ...]:
        """Every position the policy wrote, ascending; each completion exactly once."""
        return tuple(at for start, end in self.spans for at in range(start, end))


@dataclass(frozen=True)
class Member:
    """One rollout of a group: its verdict, its sequences (none when masked), and how
    many tokens the policy wrote across its calls, which a length rule is a rule on."""

    verdict: Verdict
    sequences: tuple[Sequence, ...]
    sampled_tokens: int


@dataclass(frozen=True)
class Group:
    """A task's rollouts, in plan order, stamped with the trainer's update count when
    they were sampled so credit can refuse weights that moved since."""

    task: Path
    members: tuple[Member, ...]
    updates: int | None = None


def packable(record: Record) -> bool:
    """A refused or failed call has no completion; an image's stand-in tokens are not
    tokens, and admission masks that rollout anyway."""
    return (
        record.error is None
        and IMAGE_TOKEN not in record.prompt_token_ids
        and IMAGE_TOKEN not in record.completion_token_ids
    )


def chains(records: Values[Record]) -> list[list[Record]]:
    """The token-prefix rule: a record joins the open chain whose last record's prompt
    plus completion is a prefix of its prompt, the longest such; otherwise it opens one.
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
    """One Sequence per chain; a chain in which the policy wrote nothing is left out."""
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
        out.append(Sequence(tokens, spans, None if mu is None else tuple(mu)))
    return out


def _span(record: Record) -> tuple[int, int]:
    """Where the record's completion sits in a chain it closes: after its own prompt."""
    head = len(record.prompt_token_ids)
    return head, head + len(record.completion_token_ids)


def shifted(tokens: Values[int]) -> tuple[tinker.ModelInput, TensorData]:
    """The cookbook's shift by one, the one place it is spelled: the model reads
    `tokens[:-1]` and predicts `tokens[1:]`, so position `j` of what it scores is about
    `tokens[j + 1]`, and a target at `p` sits at `p - 1`."""
    if len(tokens) < 2:
        raise ValueError("a sequence needs two tokens for the model to read one and predict one")
    return (
        tinker.ModelInput.from_ints(list(tokens[:-1])),
        TensorData.from_torch(torch.tensor(list(tokens[1:]))),
    )


def datum(
    sequence: Sequence, *, logprobs: Values[float], advantages: Values[float]
) -> tinker.Datum:
    """The datum as the cookbook's `trajectory_to_data` builds one (tinker-cookbook,
    rl/data_processing.py, Apache-2.0, reproduced here): `shifted`, with mask, logprobs
    and advantages at each target's index - 1 and zeros elsewhere."""
    model_input, target_tokens = shifted(sequence.tokens)
    positions = sequence.targets
    if len(logprobs) != len(positions) or len(advantages) != len(positions):
        raise ValueError(
            f"{len(positions)} target(s) take {len(positions)} logprobs and advantages; got "
            f"{len(logprobs)} and {len(advantages)}"
        )
    width = model_input.length
    mask, logprob_at, advantage_at = [0.0] * width, [0.0] * width, [0.0] * width
    for at, position in enumerate(positions):
        if position == 0:
            continue  # the first token is read, never predicted: the cookbook drops it too
        mask[position - 1] = 1.0
        logprob_at[position - 1] = float(logprobs[at])
        advantage_at[position - 1] = float(advantages[at])
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": TensorData.from_torch(torch.tensor(logprob_at)),
            "advantages": TensorData.from_torch(torch.tensor(advantage_at)),
            "mask": TensorData.from_torch(torch.tensor(mask)),
        },
    )


def group(rollouts: Rollouts, group_size: int) -> list[Group]:
    """One Group per task in plan order, cut in slices of `group_size` the way the plan
    laid the trials out; a masked member carries no sequences. Refused when the job holds
    no records (nothing served, nothing to train on) or does not divide into groups."""
    if rollouts.records is None:
        raise ValueError(
            f"job {rollouts.job} recorded no model calls: a gradient recipe trains on the "
            "records of a model this run serves"
        )
    if group_size < 1:
        raise ValueError(f"group_size must be at least 1; got {group_size}")
    trials = list(rollouts.trials)
    if len(trials) % group_size:
        raise ValueError(
            f"job {rollouts.job} has {len(trials)} trial(s), which do not divide into groups "
            f"of {group_size}: a batch that was not rolled out at this group size"
        )
    judged = verdicts(rollouts)
    groups: list[Group] = []
    for start in range(0, len(trials), group_size):
        tasks = {rollouts.plan[at][0] for at in range(start, start + group_size)}
        if len(tasks) != 1:
            raise ValueError(
                f"job {rollouts.job}: trials {start}..{start + group_size - 1} belong to "
                f"{len(tasks)} tasks, not one group"
            )
        members: list[Member] = []
        for at in range(start, start + group_size):
            records = list(rollouts.records.get(Path(trials[at]).name) or [])
            verdict = judged[at]
            members.append(
                Member(
                    verdict=verdict,
                    sequences=tuple(sequences(records)) if verdict.mask is None else (),
                    sampled_tokens=sum(
                        len(record.completion_token_ids)
                        for record in records
                        if record.error is None
                    ),
                )
            )
        groups.append(Group(tasks.pop(), tuple(members), rollouts.updates))
    return groups
