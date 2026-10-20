from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from statistics import fmean, pvariance
from typing import TYPE_CHECKING, Any

import torch
from harbor.models.trial.result import TrialResult
from pydantic import ValidationError

from harness_proxy.wire import extends, pictured

if TYPE_CHECKING:
    from harness_proxy.wire import Picture, Record
    from shipyard.jobs.admit import Verdict
    from shipyard.jobs.harnesses import Harness

logger = logging.getLogger(__name__)

#: The file a task's instruction is in.
INSTRUCTION = "instruction.md"
#: The file Harbor writes a trial's result to.
RESULT = "result.json"
#: Where in a Harbor trial the agent's trajectory is kept, in Harbor's ATIF format, and what the
#: tests printed.
TRAJECTORY = ("agent", "trajectory.json")
TEST_OUTPUT = ("verifier", "test-stdout.txt")


@dataclass(eq=False)
class Sequence:
    """One chain of model calls a trial made, as token ids, with an advantage per target token."""

    tokens: tuple[int, ...]
    #: (start, end) of each completion, in chain order.
    spans: tuple[tuple[int, int], ...]
    #: The sampler's logprobs at the targets, as the proxy recorded them; None if it had none.
    sampled_logprobs: tuple[float, ...] | None
    images: tuple[Picture, ...]
    trial: Trial = field(repr=False)
    #: One advantage per target token. `update` trains on the sequences that have one.
    advantages: torch.Tensor | None = None

    @property
    def targets(self) -> tuple[int, ...]:
        """Positions of every completion token, in chain order."""
        return tuple(at for start, end in self.spans for at in range(start, end))


@dataclass(eq=False)
class Trial:
    """One run of a harness at a task: its records, its verdict and the sequences it wrote."""

    job: str
    task: Path
    harness: Harness
    dir: Path
    #: The proxy's records of its model calls, in order; None when the model's provider served
    #: it, since then nothing is recorded.
    records: list[Record] | None = None
    #: Its reward, or why it is masked; set when its job is judged.
    verdict: Verdict | None = None
    #: The chains of model calls it made, as token ids; none when it is masked.
    sequences: list[Sequence] = field(default_factory=list)
    #: The renderer the proxy rendered its calls with, as `[rollout] renderer` names it; None
    #: for the model's own.
    renderer: str | None = None

    @property
    def name(self) -> str:
        return self.dir.name

    @cached_property
    def result(self) -> TrialResult | None:
        """Harbor's `result.json` for the trial, read once its job has ended; None if none."""
        return result_in(self.dir)

    @property
    def measured(self) -> bool:
        """Whether the trial counts: a masked one has no reward and no sequences."""
        return self.verdict is not None and self.verdict.mask is None

    @property
    def reward(self) -> float | None:
        """Its reward; None when it is masked."""
        if self.verdict is None or self.verdict.mask is not None:
            return None
        return float(self.verdict.reward or 0.0)

    @property
    def sampled_tokens(self) -> int:
        """Tokens the policy wrote in its model calls that did not fail."""
        return sum(len(one.completion_token_ids) for one in self.records or () if one.error is None)

    @property
    def instruction(self) -> str:
        """The task's instruction, or "" when it cannot be read."""
        return _text(self.task / INSTRUCTION)

    @property
    def steps(self) -> list[Mapping[str, Any]]:
        """What the agent did, step by step, as Harbor's ATIF trajectory has it; [] if none."""
        try:
            steps = json.loads(_text(self.dir.joinpath(*TRAJECTORY)))["steps"]
        except (ValueError, KeyError, TypeError):
            return []
        if not isinstance(steps, list):
            return []
        return [one for one in steps if isinstance(one, Mapping)]

    @property
    def transcript(self) -> str:
        """What the agent did, step by step, as text; "" when there is no trajectory."""
        return "\n\n".join(_step(one) for one in self.steps)

    @property
    def test_output(self) -> str:
        """What the tests printed, or "" when they printed nothing."""
        return _text(self.dir.joinpath(*TEST_OUTPUT))

    def set_advantage(self, value: float) -> None:
        """Give every target token of every sequence the advantage `value`."""
        for one in self.sequences:
            one.advantages = torch.full((len(one.targets),), float(value))


def result_in(trial: Path) -> TrialResult | None:
    """Return the `TrialResult` Harbor wrote in the trial directory `trial`, or None."""
    path = Path(trial) / RESULT
    try:
        return TrialResult.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    except ValidationError as error:
        logger.warning("%s does not read as Harbor's TrialResult: %s", path, error)
        return None


@dataclass(eq=False)
class Group:
    """One task's trials at one renderer, and one harness unless mixed, in the order they ran."""

    task: Path
    trials: list[Trial]

    @property
    def renderer(self) -> str | None:
        """The renderer its trials' calls were rendered with: a group never mixes them."""
        return self.trials[0].renderer if self.trials else None

    @property
    def harness(self) -> str | None:
        """The harness its trials ran, as `[agent] harness` names it; None when it mixes them."""
        ran = {one.harness.agent for one in self.trials}
        return ran.pop() if len(ran) == 1 else None

    @property
    def measured(self) -> list[Trial]:
        return [one for one in self.trials if one.measured]

    @property
    def rewards(self) -> list[float]:
        """The measured trials' rewards."""
        return [float(one.reward or 0.0) for one in self.measured]

    @property
    def reward_variance(self) -> float:
        """The variance of the rewards; 0 for fewer than two."""
        rewards = self.rewards
        return pvariance(rewards) if len(rewards) > 1 else 0.0

    @property
    def centered_rewards(self) -> list[float]:
        """Each measured trial's reward minus the group's mean."""
        rewards = self.rewards
        mean = fmean(rewards) if rewards else 0.0
        return [one - mean for one in rewards]

    @property
    def scored(self) -> list[Sequence]:
        """Its sequences that have advantages, the ones `update` trains on."""
        return [one for t in self.measured for one in t.sequences if one.advantages is not None]

    @property
    def is_scored(self) -> bool:
        """Whether any of its sequences has advantages to train on."""
        return bool(self.scored)


def merged(groups: Values[Group], *, mix: bool = False) -> list[Group]:
    """Return one group per task, renderer and, unless `mix`, harness, in order of first run."""
    found: dict[tuple[Path, str | None, str | None], list[Trial]] = {}
    for group in groups:
        key = (group.task, group.renderer, None if mix else group.harness)
        found.setdefault(key, []).extend(group.trials)
    return [Group(task, trials) for (task, _, _), trials in found.items()]


def grouped(trials: Values[Trial]) -> list[Group]:
    """Group judged trials by task, and give each measured trial its sequences."""
    found: list[Group] = []
    for trial in trials:
        if trial.measured:
            trial.sequences = _sequences(trial)
        if found and found[-1].task == trial.task:
            found[-1].trials.append(trial)
        else:
            found.append(Group(trial.task, [trial]))
    return found


def chains(records: Values[Record]) -> list[list[Record]]:
    """Link a trial's records into chains by the token-prefix rule."""
    found: list[list[Record]] = []
    for record in records:
        if record.error is not None or not pictured(record):
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


def _sequences(trial: Trial) -> list[Sequence]:
    """Return one sequence per chain in which the policy wrote: its last call's tokens."""
    found: list[Sequence] = []
    for chain in chains(trial.records or ()):
        last = chain[-1]
        spans = tuple(_span(one) for one in chain if one.completion_token_ids)
        if not spans:
            continue
        logprobs = tuple(value for one in chain for value in one.inference_logprobs)
        complete = all(
            len(one.inference_logprobs) == len(one.completion_token_ids) for one in chain
        )
        tokens = (*last.prompt_token_ids, *last.completion_token_ids)
        found.append(Sequence(tokens, spans, logprobs if complete else None, last.images, trial))
    return found


def _span(record: Record) -> tuple[int, int]:
    """Return where a record's completion sits in the chain: after its prompt."""
    head = len(record.prompt_token_ids)
    return head, head + len(record.completion_token_ids)


def _text(path: Path) -> str:
    """Return a file's text, or "" when it cannot be read."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _step(step: Mapping[str, Any]) -> str:
    """Write one step of an ATIF trajectory as text."""
    lines = [f"[{step.get('source')}]", _content(step.get("message"))]
    for call in step.get("tool_calls") or ():
        arguments = json.dumps(call.get("arguments"), ensure_ascii=False)
        lines.append(f"[call] {call.get('function_name')} {arguments}")
    for result in (step.get("observation") or {}).get("results") or ():
        lines.append(f"[result]\n{_content(result.get('content'))}")
    return "\n".join(line for line in lines if line)


def _content(value: Any) -> str:
    """Return an ATIF message or result as text, marking each part that is not text."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts = (one for one in value if isinstance(one, Mapping))
    return "\n".join(one.get("text") or f"[{one.get('type')}]" for one in parts)
