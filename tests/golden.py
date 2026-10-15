"""Test the gradient path on fixed inputs."""

from __future__ import annotations

import base64
import os
import shutil
from collections.abc import Callable, Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
import tinker

from shipyard import record, types
from shipyard.admit import BUDGET, CONTEXT, Verdict, verdicts
from shipyard.config import load
from shipyard.credit import Batch, credit
from shipyard.gradient import Preset, row
from shipyard.modules import Candidate, seed
from shipyard.pack import Group, Member, Sequence, group
from shipyard.proxy.wire import IMAGE_TOKEN, Picture, Record
from shipyard.recipes import cispo, dapo, dr_grpo, fst
from shipyard.recipes.fst import merged
from shipyard.rollout import Rollouts
from shipyard.run import Run
from shipyard.trainer import Trainer
from shipyard.training.client import TinkerClient
from shipyard.training.trainer import CISPOTrainer, DAPOTrainer, DrGRPOTrainer
from shipyard.training.trainer import Trainer as Procedure
from shipyard.types import Agent, Connection, Endpoint, Policy, Records, Trial, harness
from tests.records import FakeProxy, made
from tests.trainers import FakeTrainingClient, chunk_ids
from tests.trials import FIXTURES, write_result

#: Where the frozen outputs live, one file per case.
GOLDEN = Path(__file__).parent / "golden"
BLUEPRINTS = Path(__file__).parent / "blueprints"
VALID = Path(__file__).parent / "modules" / "valid"
#: Set in the environment to rewrite every file from the current code.
UPDATE = "GOLDEN_UPDATE"
#: Two numbers agree within this. Datums hold float32, and a sum taken in another order
#: moves the last bits; a wrong scale, sign or token moves far more.
TOLERANCE = 1e-6

PROMPT = (1, 2, 3)
PNG = base64.b64encode(b"png-bytes").decode()


# ------------------------------------------------------------------- the presets


def preset_of(kind: str, **recipe: Any) -> Preset:
    """The fixture blueprint's preset, with these `[recipe]` keys set over its own."""
    loaded = load(BLUEPRINTS / kind).recipe.model_copy(update=recipe)
    module = {"dapo": dapo, "dr-grpo": dr_grpo, "cispo": cispo, "fst": fst}[kind]
    return module.preset(loaded)


#: The recipe classes by `[recipe] kind`.
TRAINERS: dict[str, type[Procedure]] = {
    "dapo": DAPOTrainer,
    "dr-grpo": DrGRPOTrainer,
    "cispo": CISPOTrainer,
}


# ------------------------------------------------------------------- the fakes


class PositionalAnchor:
    """Scores each position minus a tenth of its token."""

    def __init__(self) -> None:
        self.asked: list[list[int]] = []

    async def compute_logprobs_async(self, prompt: Any) -> list[float | None]:
        ids = chunk_ids(prompt)
        self.asked.append(ids)
        return [None] + [-0.1 * abs(token) if token != IMAGE_TOKEN else -0.25 for token in ids[1:]]


class ScriptedTrials:
    """Each task's rollouts take its outcomes in order."""

    def __init__(self, outcomes: Mapping[str, list[dict[str, Any] | None]]) -> None:
        self.outcomes = {task: list(found) for task, found in outcomes.items()}
        self.names: list[str] = []

    async def __call__(self, config: Any) -> None:
        task = Path(str(config.task.path)).name
        self.names.append(config.trial_name)
        trial = Path(config.trials_dir) / config.trial_name
        trial.mkdir(parents=True, exist_ok=True)
        outcome = self.outcomes[task].pop(0)
        if outcome is not None:
            write_result(trial, **outcome)


@dataclass
class ScriptedProxy(FakeProxy):
    queues: dict[str, list[list[Record]]] = field(default_factory=dict)

    async def records(self, trial: str) -> list[Record]:
        self.fetched.append(trial)
        return list(self.queues[trial.split("__")[0]].pop(0))


# ------------------------------------------------------------- building groups


def sequence(
    prompt: tuple[int, ...] = PROMPT,
    *actions: tuple[int, ...],
    tools: tuple[tuple[int, ...], ...] = (),
    scored: bool = True,
    images: tuple[Picture, ...] = (),
) -> Sequence:
    """mu is minus a tenth of each action token."""
    actions = actions or ((10, 11),)
    tokens: list[int] = list(prompt)
    spans: list[tuple[int, int]] = []
    mu: list[float] = []
    for at, action in enumerate(actions):
        if at > 0 and at - 1 < len(tools):
            tokens.extend(tools[at - 1])
        spans.append((len(tokens), len(tokens) + len(action)))
        tokens.extend(action)
        mu.extend(-0.1 * token for token in action)
    return Sequence(tuple(tokens), tuple(spans), tuple(mu) if scored else None, images)


def member(
    reward: float | None,
    *,
    mask: str | None = None,
    wrote: int | None = None,
    sequences: Values[Sequence] | None = None,
    ended: str | None = None,
) -> Member:
    found = () if mask else tuple(sequences if sequences is not None else (sequence(),))
    if wrote is None:
        wrote = sum(len(one.targets) for one in found)
    return Member(Verdict(reward, mask, ended), found, wrote)


def grouped(*members: Member, task: str = "t", updates: int | None = 0) -> Group:
    return Group(Path(task), tuple(members), updates)


def _varied(reward: float, *tokens: int) -> Member:
    return member(reward, sequences=[sequence(PROMPT, tuple(tokens))])


# ------------------------------------------------------------------- the cases


@dataclass(frozen=True)
class Credited:
    groups: Callable[[], list[Group]]
    kind: str
    #: `[recipe]` keys set over the fixture blueprint's.
    recipe: Mapping[str, Any] = field(default_factory=dict)
    #: Preset fields the trainer cannot express; such a case replays on the old path alone.
    preset_only: Mapping[str, Any] = field(default_factory=dict)
    limit: int | None = None
    budget: int | None = None
    anchored: bool = False
    updates: int = 0

    @property
    def preset(self) -> Preset:
        return replace(preset_of(self.kind, **self.recipe), **self.preset_only)


@dataclass(frozen=True)
class Jobbed:
    """A job through `Run.sample`; `texts` above one is an fst step."""

    kind: str
    group_size: int
    tasks: tuple[str, ...]
    outcomes: Mapping[str, list[dict[str, Any] | None]]
    records: Mapping[str, list[list[Record]]]
    preset: Preset | None = None
    budget: int | None = None
    texts: int = 1
    anchored: bool = False


def _mixed() -> list[Group]:
    """Spread, flat, spread with a mask, spread."""
    return [
        grouped(
            _varied(1.0, 10, 11, 12),
            _varied(0.0, 13),
            _varied(1.0, 14, 15),
            _varied(0.0, 16, 17, 18, 19),
            task="a",
        ),
        grouped(_varied(1.0, 20), _varied(1.0, 21), _varied(1.0, 22), _varied(1.0, 23), task="b"),
        grouped(
            _varied(1.0, 30, 31),
            _varied(0.0, 32, 33),
            member(None, mask="grading_error", wrote=7),
            _varied(0.0, 34),
            task="c",
        ),
        grouped(
            _varied(0.0, 40), _varied(1.0, 41, 42), _varied(0.0, 43), _varied(1.0, 44), task="d"
        ),
    ]


def _overlong() -> list[Group]:
    """Under a budget of 100: docked, flat before the term, inside the buffer."""
    return [
        grouped(
            member(1.0, wrote=50, sequences=[sequence(PROMPT, (10, 11))]),
            member(1.0, wrote=90, sequences=[sequence(PROMPT, (12, 13))]),
            member(0.0, wrote=10, sequences=[sequence(PROMPT, (14,))]),
            member(0.0, wrote=100, sequences=[sequence(PROMPT, (15, 16, 17))]),
            task="a",
        ),
        grouped(
            member(0.0, wrote=100, sequences=[sequence(PROMPT, (20, 21))]),
            member(0.0, wrote=10, sequences=[sequence(PROMPT, (22,))]),
            task="b",
        ),
        grouped(
            member(1.0, wrote=70, sequences=[sequence(PROMPT, (30, 31))]),
            member(0.0, wrote=79, sequences=[sequence(PROMPT, (32, 33))]),
            member(1.0, wrote=81, sequences=[sequence(PROMPT, (34,))]),
            task="c",
        ),
    ]


def _chained() -> list[Group]:
    """Several chains and spans."""
    two_chains = member(1.0, sequences=[sequence(PROMPT, (10, 11)), sequence((4, 5), (12, 13, 14))])
    three_spans = member(
        0.0, sequences=[sequence(PROMPT, (20,), (21, 22), (23,), tools=((50, 51), (52,)))]
    )
    return [
        grouped(two_chains, three_spans, _varied(1.0, 30), _varied(0.0, 31, 32), task="a"),
        grouped(_varied(1.0, 40, 41), _varied(0.0, 42), task="b"),
    ]


def _partial() -> list[Group]:
    return [
        grouped(
            _varied(0.25, 10), _varied(0.5, 11, 12), _varied(0.75, 13), _varied(1.0, 14), task="a"
        ),
        grouped(_varied(0.5, 20), _varied(0.5, 21), _varied(1.0, 22), task="b"),
    ]


def _masked() -> list[Group]:
    """All masked; one measured; a pair around masked members."""
    return [
        grouped(
            member(None, mask="env_error", wrote=0, ended=None),
            member(None, mask="timeout", wrote=12, ended="AgentTimeoutError"),
            task="a",
        ),
        grouped(_varied(1.0, 20), member(None, mask="api_error", wrote=3), task="b"),
        grouped(
            member(None, mask="context_overflow", wrote=0),
            _varied(1.0, 30, 31),
            member(None, mask="multimodal", wrote=9),
            _varied(0.0, 32),
            task="c",
        ),
    ]


def _lengths() -> list[Group]:
    """A tie broken by length, a lone success, a long failure, failures only."""
    return [
        grouped(
            member(1.0, wrote=30, sequences=[sequence(PROMPT, (10, 11))]),
            member(1.0, wrote=10, sequences=[sequence(PROMPT, (12,))]),
            member(0.0, wrote=50, sequences=[sequence(PROMPT, (13, 14))]),
            task="a",
        ),
        grouped(
            member(1.0, wrote=1000, sequences=[sequence(PROMPT, (20, 21))]),
            member(1.0, wrote=10, sequences=[sequence(PROMPT, (22,))]),
            member(0.0, wrote=5, sequences=[sequence(PROMPT, (23,))]),
            task="b",
        ),
        grouped(
            member(1.0, wrote=30, sequences=[sequence(PROMPT, (30,))]),
            member(0.0, wrote=10, sequences=[sequence(PROMPT, (31, 32))]),
            task="c",
        ),
        grouped(
            member(0.0, wrote=30, sequences=[sequence(PROMPT, (40,))]),
            member(0.0, wrote=10, sequences=[sequence(PROMPT, (41,))]),
            task="d",
        ),
    ]


def _pictured() -> list[Group]:
    shot = Picture(1, 3, "png", PNG)
    seen = sequence(
        (1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 5), (6, 7), (8,), tools=((9,),), images=(shot,)
    )
    return [
        grouped(member(1.0, sequences=[seen]), _varied(0.0, 10, 11), _varied(1.0, 12), task="a"),
    ]


def _flat() -> list[Group]:
    return [
        grouped(_varied(1.0, 10), _varied(1.0, 11), task="a"),
        grouped(_varied(0.0, 20), _varied(0.0, 21), task="b"),
        grouped(_varied(1.0, 30), task="c"),
    ]


#: The credit cases by name.
CREDITED: dict[str, Credited] = {
    "dapo-mixed": Credited(_mixed, "dapo", limit=2),
    "dapo-mixed-unlimited": Credited(_mixed, "dapo"),
    "dapo-overlong": Credited(_overlong, "dapo", budget=100),
    "dapo-overlong-no-budget": Credited(_overlong, "dapo"),
    "dapo-chained": Credited(_chained, "dapo"),
    "dapo-partial": Credited(_partial, "dapo"),
    "dapo-masked": Credited(_masked, "dapo"),
    "dapo-pictured-kl": Credited(_pictured, "dapo", {"kl_coef": 0.5}, anchored=True),
    "dapo-kl": Credited(_chained, "dapo", {"kl_coef": 0.5}, anchored=True),
    "dapo-flat": Credited(_flat, "dapo"),
    "dapo-warmup": Credited(_chained, "dapo", {"warmup": 5}, updates=2),
    "dapo-sum": Credited(_chained, "dapo", preset_only={"aggregation": "sum"}),
    "dr-grpo-lengths": Credited(_lengths, "dr-grpo"),
    "dr-grpo-partial": Credited(_partial, "dr-grpo"),
    "dr-grpo-floor-cap": Credited(
        _lengths, "dr-grpo", {"length_penalty": 0.5, "length_floor": 10, "length_cap": 0.9}
    ),
    "cispo-chained": Credited(_chained, "cispo"),
    "cispo-mixed": Credited(_mixed, "cispo", limit=2),
    "cispo-kl": Credited(_chained, "cispo", {"kl_coef": 0.5}, anchored=True),
    "fst-chained": Credited(_chained, "fst", anchored=True),
}


def _tool_loop(*actions: tuple[int, ...]) -> list[Record]:
    """Each prompt extends the last, scored minus a tenth of each token."""
    out: list[Record] = []
    prompt: tuple[int, ...] = PROMPT
    for at, action in enumerate(actions):
        found = made(prompt, action, seq=at + 1)
        out.append(replace(found, inference_logprobs=tuple(-0.1 * token for token in action)))
        prompt = (*prompt, *action, 50 + at)
    return out


def _served() -> Jobbed:
    """The served endings, one per trial."""
    return Jobbed(
        kind="dapo",
        group_size=4,
        tasks=("a", "b"),
        outcomes={
            "a": [
                {"reward": 1.0},
                {"reward": 1.0},
                {"reward": 0.0, "exception": "UnknownApiError"},
                {"reward": 0.0},
            ],
            "b": [
                {"reward": 1.0},
                {"reward": 0.0},
                {"reward": 1.0},
                {"reward": 1.0, "exception": "AgentTimeoutError"},
            ],
        },
        records={
            "a": [
                _tool_loop((10, 11), (12,)),
                [*_tool_loop((13, 14)), made(seq=2, error=BUDGET)],
                [made(seq=1, error=CONTEXT)],
                [*_tool_loop((15,)), made(seq=2, error="sampler: ConnectError")],
            ],
            "b": [
                [],
                _tool_loop((20,)),
                [replace(_tool_loop((21, 22))[0], bridged=True, cached_tokens=2)],
                _tool_loop((23,)),
            ],
        },
        preset=preset_of("dapo"),
        budget=100,
    )


def _fst(share: int) -> Jobbed:
    """Task a's spread comes from the texts, task b's from sampling."""
    first = [{"reward": 1.0}] * share + [{"reward": float(n % 2 == 0)} for n in range(share)]
    second = [{"reward": 0.0}] * share + [{"reward": float(n % 2 == 1)} for n in range(share)]
    return Jobbed(
        kind="fst",
        group_size=2 * share,
        tasks=("a", "b"),
        outcomes={"a": first[:share] + second[:share], "b": first[share:] + second[share:]},
        records={
            "a": [_tool_loop((10 + n,), (20 + n,)) for n in range(share)]
            + [_tool_loop((30 + n,)) for n in range(share)],
            "b": [_tool_loop((40 + n,)) for n in range(share)]
            + [_tool_loop((60 + n, 61 + n)) for n in range(share)],
        },
        preset=preset_of("fst"),
        texts=2,
        anchored=True,
    )


def _provider() -> Jobbed:
    return Jobbed(
        kind="evaluate",
        group_size=2,
        tasks=("alpha", "beta"),
        outcomes={
            "alpha": [
                {"reward": 1.0, "tokens": (100, 20, 10), "provider": "OpenRouter", "seconds": 30.0},
                {"reward": 0.5, "tokens": (80, 0, 12), "provider": "OpenRouter", "seconds": 12.0},
            ],
            "beta": [{"reward": 0.0, "exception": "AgentTimeoutError", "seconds": 10.0}, None],
        },
        records={},
    )


#: The job cases by name.
JOBBED: dict[str, Jobbed] = {
    "job-served": _served(),
    "job-fst-share-1": _fst(1),
    "job-fst-share-2": _fst(2),
    "job-provider": _provider(),
}

CASES: tuple[str, ...] = (*CREDITED, *JOBBED)


# ------------------------------------------------------------------- running one


async def replay(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    if name in CREDITED:
        return await _credited(CREDITED[name])
    return await _jobbed(JOBBED[name], tmp_path, monkeypatch)


async def _credited(case: Credited) -> dict[str, Any]:
    trainer = Trainer(FakeTrainingClient(positional=True), updates=case.updates)
    anchor = PositionalAnchor() if case.anchored else None
    # Stamped with the trainer's count, as a batch sampled at these weights would be.
    groups = [replace(one, updates=case.updates) for one in case.groups()]
    batch = await credit(groups, case.preset, trainer, anchor, limit=case.limit, budget=case.budget)
    return await _stepped(batch, case.preset, trainer, anchor)


async def _stepped(batch: Batch, preset: Preset, trainer: Trainer, anchor: Any) -> dict[str, Any]:
    update = None if batch.empty else await trainer.apply(batch, preset)
    client = trainer.client
    return {
        "batch": _batch(batch),
        "client": [_call(one) for one in client.calls],
        "sent": [_sent(one) for one in client.sent],
        "anchor": None if anchor is None else list(anchor.asked),
        "update": None if update is None else _update(update),
        "row": {key: value for key, value in row(0, batch, update).items() if key != "seconds"},
        "updates": trainer.updates,
    }


async def _jobbed(case: Jobbed, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    home, _ = _staged(case, tmp_path, monkeypatch)
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = ScriptedTrials(case.outcomes)
    trainer = Trainer(FakeTrainingClient(positional=True))
    opened.trainer = trainer
    batch_tasks = [tmp_path / "tasks" / "golden" / task for task in case.tasks]
    texts = _texts(tmp_path, case.texts)
    share = case.group_size // case.texts
    with opened:
        parts = [
            await opened.sample(batch_tasks, rollouts=share, index=0, modules=one) for one in texts
        ]
        # The proxy's budget, read while it is up, as the step reads it.
        budget = opened.endpoint.budget
    rolled = parts[0] if len(parts) == 1 else merged(parts, share)
    out: dict[str, Any] = {
        "verdicts": [_verdict(one) for one in verdicts(rolled)],
        "jobs": [_job(one) for one in record.read(opened.directory / record.JOBS)],
        "requests": _requests(opened.directory, rolled),
        "costs": record.read_json(opened.directory / record.COSTS),
    }
    if case.preset is None:
        return out
    groups = group(rolled, case.group_size)
    out["groups"] = [_group(one) for one in groups]
    anchor = PositionalAnchor() if case.anchored else None
    batch = await credit(groups, case.preset, trainer, anchor, budget=budget)
    return {**out, **(await _stepped(batch, case.preset, trainer, anchor))}


def _texts(tmp_path: Path, count: int) -> list[Candidate | None]:
    if count == 1:
        return [None]
    found: list[Candidate | None] = []
    for at in range(count):
        home = tmp_path / "texts" / str(at)
        shutil.copytree(VALID, home)
        if at:
            skill = home / "skills" / "solving" / "SKILL.md"
            skill.write_text(skill.read_text() + f"\nText {at}.\n", encoding="utf-8")
        found.append(seed(home))
    return found


# ------------------------------------------------------- through the trainer


class Budgeted(Endpoint):
    """An endpoint that only reports its token budget."""

    def __init__(self, budget: int | None) -> None:
        self.budget = budget

    async def open(self) -> None: ...

    def close(self) -> None: ...

    def connection(self, trial: Trial) -> Connection:
        return Connection(model="m")

    async def serve(self, path: str | None) -> None: ...

    async def record(self, job: str, trials: Values[Trial]) -> Records | None:
        return None


def typed(found: Values[Group], agent: Agent) -> list[types.Group]:
    """The case's groups as the trainer takes them: one trial per member, under the
    agent's one harness."""
    out: list[types.Group] = []
    for each in found:
        members = [
            types.Member(
                Trial("g", each.task, at, agent.harness[0], Path("g") / f"{each.task.name}__{at}"),
                one.verdict,
                one.sequences,
                one.sampled_tokens,
            )
            for at, one in enumerate(each.members)
        ]
        out.append(types.Group(each.task, agent, agent.harness[0], members))
    return out


async def _stepped_by(
    trainer: Procedure, agent: Agent, found: list[types.Group], client: FakeTrainingClient
) -> dict[str, Any]:
    """`step` and `log` through the trainer, in the frozen shape. A save is the policy's,
    not the step's, so the client's calls are read without it."""
    for each in found:
        each.shaped = trainer.shape(each)
        each.advantages = trainer.advantage(each)
    update = await trainer.step(agent, found)
    row = trainer.log(found, update)
    return {
        "client": [_call(one) for one in client.calls if one[0] != "save_weights_for_sampler"],
        "sent": [_sent(one) for one in client.sent],
        "anchor": None if trainer.anchor is None else list(trainer.anchor.asked),
        "update": None if update is None else _update(update),
        "row": {key: value for key, value in row.items() if key not in ("seconds", "jobs")},
    }


async def trained(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The case through the trainer: a credit case stepped from its groups, a job case
    sampled through `job` first."""
    if name in CREDITED:
        return await _credited_by(CREDITED[name], tmp_path)
    return await _jobbed_by(JOBBED[name], tmp_path, monkeypatch)


async def _credited_by(case: Credited, tmp_path: Path) -> dict[str, Any]:
    run = Run.open(BLUEPRINTS / case.kind, root=tmp_path / "runs")
    run.config.recipe = run.config.recipe.model_copy(update=dict(case.recipe))
    client = FakeTrainingClient(positional=True)
    agent = Agent(Policy(Budgeted(case.budget), TinkerClient(client)), [harness("pi")])
    trainer = TRAINERS[case.kind](run, anchor=PositionalAnchor() if case.anchored else None)
    trainer.updates = case.updates
    with run:
        return await _stepped_by(trainer, agent, typed(case.groups(), agent), client)


async def _jobbed_by(
    case: Jobbed, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    home, proxy = _staged(case, tmp_path, monkeypatch)
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = ScriptedTrials(case.outcomes)
    client = FakeTrainingClient(positional=True)
    served = opened.config.model.served
    policy = Policy(opened.endpoint, TinkerClient(client) if served else None)
    agent = Agent(policy, [harness(opened.config.rollout.harness)])
    procedure = TRAINERS.get(case.kind, Procedure)
    trainer = procedure(opened, anchor=PositionalAnchor() if case.anchored else None)
    batch_tasks = [tmp_path / "tasks" / "golden" / task for task in case.tasks]
    with opened:
        async with agent.policy:
            found = await trainer.job(agent, batch_tasks)
            members = [one for group in found for one in group.members]
            out: dict[str, Any] = {
                "verdicts": [_verdict(one.verdict) for one in members],
                "jobs": [_job(one) for one in record.read(opened.directory / record.JOBS)],
                "requests": _requests_of(opened.directory, [one.trial.name for one in members]),
                "costs": record.read_json(opened.directory / record.COSTS),
            }
            if case.preset is None:
                return out
            out["groups"] = [_members(group) for group in found]
            return {**out, **(await _stepped_by(trainer, agent, found, client))}


def _staged(
    case: Jobbed, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, ScriptedProxy]:
    """The fixture tasks and the case's blueprint in a working directory, the proxy
    scripted."""
    monkeypatch.chdir(tmp_path)
    for task in case.tasks:
        source = FIXTURES / "fixture" / ("alpha" if task not in ("alpha", "beta") else task)
        shutil.copytree(source, tmp_path / "tasks" / "golden" / task)
    home = tmp_path / "bp"
    home.mkdir()
    text = (BLUEPRINTS / case.kind / "run.toml").read_text(encoding="utf-8")
    text = text.replace('"aime-train"', '"golden"').replace('"fixture"', '"golden"')
    (home / "run.toml").write_text(text, encoding="utf-8")
    proxy = ScriptedProxy(
        queues={task: list(found) for task, found in case.records.items()},
        budget=case.budget,
    )
    monkeypatch.setattr("shipyard.sampling.endpoint.Proxy", proxy)
    return home, proxy


def _members(one: types.Group) -> dict[str, Any]:
    return {
        "task": one.task.name,
        "members": [
            {
                **_verdict(each.verdict),
                "sampled_tokens": each.sampled_tokens,
                "sequences": [_sequence(found) for found in each.sequences],
            }
            for each in one.members
        ],
    }


# ------------------------------------------------------------------ the shape


def _batch(batch: Batch) -> dict[str, Any]:
    fields = (
        "owners",
        "groups",
        "rollouts",
        "graded",
        "masked",
        "degenerate",
        "surplus",
        "overlong",
        "credited",
        "sequences",
        "reward_mean",
        "reward_spread",
        "length_mean",
        "anchor_kl",
        "reference_tokens",
        "anchor_tokens",
    )
    return {
        "datums": [_datum(one) for one in batch.datums],
        **{key: _plain(getattr(batch, key)) for key in fields},
    }


def _datum(one: tinker.Datum) -> dict[str, Any]:
    return {
        "input": _chunks(one.model_input),
        **{key: value.tolist() for key, value in one.loss_fn_inputs.items()},
    }


def _chunks(found: tinker.ModelInput) -> list[Any]:
    out: list[Any] = []
    for chunk in found.chunks:
        if hasattr(chunk, "tokens"):
            out.append(list(chunk.tokens))
        else:
            data = (
                chunk.data if isinstance(chunk.data, str) else base64.b64encode(chunk.data).decode()
            )
            out.append({"image": int(chunk.length), "format": chunk.format, "data": data})
    return out


def _call(one: tuple[Any, ...]) -> list[Any]:
    if one[0] == "optim_step":
        return ["optim_step", one[1].model_dump()]
    return [_plain(part) for part in one]


def _sent(one: tinker.Datum) -> dict[str, Any]:
    return _datum(one)


def _update(update: Any) -> dict[str, Any]:
    return {
        key: getattr(update, key)
        for key in (
            "train_tokens",
            "substeps",
            "learning_rate",
            "loss_fn",
            "kl_v1",
            "kl_v2",
            "entropy",
        )
    }


def _verdict(one: Verdict) -> dict[str, Any]:
    return {"reward": one.reward, "mask": one.mask, "ended": one.ended}


def _group(one: Group) -> dict[str, Any]:
    return {
        "task": one.task.name,
        "updates": one.updates,
        "members": [
            {
                **_verdict(each.verdict),
                "sampled_tokens": each.sampled_tokens,
                "sequences": [_sequence(found) for found in each.sequences],
            }
            for each in one.members
        ],
    }


def _sequence(one: Sequence) -> dict[str, Any]:
    return {
        "tokens": list(one.tokens),
        "spans": [list(span) for span in one.spans],
        "mu": None if one.mu is None else list(one.mu),
        "images": [[shot.offset, shot.length, shot.format, shot.data] for shot in one.images],
    }


def _job(one: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in one.items() if key not in ("at", "job")}


def _requests(directory: Path, rolled: Rollouts) -> list[dict[str, Any]]:
    return _requests_of(directory, [Path(trial).name for trial in rolled.trials])


def _requests_of(directory: Path, trials: Values[str]) -> list[dict[str, Any]]:
    places = {name: at for at, name in enumerate(trials)}
    out: list[dict[str, Any]] = []
    for one in record.read(directory / record.REQUESTS):
        out.append(
            {
                **{k: v for k, v in one.items() if k not in ("at", "job", "trial")},
                "trial": places[one["trial"]],
            }
        )
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(one) for one in value]
    if isinstance(value, dict):
        return {str(key): _plain(one) for key, one in value.items()}
    if isinstance(value, list):
        return [_plain(one) for one in value]
    return value


# ---------------------------------------------------------------- the comparison


def differences(found: Any, frozen: Any, at: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(frozen, dict) and isinstance(found, dict):
        for key in sorted(set(frozen) | set(found)):
            if key not in found:
                out.append(f"{at}.{key}: missing (frozen {frozen[key]!r})")
            elif key not in frozen:
                out.append(f"{at}.{key}: new (found {found[key]!r})")
            else:
                out.extend(differences(found[key], frozen[key], f"{at}.{key}"))
        return out
    if isinstance(frozen, list) and isinstance(found, list):
        if len(found) != len(frozen):
            out.append(f"{at}: {len(found)} item(s), frozen {len(frozen)}")
            return out
        for index, (a, b) in enumerate(zip(found, frozen, strict=True)):
            out.extend(differences(a, b, f"{at}[{index}]"))
        return out
    if _number(found) and _number(frozen):
        if abs(float(found) - float(frozen)) > TOLERANCE:
            out.append(f"{at}: {found!r}, frozen {frozen!r}")
        return out
    if found != frozen:
        out.append(f"{at}: {found!r}, frozen {frozen!r}")
    return out


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def updating() -> bool:
    return bool(os.environ.get(UPDATE))


def frozen_path(name: str) -> Path:
    return GOLDEN / f"{name}.json"


def read_frozen(name: str) -> dict[str, Any] | None:
    path = frozen_path(name)
    return record.read_json(path) if path.is_file() else None


def write_frozen(name: str, found: Mapping[str, Any]) -> None:
    GOLDEN.mkdir(exist_ok=True)
    record.write_json(frozen_path(name), dict(found))
