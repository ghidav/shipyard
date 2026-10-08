"""The blueprint: what `run.toml` may say, loaded and validated into one object."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harbor.models.environment_type import EnvironmentType
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator

from shipyard.data import NoSuchDataset, held_out, home_of, tasks
from shipyard.modules import Inadmissible, seed
from shipyard.record import CONFIG

TINKER = "tinker"
#: The recipe kinds, one module each under `shipyard.recipes`; the first three train.
KINDS = ("dapo", "dr-grpo", "cispo", "gepa", "fst", "evaluate")
#: The SDK's floor on a checkpoint's TTL: `TrainingClient.save_state` takes "between
#: 1 hour (3600) and 10 years" (tinker/lib/public_interfaces/training_client.py).
MIN_TTL_HOURS = 1.0
#: What `[rollout] sandbox` may name: Harbor's environment types.
SANDBOXES = tuple(kind.value for kind in EnvironmentType)
#: What `check` says of a blueprint that cannot run yet.
NO_HARNESS = (
    "[rollout] harness: name the harness; Harbor reads an unnamed agent as its oracle, "
    "which runs the task's reference solution"
)
NO_REFLECTOR = (
    "[recipe] reflection_harness: name the harness the reflector runs as; Harbor reads an "
    "unnamed agent as its oracle, and the reflection task has no reference solution"
)
#: The reflector's container when a gepa blueprint names none: a shell and a Python.
REFLECTION_IMAGE = "python:3.12-slim"


class ConfigError(ValueError):
    """Every problem a blueprint has, one per line, so a reader fixes them in one pass."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


class _Table(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Model(_Table):
    name: str
    provider: str = TINKER
    from_checkpoint: str | None = None
    lora_rank: int | None = None
    restore_optimizer: bool = False

    @property
    def served(self) -> bool:
        """Whether this run serves the weights itself, which only Tinker's can be."""
        return self.provider == TINKER


class Data(_Table):
    dataset: str | list[str]
    batch_size: int = Field(ge=1)
    group_size: int = Field(default=1, ge=1)
    epochs: int = Field(default=1, ge=1)
    seed: int = 0

    @field_validator("dataset", mode="before")
    @classmethod
    def _named(cls, value: Any) -> Any:
        """One name or a list of names; anything else is said once, not once per shape."""
        if isinstance(value, str) or (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        ):
            return value
        raise ValueError("a dataset name or a list of them")


class Rollout(_Table):
    harness: str
    sandbox: str = "docker"
    concurrency: int = Field(default=4, ge=1)
    timeout: float | None = None
    setup_timeout: float | None = None
    env: dict[str, str] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    endpoint_url: str = ""
    host: str = "host.docker.internal"
    bind: str = "0.0.0.0"
    bind_port: int = Field(default=0, ge=0, le=65535)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = Field(default=8192, ge=1)
    max_context: int = Field(default=0, ge=0)
    renderer: str = ""
    cut_volatile: bool = True
    fill_context: bool = False
    check_turns: bool = False
    jobs_dir: str = "jobs"


#: Optimizer steps per batch for dapo and cispo: DAPO 4.1 ("16 gradient updates for each
#: rollout step") and MiniMax-M1 3.1 ("16 rounds of off-policy updates per generation batch").
PAPER_SUBSTEPS = 16
#: Extra sampling rounds a dapo or cispo step may take to fill its batch: DAPO's released
#: recipe stops at ten generation batches (verl recipe/dapo, `max_num_gen_batches=10`).
PAPER_REFILL = 9
#: cispo's ceiling as an epsilon, 4.0 on the weight: ScaleRL A.17.2 and FST appendix D.
PAPER_CISPO_HIGH = 3.0
#: DAPO's soft overlong punishment (Eq. 13), which MiniMax-M1 3.1 takes for CISPO: a ramp
#: over the last L_cache / L_max of the budget, 4,096 of 20,480 tokens (4.1), down to -1 at
#: the limit on DAPO's reward of -1 or 1 (Eq. 7). On Harbor's 0 to 1 that -1 is -0.5, which
#: gives the same advantages once divided by the spread.
PAPER_OVERLONG_BUFFER = 0.2
PAPER_OVERLONG_PENALTY = 0.5
#: The most dr-grpo's length rule takes from a solved answer by default, so a solved answer
#: never falls below a failure scored 0.5 or less.
LENGTH_CAP = 0.5


class Gradient(_Table):
    """What the three gradient recipes share; each adds its own clipping and shaping knobs.
    One substep is FST's (one mini-batch a step) and Dr. GRPO's, which states no other."""

    learning_rate: float = Field(gt=0)
    substeps: int = Field(default=1, ge=1)
    #: Steps over which the learning rate rises to `learning_rate`. Off by default: the
    #: papers' warm-ups (DAPO's 20, FST's 10) are sized for runs of hundreds of steps.
    warmup: int = Field(default=0, ge=0)
    reference: Literal["trainer", "sampler"] = "trainer"
    kl_coef: float = Field(default=0.0, ge=0)
    modules: str | None = None


class DapoRecipe(Gradient):
    kind: Literal["dapo"]
    substeps: int = Field(default=PAPER_SUBSTEPS, ge=1)
    clip_low: float = Field(default=0.2, gt=0, lt=1)
    clip_high: float = Field(default=0.28, gt=0)
    refill: int = Field(default=PAPER_REFILL, ge=0)
    overlong_penalty: float = Field(default=PAPER_OVERLONG_PENALTY, ge=0, allow_inf_nan=False)
    overlong_buffer: float = Field(default=PAPER_OVERLONG_BUFFER, gt=0, le=1)


class DrGrpoRecipe(Gradient):
    kind: Literal["dr-grpo"]
    clip: float = Field(default=0.2, gt=0, lt=1)
    length_penalty: float = Field(default=0.0, ge=0)
    length_floor: int = Field(default=0, ge=0)
    length_cap: float = Field(default=LENGTH_CAP, gt=0)
    refill: int = Field(default=0, ge=0)


class CispoRecipe(Gradient):
    kind: Literal["cispo"]
    substeps: int = Field(default=PAPER_SUBSTEPS, ge=1)
    clip_high: float = Field(default=PAPER_CISPO_HIGH, gt=0)
    refill: int = Field(default=PAPER_REFILL, ge=0)
    overlong_penalty: float = Field(default=PAPER_OVERLONG_PENALTY, ge=0, allow_inf_nan=False)
    overlong_buffer: float = Field(default=PAPER_OVERLONG_BUFFER, gt=0, le=1)


#: The share of a run's tasks gepa holds out to select on when `[recipe] pareto` is unset,
#: rounded down, as in three of GEPA's four benchmarks: GEPA §4.1 (split sizes) and §4.3
#: (the validation set is D_pareto). HotpotQA, IFBench and HoVer hold 300 validation tasks
#: beside 150 training ones (PUPA alone splits 111 and 111), and the one split the paper
#: draws itself, IFBench's, is one of those.
PAPER_PARETO = (2, 3)
#: GEPA 4.3: "All GEPA optimization runs use a minibatch size of 3".
PAPER_MINIBATCH = 3


class GepaRecipe(_Table):
    """The search's knobs: the reflector in three halves, the seed directory, the tasks held
    out to select on, the minibatch a child is judged on, the rollouts the search may spend,
    and when a quiet proposer ends it."""

    kind: Literal["gepa"]
    reflection_harness: str
    reflection_model: str | None = None
    reflection_image: str = REFLECTION_IMAGE
    modules: str = "modules"
    pareto: int | str | None = None
    minibatch: int = Field(default=PAPER_MINIBATCH, ge=1)
    budget: int | None = Field(default=None, ge=1)
    patience: int = Field(default=3, ge=1)
    edits: Literal["rewrite", "incremental"] = "rewrite"

    @field_validator("pareto", mode="before")
    @classmethod
    def _held_out(cls, value: Any) -> Any:
        """A dataset name or a count; a boolean, a fraction or a blank name is neither."""
        if value is None or (isinstance(value, str) and value.strip()):
            return value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        raise ValueError("a dataset name, or how many of the run's tasks to hold out (0 or more)")


#: The knobs each slow recipe of fst takes; a knob of another one is an error.
SLOW_KNOBS = {
    "dapo": ("clip_low", "clip_high"),
    "dr-grpo": ("clip", "length_penalty", "length_floor", "length_cap"),
    "cispo": ("clip_high",),
}


class FstRecipe(Gradient):
    """Fast-slow training: a slow recipe for the weights, gepa's knobs for the population
    of texts, and the cycle that interleaves them. Defaults are the paper's."""

    kind: Literal["fst"]
    slow: Literal["dapo", "dr-grpo", "cispo"] = "cispo"
    kl_coef: float = Field(default=0.001, ge=0)
    cycle: int = Field(default=6, ge=1)
    population: int = Field(default=4, ge=1)
    anchor: int | None = Field(default=None, ge=1)
    clip_low: float | None = Field(default=None, gt=0, lt=1)
    clip_high: float | None = Field(default=None, gt=0)
    clip: float | None = Field(default=None, gt=0, lt=1)
    length_penalty: float | None = Field(default=None, ge=0)
    length_floor: int | None = Field(default=None, ge=0)
    length_cap: float | None = Field(default=None, gt=0)
    reflection_harness: str
    reflection_model: str | None = None
    reflection_image: str = REFLECTION_IMAGE
    modules: str = "modules"
    minibatch: int = Field(default=3, ge=1)
    budget: int | None = Field(default=None, ge=1)
    patience: int = Field(default=3, ge=1)
    edits: Literal["rewrite", "incremental"] = "incremental"

    @field_validator(
        "clip_low", "clip_high", "clip", "length_penalty", "length_floor", "length_cap"
    )
    @classmethod
    def _of_slow(cls, value: Any, info: Any) -> Any:
        slow = info.data.get("slow", "cispo")
        if value is not None and info.field_name not in SLOW_KNOBS[slow]:
            raise ValueError(
                f"not a knob of slow = {slow!r}, which takes {', '.join(SLOW_KNOBS[slow])}"
            )
        return value


class EvaluateRecipe(_Table):
    kind: Literal["evaluate"]
    modules: str | None = None


Recipe = DapoRecipe | DrGrpoRecipe | CispoRecipe | GepaRecipe | FstRecipe | EvaluateRecipe


class Checkpoints(_Table):
    every: int = Field(default=1, ge=1)
    ttl_hours: float = Field(default=168.0, ge=MIN_TTL_HOURS)


class Blueprint(_Table):
    model: Model
    data: Data
    rollout: Rollout
    recipe: Recipe = Field(discriminator="kind")
    checkpoints: Checkpoints = Field(default_factory=Checkpoints)
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)
    _home: Path = PrivateAttr(default_factory=Path)

    @property
    def datasets(self) -> list[str]:
        named = self.data.dataset
        return [named] if isinstance(named, str) else list(named)

    @property
    def raw(self) -> dict[str, Any]:
        """The TOML as parsed, before defaults: what the run's verbatim copy holds."""
        return self._raw

    @property
    def trains(self) -> bool:
        """Whether the recipe moves the weights: the gradient recipes and fst."""
        return isinstance(self.recipe, Gradient)

    @property
    def home(self) -> Path:
        """The directory the blueprint was loaded from, which `[recipe] modules` is under."""
        return self._home


def config_file(path: Path) -> Path:
    """The `run.toml` a blueprint path names: the file itself, or the one in the directory."""
    path = Path(path)
    return path if path.is_file() else path / CONFIG


def load(path: Path) -> Blueprint:
    """The blueprint at `path`, a directory holding `run.toml` or the file itself.
    Raises `ConfigError` listing every problem rather than the first one met."""
    file = config_file(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError([f"cannot read {file}: {error.strerror or error}"]) from error
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError([f"{file.name} is not readable TOML: {error}"]) from error
    try:
        blueprint = Blueprint.model_validate(raw)
    except ValidationError as error:
        raise ConfigError([_problem(found) for found in error.errors()]) from error
    blueprint._raw = raw
    blueprint._home = file.parent
    return blueprint


def modules_dir(loaded: Blueprint) -> Path | None:
    """The directory `[recipe] modules` names, relative to the blueprint; None when unset."""
    named = getattr(loaded.recipe, "modules", None)
    return loaded.home / named if named else None


def search_sets(loaded: Blueprint) -> tuple[list[Path], list[Path]]:
    """gepa's two task lists, GEPA's D_feedback and D_pareto (Alg. 1 line 1): the tasks its
    minibatches are drawn from, and the tasks every candidate it keeps is scored on. A dataset
    name is D_pareto beside the run's tasks; a count holds that many of the run's tasks out,
    drawn with `[data] seed`, and 0 keeps every task in both (GEPA 6); unset holds out two
    thirds, as in three of GEPA's four benchmarks. Raises ValueError when either list would
    be empty."""
    listed = [task for name in loaded.datasets for task in tasks(name)]
    named = loaded.recipe.pareto
    if isinstance(named, str):
        return listed, tasks(named)
    if named == 0:
        return listed, list(listed)
    share, whole = PAPER_PARETO
    count = len(listed) * share // whole if named is None else named
    if count < 1:
        raise ValueError(
            "one task cannot be split; 0 selects on the task the search reflects on, and a "
            "dataset name selects on that dataset"
        )
    if count >= len(listed):
        raise ValueError(f"holding out {count} of {_tasks(len(listed))} leaves none to reflect on")
    return held_out(listed, count, seed=loaded.data.seed)


def _problem(error: dict[str, Any]) -> str:
    """One validation error as `[table] key: problem`, in the blueprint's own words."""
    loc = [str(part) for part in error["loc"]]
    kind, message = error["type"], error["msg"]
    if len(loc) > 1 and loc[0] == "recipe" and loc[1] in KINDS:
        del loc[1]  # pydantic names the matched union member; the file does not
    if kind == "union_tag_invalid":
        tag = (error.get("ctx") or {}).get("tag")
        return f"[recipe] kind: no recipe called {tag!r}; one of {', '.join(KINDS)}"
    if kind == "union_tag_not_found":
        return "[recipe] kind: field required"
    if message.startswith("Value error, "):
        message = message[len("Value error, ") :]
    if len(loc) == 1:
        if kind == "extra_forbidden":
            what = "unknown table" if isinstance(error.get("input"), dict) else "unknown key"
        elif kind == "missing":
            what = "table required"
        else:
            what = message
        return f"[{loc[0]}]: {what}"
    table, key = ".".join(loc[:-1]), loc[-1]
    what = {"missing": "field required", "extra_forbidden": "unknown key"}.get(kind, message)
    return f"[{table}] {key}: {what}"


@dataclass(frozen=True)
class Finding:
    level: Literal["ok", "warning", "blocked"]
    text: str


def check(blueprint: Path) -> list[Finding]:
    """What `shipyard check` says: blocked once per problem, else one ok line per fact.
    Datasets are looked for under `tasks/` in the working directory, as `run` looks."""
    try:
        loaded = load(blueprint)
    except ConfigError as error:
        return [Finding("blocked", problem) for problem in error.problems]
    return findings(loaded)


def findings(loaded: Blueprint) -> list[Finding]:
    """The findings of a blueprint that loaded; the sandbox's and a served model's come
    from `preflight`, imported here so the CLI stays light."""
    from shipyard.preflight import findings as preflight_findings

    model, rollout = loaded.model, loaded.rollout
    served = " (served by this run)" if model.served else ""
    plural = "s" if len(loaded.datasets) > 1 else ""
    found = [
        Finding("ok", f"recipe {loaded.recipe.kind}"),
        Finding("ok", f"model {model.name}, provider {model.provider}{served}"),
        Finding("ok", f"dataset{plural} {', '.join(loaded.datasets)}"),
        Finding("ok", f"harness {rollout.harness}, sandbox {rollout.sandbox}"),
    ]
    found += [_dataset_finding(name) for name in loaded.datasets]
    if (checkpoint := _checkpoint_finding(loaded)) is not None:
        found.append(checkpoint)
    if (directory := modules_dir(loaded)) is not None:
        found.append(_modules_finding(directory))
    if isinstance(loaded.recipe, GepaRecipe | FstRecipe):
        found.append(_reflector_finding(loaded.recipe))
    if isinstance(loaded.recipe, GepaRecipe):
        found += _pareto_findings(loaded)
    if isinstance(loaded.recipe, FstRecipe) and loaded.data.group_size % loaded.recipe.population:
        k, g = loaded.recipe.population, loaded.data.group_size
        found.append(
            Finding(
                "blocked",
                f"[recipe] population: {k} does not divide [data] group_size {g}; each "
                "candidate takes group_size / population rollouts of every task's group",
            )
        )
    if (cap := _length_cap(loaded.recipe)) is not None and cap >= 1:
        found.append(
            Finding(
                "warning",
                f"[recipe] length_cap: {cap} lets the length rule take a solved answer's "
                "whole reward, so a long solved answer can score as low as a failure",
            )
        )
    if not rollout.harness.strip():
        found.append(Finding("blocked", NO_HARNESS))
    if rollout.sandbox not in SANDBOXES:
        named = f"no Harbor environment called {rollout.sandbox!r}; one of {', '.join(SANDBOXES)}"
        found.append(Finding("blocked", f"[rollout] sandbox: {named}"))
    found += preflight_findings(loaded)
    return found


def _length_cap(recipe: Any) -> float | None:
    """dr-grpo's cap on its length rule, as set or defaulted, while the rule is on
    (`length_penalty > 0`); None for a recipe without the rule or with it off."""
    if isinstance(recipe, DrGrpoRecipe):
        penalty, cap = recipe.length_penalty, recipe.length_cap
    elif isinstance(recipe, FstRecipe) and recipe.slow == "dr-grpo":
        penalty = recipe.length_penalty or 0.0
        cap = LENGTH_CAP if recipe.length_cap is None else recipe.length_cap
    else:
        return None
    return cap if penalty > 0 else None


def _checkpoint_finding(loaded: Blueprint) -> Finding | None:
    """`blocked` when `from_checkpoint` is the other column of `checkpoints.jsonl`: a training
    run resumes from a `state_path`, and anything that only samples needs a `sampler_path`."""
    named = loaded.model.from_checkpoint or ""
    if loaded.trains and "/sampler_weights/" in named:
        return Finding(
            "blocked",
            "[model] from_checkpoint: a training run resumes from a checkpoint's state_path "
            "(tinker://.../weights/...), not its sampler_path",
        )
    if not loaded.trains and "/weights/" in named and "/sampler_weights/" not in named:
        return Finding(
            "blocked",
            f"[model] from_checkpoint: {loaded.recipe.kind} samples a checkpoint's sampler_path "
            "(tinker://.../sampler_weights/...), not its state_path",
        )
    return None


def _modules_finding(directory: Path) -> Finding:
    """`ok` with the component count and digest, or `blocked` with why `seed` refused."""
    try:
        candidate = seed(directory)
    except Inadmissible as refused:
        return Finding("blocked", f"[recipe] modules: {refused}")
    return Finding(
        "ok", f"modules: {len(candidate)} component(s) under {directory} ({candidate.digest})"
    )


def _reflector_finding(recipe: GepaRecipe | FstRecipe) -> Finding:
    """`ok` naming the reflector's harness, model and image, or `blocked` for a blank
    harness; the image is a string by schema, and Harbor pulls it at the first round."""
    if not recipe.reflection_harness.strip():
        return Finding("blocked", NO_REFLECTOR)
    model = recipe.reflection_model or "the harness's own default"
    return Finding(
        "ok",
        f"reflector {recipe.reflection_harness}, model {model}, image {recipe.reflection_image}",
    )


def _pareto_findings(loaded: Blueprint) -> list[Finding]:
    """A named held-out dataset's own finding, then `ok` with how the tasks split, or
    `blocked` with why they cannot; no split line when a dataset is missing, which that
    dataset's own finding already says."""
    named = loaded.recipe.pareto
    found = [_dataset_finding(named)] if isinstance(named, str) else []
    try:
        feedback, pareto = search_sets(loaded)
    except NoSuchDataset:
        return found
    except ValueError as refused:
        return [*found, Finding("blocked", f"[recipe] pareto: {refused}")]
    if isinstance(named, str):
        said = f"dataset {named}'s {_tasks(len(pareto))} to select on"
        said += f", the run's {_tasks(len(feedback))} to reflect on"
    elif named == 0:
        said = f"none held out; {_tasks(len(feedback))} both reflected and selected on"
    else:
        said = f"{len(pareto)} of {_tasks(len(feedback) + len(pareto))} held out with seed "
        said += f"{loaded.data.seed} to select on, {len(feedback)} to reflect on"
    return [*found, Finding("ok", f"pareto: {said}")]


def _tasks(count: int) -> str:
    return f"{count} task" if count == 1 else f"{count} tasks"


def _dataset_finding(name: str) -> Finding:
    """`ok` with the task count under `tasks/<name>`, or `blocked` with the path."""
    try:
        listed = tasks(name)
    except NoSuchDataset as missing:
        return Finding("blocked", str(missing))
    plural = "" if len(listed) == 1 else "s"
    return Finding("ok", f"dataset {name}: {len(listed)} task{plural} under {home_of(name)}")
