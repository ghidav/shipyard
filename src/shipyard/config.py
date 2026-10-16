"""The blueprint: the keys `run.toml` accepts, loaded and validated into one object."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from shipyard.record import CONFIG

TINKER = "tinker"
#: The recipe kinds, each but evaluate a module under `shipyard.training.recipes`.
KINDS = ("dapo", "dr-grpo", "cispo", "gepa", "fst", "evaluate")
#: The SDK's floor on a checkpoint's TTL: `TrainingClient.save_state` takes "between
#: 1 hour (3600) and 10 years" (tinker/lib/public_interfaces/training_client.py).
MIN_TTL_HOURS = 1.0
#: What `[rollout] sandbox` may name: Harbor's environment types.
REFLECTION_IMAGE = "python:3.12-slim"


class ConfigError(ValueError):
    """All the problems found in a blueprint, one per line."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


class _Table(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Agent(_Table):
    """What acts in a trial: the model, who serves it and where its weights start, the
    harness that drives it with its Harbor kwargs and env, and the modules it carries."""

    model: str
    provider: str = TINKER
    from_checkpoint: str | None = None
    harness: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    #: A modules directory, relative to the blueprint: carried into every job, or the seed
    #: of a search (`modules` when unset).
    modules: str | None = None

    @property
    def served(self) -> bool:
        """True when the provider is Tinker, the only provider whose weights this run serves."""
        return self.provider == TINKER


class Data(_Table):
    #: The tasks to learn from; the evaluate kind has none.
    train: str | list[str] | None = None
    #: The tasks to measure on; a run that names them ends with an evaluation.
    test: str | list[str] | None = None
    batch_size: int = Field(ge=1)
    group_size: int = Field(default=1, ge=1)
    #: Attempts per test task; `group_size` when unset.
    test_group_size: int | None = Field(default=None, ge=1)
    seed: int = 0

    @model_validator(mode="before")
    @classmethod
    def _dataset(cls, value: Any) -> Any:
        """`dataset` is the older spelling of `train`; a run is no longer counted in epochs."""
        if isinstance(value, dict) and (
            gone := sorted({"epochs", "evaluate_every"} & value.keys())
        ):
            raise ValueError(
                f"{' and '.join(gone)}: a run is counted in steps; set [recipe] steps and "
                "[recipe] evaluate_every"
            )
        if isinstance(value, dict) and "dataset" in value:
            if "train" in value:
                raise ValueError("`dataset` is the older spelling of `train`; name one of them")
            value = dict(value)
            value["train"] = value.pop("dataset")
        return value

    @field_validator("train", "test", mode="before")
    @classmethod
    def _named(cls, value: Any) -> Any:
        """A dataset name or a list of names."""
        if (
            value is None
            or isinstance(value, str)
            or (isinstance(value, list) and all(isinstance(item, str) for item in value))
        ):
            return value
        raise ValueError("a dataset name or a list of them")


class Rollout(_Table):
    """How trials run and how the model reaches them: the sandbox, the limits, the sampling
    settings and the proxy."""

    sandbox: str = "docker"
    concurrency: int = Field(default=4, ge=1)
    timeout: float | None = None
    setup_timeout: float | None = None
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
    fill_context: bool = False
    check_turns: bool = False
    jobs_dir: str = "jobs"
    #: A variable each trial's environment gets that no other trial in flight may share, by
    #: the variable of this process holding its values: one per trial at once.
    secrets: dict[str, str] = Field(default_factory=dict)


#: Optimizer steps per batch for dapo and cispo: DAPO 4.1 ("16 gradient updates for each
#: rollout step") and MiniMax-M1 3.1 ("16 rounds of off-policy updates per generation batch").
PAPER_SUBSTEPS = 16
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
    """The knobs the gradient recipes share; each adds its own clipping and shaping knobs.
    The default of one substep is FST's (one mini-batch a step) and Dr. GRPO's, which states
    no other."""

    learning_rate: float = Field(gt=0)
    #: The steps the run takes; each draws its batches from the generator.
    steps: int = Field(ge=1)
    #: Evaluate after every this many steps; 0 is at the end only.
    evaluate_every: int = Field(default=0, ge=0)
    substeps: int = Field(default=1, ge=1)
    #: Steps over which the learning rate rises to `learning_rate`. Off by default: the
    #: papers' warm-ups (DAPO's 20, FST's 10) are sized for runs of hundreds of steps.
    warmup: int = Field(default=0, ge=0)
    reference: Literal["trainer", "sampler"] = "trainer"
    kl_coef: float = Field(default=0.0, ge=0)
    #: The LoRA rank of the adapter trained; Tinker's default when unset.
    lora_rank: int | None = None
    #: With `[agent] from_checkpoint`, resume its optimizer state too.
    restore_optimizer: bool = False


class DapoRecipe(Gradient):
    kind: Literal["dapo"]
    substeps: int = Field(default=PAPER_SUBSTEPS, ge=1)
    clip_low: float = Field(default=0.2, gt=0, lt=1)
    clip_high: float = Field(default=0.28, gt=0)
    overlong_penalty: float = Field(default=PAPER_OVERLONG_PENALTY, ge=0, allow_inf_nan=False)
    overlong_buffer: float = Field(default=PAPER_OVERLONG_BUFFER, gt=0, le=1)


class DrGrpoRecipe(Gradient):
    kind: Literal["dr-grpo"]
    clip: float = Field(default=0.2, gt=0, lt=1)
    length_penalty: float = Field(default=0.0, ge=0)
    length_floor: int = Field(default=0, ge=0)
    length_cap: float = Field(default=LENGTH_CAP, gt=0)


class CispoRecipe(Gradient):
    kind: Literal["cispo"]
    substeps: int = Field(default=PAPER_SUBSTEPS, ge=1)
    clip_high: float = Field(default=PAPER_CISPO_HIGH, gt=0)
    overlong_penalty: float = Field(default=PAPER_OVERLONG_PENALTY, ge=0, allow_inf_nan=False)
    overlong_buffer: float = Field(default=PAPER_OVERLONG_BUFFER, gt=0, le=1)


#: The seed a search reads when `[agent] modules` names none.
SEARCH_MODULES = "modules"
#: GEPA 4.3: "All GEPA optimization runs use a minibatch size of 3".
PAPER_MINIBATCH = 3
#: No patience: the budget alone ends the search, as in GEPA's released code, whose
#: `NoImprovementStopper` (gepa 0.1.4, utils/stop_condition.py) runs only when passed in
#: `stop_callbacks`, and `api.py` adds none of its own.
GEPA_PATIENCE = None


class Search(_Table):
    """The search's knobs, gepa's and fst's: the reflector (harness, model and image), the
    minibatch a child is judged on, the rollout budget, the number of rounds without a better
    Pareto that ends the search, and how the reflector edits. It seeds from `[agent]
    modules`."""

    reflection_harness: str
    reflection_model: str | None = None
    reflection_image: str = REFLECTION_IMAGE
    minibatch: int = Field(default=PAPER_MINIBATCH, ge=1)
    budget: int | None = Field(default=None, ge=1)
    patience: int | None = Field(default=GEPA_PATIENCE, ge=1)
    edits: Literal["rewrite", "incremental"] = "rewrite"


class GepaRecipe(Search):
    """The search alone, with the tasks held out to select on."""

    kind: Literal["gepa"]
    pareto: int | str | None = None

    @field_validator("pareto", mode="before")
    @classmethod
    def _held_out(cls, value: Any) -> Any:
        """A dataset name or a count. Booleans, fractions and blank names are rejected."""
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


class FstRecipe(Search, Gradient):
    """Fast-slow training: a slow recipe for the weights, the search's knobs for the
    population of texts, and the cycle that interleaves them. Defaults are the paper's."""

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


Recipe = DapoRecipe | DrGrpoRecipe | CispoRecipe | GepaRecipe | FstRecipe | EvaluateRecipe


class Checkpoints(_Table):
    every: int = Field(default=1, ge=1)
    ttl_hours: float = Field(default=168.0, ge=MIN_TTL_HOURS)


class Wandb(_Table):
    """The wandb project the trainer's rows go to, beside its directory."""

    project: str
    entity: str | None = None


class Logging(_Table):
    """Where the trainer writes besides its directory: each a table of its own."""

    wandb: Wandb | None = None


class Blueprint(_Table):
    agent: Agent
    data: Data
    rollout: Rollout = Field(default_factory=Rollout)
    recipe: Recipe = Field(discriminator="kind")
    checkpoints: Checkpoints = Field(default_factory=Checkpoints)
    logging: Logging = Field(default_factory=Logging)
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)
    _home: Path = PrivateAttr(default_factory=Path)

    @model_validator(mode="after")
    def _sets(self) -> Blueprint:
        """The evaluate kind measures: `train` alone is read as `test`. A search seeds from
        `modules` when `[agent] modules` names nothing."""
        if isinstance(self.recipe, Search) and self.agent.modules is None:
            self.agent.modules = SEARCH_MODULES
        data = self.data
        if self.recipe.kind == "evaluate":
            if data.test is None:
                data.test, data.train = data.train, None
            if data.test is None:
                raise ValueError("[data] test: field required")
        elif data.train is None:
            raise ValueError("[data] train: field required")
        return self

    @property
    def datasets(self) -> list[str]:
        """The train datasets by name."""
        return _names(self.data.train)

    @property
    def test_sets(self) -> list[str]:
        return _names(self.data.test)

    @property
    def raw(self) -> dict[str, Any]:
        """The TOML as parsed, before defaults. The run's verbatim copy holds this."""
        return self._raw

    @property
    def trains(self) -> bool:
        """Whether the recipe moves the weights: the gradient recipes and fst."""
        return isinstance(self.recipe, Gradient)

    @property
    def home(self) -> Path:
        """The directory the blueprint was loaded from; `[agent] modules` is relative to it."""
        return self._home


def _names(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def config_file(path: Path) -> Path:
    """The `run.toml` for a blueprint path: the path if it is a file, else the `run.toml` in it."""
    path = Path(path)
    return path if path.is_file() else path / CONFIG


def load(path: Path) -> Blueprint:
    """Load the blueprint at `path`, a directory holding `run.toml` or the file.
    Raises `ConfigError` listing every problem."""
    file = config_file(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError([f"cannot read {file}: {error.strerror or error}"]) from error
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError([f"{file.name} is not readable TOML: {error}"]) from error
    if moved := _moved(raw):
        raise ConfigError(moved)
    try:
        blueprint = Blueprint.model_validate(raw)
    except ValidationError as error:
        raise ConfigError([_problem(found) for found in error.errors()]) from error
    blueprint._raw = raw
    blueprint._home = file.parent
    return blueprint


def modules_dir(loaded: Blueprint) -> Path | None:
    """The directory `[agent] modules` names, relative to the blueprint; None when unset."""
    named = loaded.agent.modules
    return loaded.home / named if named else None


#: The keys that moved table, by where they were: where they are now.
MOVED = {
    ("model", "name"): "[agent] model",
    ("model", "provider"): "[agent] provider",
    ("model", "from_checkpoint"): "[agent] from_checkpoint",
    ("model", "lora_rank"): "[recipe] lora_rank",
    ("model", "restore_optimizer"): "[recipe] restore_optimizer",
    ("rollout", "harness"): "[agent] harness",
    ("rollout", "kwargs"): "[agent] kwargs",
    ("rollout", "env"): "[agent] env",
    ("recipe", "modules"): "[agent] modules",
    ("wandb", "project"): "[logging.wandb] project",
    ("wandb", "entity"): "[logging.wandb] entity",
}


def _moved(raw: dict[str, Any]) -> list[str]:
    """A problem for each key the blueprint names where it used to live."""
    return [
        f"[{table}] {key}: now {where}"
        for (table, key), where in MOVED.items()
        if isinstance(raw.get(table), dict) and key in raw[table]
    ]


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
    if not loc:
        return message
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
