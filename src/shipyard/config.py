"""Blueprint schema: the keys `run.toml` accepts, and loading and validating the file."""

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
#: Recipe kinds. Each one except `evaluate` is a module under `shipyard.training.recipes`.
KINDS = ("dapo", "dr-grpo", "cispo", "opd", "gepa", "fst", "evaluate")
#: Minimum checkpoint TTL in hours, the SDK's floor: `TrainingClient.save_state` takes a TTL
#: "between 1 hour (3600) and 10 years" (tinker/lib/public_interfaces/training_client.py).
MIN_TTL_HOURS = 1.0
#: Default container image for the reflector's trial: a shell and Python, nothing more.
REFLECTION_IMAGE = "python:3.12-slim"


class ConfigError(ValueError):
    """Raised for an invalid blueprint. The message lists every problem, one per line."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


class _Table(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Agent(_Table):
    """The `[agent]` table: the model, who serves it and the checkpoint it starts from, the
    harness that drives it with its Harbor kwargs and env, and the modules it carries."""

    model: str
    provider: str = TINKER
    from_checkpoint: str | None = None
    harness: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    #: Modules directory, relative to the blueprint. Its modules are carried into every job,
    #: or seed the search for `gepa` and `fst`, which default it to `modules` when unset.
    modules: str | None = None

    @property
    def served(self) -> bool:
        """Whether this run serves the weights itself, which it does only for Tinker."""
        return self.provider == TINKER


class Data(_Table):
    """The `[data]` table: the datasets to train and test on, and how they are batched."""

    #: Dataset or datasets to train on. For `evaluate`, a lone `train` is read as `test`.
    train: str | list[str] | None = None
    #: Dataset or datasets to evaluate on. A run that sets them ends with an evaluation.
    test: str | list[str] | None = None
    batch_size: int = Field(ge=1)
    group_size: int = Field(default=1, ge=1)
    #: Attempts per test task. Defaults to `group_size`.
    test_group_size: int | None = Field(default=None, ge=1)
    seed: int = 0

    @model_validator(mode="before")
    @classmethod
    def _dataset(cls, value: Any) -> Any:
        """Accept `dataset` as the old name of `train`, and reject the removed `epochs` and
        `evaluate_every` keys, since a run is now counted in steps."""
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
        """Accept a dataset name or a list of names."""
        if (
            value is None
            or isinstance(value, str)
            or (isinstance(value, list) and all(isinstance(item, str) for item in value))
        ):
            return value
        raise ValueError("a dataset name or a list of them")


class Rollout(_Table):
    """The `[rollout]` table: how trials run and how the model reaches them. It covers the
    sandbox, time limits, sampling settings and the proxy."""

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
    #: Per-trial secrets. Maps a variable of the trial's environment to a variable of this
    #: process's environment that holds a pool of values, separated by commas or whitespace.
    #: Each trial in flight holds one value alone, so no more trials run at once than there
    #: are values. Several pools pair by position.
    secrets: dict[str, str] = Field(default_factory=dict)


#: Default optimizer steps per batch for `dapo` and `cispo`: DAPO 4.1 ("16 gradient updates
#: for each rollout step") and MiniMax-M1 3.1 ("16 rounds of off-policy updates per
#: generation batch").
PAPER_SUBSTEPS = 16
#: Default `cispo` `clip_high`, an epsilon that caps the importance weight at 4.0: ScaleRL
#: A.17.2 and FST appendix D.
PAPER_CISPO_HIGH = 3.0
#: Defaults for DAPO's soft overlong punishment (Eq. 13), which MiniMax-M1 3.1 also uses for
#: CISPO. The penalty ramps over the last L_cache / L_max of the token budget, 4,096 of
#: 20,480 tokens (4.1), reaching -1 at the limit where DAPO's rewards are -1 or 1 (Eq. 7). On
#: Harbor's 0 to 1 scale that -1 is -0.5, which gives the same advantages once divided by the
#: spread.
PAPER_OVERLONG_BUFFER = 0.2
PAPER_OVERLONG_PENALTY = 0.5
#: Default cap on what the `dr-grpo` length rule takes from a solved answer, so that a solved
#: answer never scores below a failure scored 0.5 or less.
LENGTH_CAP = 0.5


class Gradient(_Table):
    """Settings shared by the gradient recipes (`dapo`, `dr-grpo`, `cispo`, `opd` and `fst`).
    Each recipe adds its own clipping and reward-shaping settings.

    The default of one substep follows FST (one mini-batch per step) and Dr. GRPO, which
    states no other."""

    learning_rate: float = Field(gt=0)
    #: Number of training steps. Each step draws its batches from the task generator.
    steps: int = Field(ge=1)
    #: Evaluate after every this many steps; 0 evaluates only at the end.
    evaluate_every: int = Field(default=0, ge=0)
    substeps: int = Field(default=1, ge=1)
    #: Steps over which the learning rate rises to `learning_rate`. Off by default, because
    #: the papers' warm-ups (DAPO's 20, FST's 10) are sized for runs of hundreds of steps.
    warmup: int = Field(default=0, ge=0)
    reference: Literal["trainer", "sampler"] = "trainer"
    kl_coef: float = Field(default=0.0, ge=0)
    #: LoRA rank of the trained adapter. Tinker's default when unset.
    lora_rank: int | None = None
    #: With `[agent] from_checkpoint`, also restore the checkpoint's optimizer state.
    restore_optimizer: bool = False
    #: Weight of the teacher's term in each token's advantage, `distill × (teacher − reference)`.
    #: 0 is off.
    distill: float = Field(default=0.0, ge=0)
    #: Model whose logprobs score the policy's tokens: a base model the backend serves, or a
    #: `tinker://` sampler path. It must share the policy's tokenizer.
    teacher: str | None = Field(default=None, validate_default=True)

    @field_validator("teacher")
    @classmethod
    def _taught(cls, value: str | None, info: Any) -> str | None:
        """Require a teacher exactly when `distill` weighs one, and a sampler path for a
        checkpoint."""
        distill = info.data.get("distill", 0.0)
        if value is None and distill > 0:
            raise ValueError(f"distill = {distill} weighs a teacher's logprobs; name the teacher")
        if value is not None and not distill:
            raise ValueError("set distill above 0 to weigh the teacher, or leave it unset")
        if value is not None and value.startswith("tinker://") and "/sampler_weights/" not in value:
            raise ValueError(
                "a teacher is sampled, so a checkpoint is named by its sampler_path "
                "(tinker://.../sampler_weights/...)"
            )
        return value


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


class OpdRecipe(Gradient):
    """The `opd` recipe: on-policy distillation. The teacher's logprobs are the only signal;
    rewards are measured but not trained on."""

    kind: Literal["opd"]
    distill: float = Field(default=1.0, gt=0)
    teacher: str


#: Default modules directory a search seeds from when `[agent] modules` is unset.
SEARCH_MODULES = "modules"
#: Default minibatch size. GEPA 4.3: "All GEPA optimization runs use a minibatch size of 3".
PAPER_MINIBATCH = 3
#: Default patience: none, so the budget alone ends the search. This matches GEPA's released
#: code, whose `NoImprovementStopper` (gepa 0.1.4, utils/stop_condition.py) runs only when
#: passed in `stop_callbacks`, and `api.py` passes none of its own.
GEPA_PATIENCE = None


class Search(_Table):
    """Settings shared by the search recipes, `gepa` and `fst`.

    They set the reflector (harness, model and image), the minibatch size each new candidate
    is judged on, the rollout budget, the number of rounds without a better Pareto score
    before the search stops, and how the reflector edits. The search seeds from
    `[agent] modules`."""

    reflection_harness: str
    reflection_model: str | None = None
    reflection_image: str = REFLECTION_IMAGE
    minibatch: int = Field(default=PAPER_MINIBATCH, ge=1)
    budget: int | None = Field(default=None, ge=1)
    patience: int | None = Field(default=GEPA_PATIENCE, ge=1)
    edits: Literal["rewrite", "incremental"] = "rewrite"


class GepaRecipe(Search):
    """The `gepa` recipe: the search alone, without training the weights.

    `pareto` chooses the tasks every kept candidate is scored on: a dataset name, a number
    of the run's tasks to hold out, or 0 to select on the tasks the search reflects on.
    Unset, it holds out two thirds of the run's tasks (see `data.search_sets`)."""

    kind: Literal["gepa"]
    pareto: int | str | None = None

    @field_validator("pareto", mode="before")
    @classmethod
    def _held_out(cls, value: Any) -> Any:
        """Accept a dataset name or a count of 0 or more; reject booleans, fractions and
        blank names."""
        if value is None or (isinstance(value, str) and value.strip()):
            return value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        raise ValueError("a dataset name, or how many of the run's tasks to hold out (0 or more)")


#: The recipe-specific settings `fst` accepts for each `slow` recipe. Setting one that
#: belongs to another recipe is an error.
SLOW_KNOBS = {
    "dapo": ("clip_low", "clip_high"),
    "dr-grpo": ("clip", "length_penalty", "length_floor", "length_cap"),
    "cispo": ("clip_high",),
}


class FstRecipe(Search, Gradient):
    """The `fst` recipe (fast-slow training): a gradient recipe, `slow`, trains the weights,
    a search evolves a population of texts, and `cycle` sets how the two interleave.
    Defaults follow the FST paper."""

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


Recipe = (
    DapoRecipe | DrGrpoRecipe | CispoRecipe | OpdRecipe | GepaRecipe | FstRecipe | EvaluateRecipe
)


class Checkpoints(_Table):
    every: int = Field(default=1, ge=1)
    ttl_hours: float = Field(default=168.0, ge=MIN_TTL_HOURS)


class Wandb(_Table):
    """The `[logging.wandb]` table: the wandb project the trainer also logs to."""

    project: str
    entity: str | None = None


class Logging(_Table):
    """The `[logging]` table: destinations the trainer writes to besides the run directory,
    one sub-table each."""

    wandb: Wandb | None = None


class Blueprint(_Table):
    """A validated `run.toml`. Use `load` to read one from disk."""

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
        """Apply defaults that span tables and require the datasets the recipe needs.

        For `evaluate`, a lone `train` is read as `test`, and `test` is required. A search
        defaults `[agent] modules` to `modules`. Every other recipe requires `train`."""
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
        """Names of the training datasets."""
        return _names(self.data.train)

    @property
    def test_sets(self) -> list[str]:
        return _names(self.data.test)

    @property
    def raw(self) -> dict[str, Any]:
        """The TOML as parsed, before defaults are applied. The run keeps a verbatim copy of
        the file."""
        return self._raw

    @property
    def trains(self) -> bool:
        """Whether the recipe trains the weights: the gradient recipes and `fst` do."""
        return isinstance(self.recipe, Gradient)

    @property
    def home(self) -> Path:
        """Directory the blueprint was loaded from. `[agent] modules` is relative to it."""
        return self._home


def _names(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def config_file(path: Path) -> Path:
    """Return the `run.toml` for a blueprint path: the path itself if it is a file, else the
    `run.toml` inside it."""
    path = Path(path)
    return path if path.is_file() else path / CONFIG


def load(path: Path) -> Blueprint:
    """Load and validate the blueprint at `path`, a `run.toml` or a directory holding one.

    Raises `ConfigError` listing every problem found."""
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
    """Return the directory `[agent] modules` names, resolved against the blueprint's
    directory, or None when unset."""
    named = loaded.agent.modules
    return loaded.home / named if named else None


#: Keys that moved to another table: each old (table, key) mapped to its new location.
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
    """Return a problem for each key the blueprint sets at its old location."""
    return [
        f"[{table}] {key}: now {where}"
        for (table, key), where in MOVED.items()
        if isinstance(raw.get(table), dict) and key in raw[table]
    ]


def _problem(error: dict[str, Any]) -> str:
    """Format a pydantic validation error as `[table] key: problem`, in the blueprint's
    terms rather than pydantic's."""
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
