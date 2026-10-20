from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from shipyard.run import record

#: Minimum checkpoint TTL in hours, the SDK's floor: `TrainingClient.save_state` takes a TTL
#: "between 1 hour (3600) and 10 years" (tinker/lib/public_interfaces/training_client.py).
MIN_TTL_HOURS = 1.0


class ConfigError(ValueError):
    """Raised for an invalid blueprint. The message lists every problem, one per line."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


class _Table(BaseModel):
    """A table of the blueprint, which refuses keys it does not know."""

    model_config = ConfigDict(extra="forbid")


class Agent(_Table):
    """The `[agent]` table: the run's own policy, and the harnesses that drive it."""

    model: str
    #: Who serves the model: `tinker` or `together`, which the run trains, or a provider such
    #: as `anthropic`.
    provider: str = "tinker"
    from_checkpoint: str | None = None
    #: The harness, or the harnesses, that drive it: Harbor agents, each with an optional
    #: version, such as `pi@0.85.1`. Read as a list.
    harness: list[str]
    #: With several harnesses, split each task's attempts across them, so a group mixes them;
    #: otherwise their tasks are split, in turn, so each group is one harness's.
    mix_harnesses: bool = False
    kwargs: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    #: Modules directory, relative to the blueprint, carried into every job.
    modules: str | None = None

    @field_validator("harness", mode="before")
    @classmethod
    def _harnesses(cls, value: Any) -> Any:
        """Accept a harness or a list of them, each named once."""
        found = [value] if isinstance(value, str) else value
        if not isinstance(found, list) or not found:
            raise ValueError("a harness or a list of them")
        if not all(isinstance(one, str) and one.strip() for one in found):
            raise ValueError("a harness or a list of them")
        if len(set(found)) < len(found):
            raise ValueError("each harness is named once")
        return found


class Data(_Table):
    """The `[data]` table: the datasets to train and test on, and how they are batched."""

    #: Dataset or datasets to train on.
    train: str | list[str]
    #: Dataset or datasets to evaluate on. A run that sets them ends with an evaluation.
    test: str | list[str] | None = None
    batch_size: int = Field(ge=1)
    group_size: int = Field(default=1, ge=1)
    #: Attempts per test task. Defaults to `group_size`.
    test_group_size: int | None = Field(default=None, ge=1)
    seed: int = 0

    @field_validator("train", "test", mode="before")
    @classmethod
    def _named(cls, value: Any) -> Any:
        """Accept a dataset name or a list of names, and say so for anything else."""
        if (
            value is None
            or isinstance(value, str)
            or (isinstance(value, list) and all(isinstance(item, str) for item in value))
        ):
            return value
        raise ValueError("a dataset name or a list of them")


class Rollout(_Table):
    """The `[rollout]` table: how trials run and how the model reaches them."""

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
    #: The cookbook renderer, or the renderers, the proxy renders the agent's calls with; none
    #: for the model's own. Several, such as reasoning efforts (`qwen3_8_low_reasoning`,
    #: `qwen3_8_xhigh_reasoning`), take a batch's tasks in turn, so each group is one
    #: renderer's, and the test set is evaluated at each. Read as a list.
    renderer: list[str] = Field(default_factory=list)
    fill_context: bool = False
    jobs_dir: str = "jobs"
    #: Per-trial secrets. Maps a variable of the trial's environment to a variable of this
    #: process's environment that holds a pool of values, separated by commas or whitespace.
    #: Each trial in flight holds one value alone, so no more trials run at once than there
    #: are values. Several pools pair by position.
    secrets: dict[str, str] = Field(default_factory=dict)

    @field_validator("renderer", mode="before")
    @classmethod
    def _renderers(cls, value: Any) -> Any:
        """Accept a renderer or a list of them, each named once; "" is the model's own."""
        found = ([value] if value else []) if isinstance(value, str) else value
        if not isinstance(found, list) or not all(
            isinstance(one, str) and one.strip() for one in found
        ):
            raise ValueError("a renderer or a list of them")
        if any("/" in one for one in found):
            raise ValueError("a renderer is named in its trials' addresses, so it holds no '/'")
        if len(set(found)) < len(found):
            raise ValueError("each renderer is named once")
        return found


class Recipe(_Table):
    """The `[recipe]` table: the keys every run reads; a trainer declares its own in a subclass."""

    steps: int = Field(ge=1)
    #: The rate `update` trains at, warmed up over `warmup` updates. A recipe that never
    #: updates, such as a search over the modules, leaves it out; `update` refuses without it.
    learning_rate: float | None = Field(default=None, gt=0)
    #: Parts each update is split into, by prompt: one optimizer step each.
    substeps: int = Field(default=1, ge=1)
    #: Updates over which the learning rate warms up linearly.
    warmup: int = Field(default=0, ge=0)
    #: Evaluate on `[data] test` every this many steps; 0 evaluates only at the end.
    evaluate_every: int = Field(default=0, ge=0)
    lora_rank: int | None = Field(default=None, ge=1)
    #: With `[agent] from_checkpoint`, also restore its optimizer state.
    restore_optimizer: bool = False
    #: Grade a rollout Harbor cut at its timeout by the verifier's reward, as any other. By
    #: default it is masked, as DAPO filters answers cut at their length: its transcript ends
    #: where the clock did.
    grade_timeouts: bool = False


class Checkpoints(_Table):
    """The `[checkpoints]` table: how often the run saves, and how long a save is kept."""

    #: Save the run's state every this many steps, and at the end.
    every: int = Field(default=1, ge=1)
    ttl_hours: float = Field(default=168.0, ge=MIN_TTL_HOURS)


class Wandb(_Table):
    """The `[logging.wandb]` table: the wandb project the trainer also logs to."""

    project: str
    entity: str | None = None


class Logging(_Table):
    """The `[logging]` table: where the trainer writes besides the run directory."""

    wandb: Wandb | None = None


class Blueprint[R: Recipe](_Table):
    """A validated `run.toml`, its `[recipe]` read as a trainer's `Recipe`."""

    agent: Agent
    data: Data
    rollout: Rollout = Field(default_factory=Rollout)
    recipe: R
    checkpoints: Checkpoints = Field(default_factory=Checkpoints)
    logging: Logging = Field(default_factory=Logging)
    _home: Path = PrivateAttr(default_factory=Path)

    @model_validator(mode="after")
    def _shares(self) -> Self:
        """Refuse groups too small to give each harness a share, when they are mixed."""
        count = len(self.agent.harness)
        if self.agent.mix_harnesses and count > 1:
            for key in ("group_size", "test_group_size"):
                size = getattr(self.data, key)
                if size is not None and size < count:
                    raise ValueError(
                        f"[data] {key}: {size} is fewer than the {count} harnesses "
                        "[agent] mix_harnesses splits it across"
                    )
        return self

    @property
    def datasets(self) -> list[str]:
        return _names(self.data.train)

    @property
    def test_sets(self) -> list[str]:
        return _names(self.data.test)

    @property
    def home(self) -> Path:
        """Directory the blueprint was loaded from."""
        return self._home


def _names(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def config_file(path: Path) -> Path:
    """Return the `run.toml` for a blueprint path: the file itself, or the one inside it."""
    path = Path(path)
    return path if path.is_file() else path / record.CONFIG


def load(path: Path, recipe: type[Recipe] = Recipe) -> Blueprint:
    """Load and validate the blueprint at `path`, raising `ConfigError` with every problem."""
    file = config_file(path)
    try:
        raw = tomllib.loads(file.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError([f"cannot read {file}: {error.strerror or error}"]) from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError([f"{file.name} is not readable TOML: {error}"]) from error
    try:
        blueprint = Blueprint[recipe].model_validate(raw)  # type: ignore[valid-type]
    except ValidationError as error:
        raise ConfigError([_problem(one) for one in error.errors()]) from error
    blueprint._home = file.parent
    return blueprint


def _problem(error: Any) -> str:
    """Say a validation error in the blueprint's terms: `[table] key: problem`."""
    loc = [str(part) for part in error["loc"]]
    message = str(error["msg"]).removeprefix("Value error, ")
    if error["type"] == "missing":
        message = "field required" if len(loc) > 1 else "table required"
    elif error["type"] == "extra_forbidden":
        message = "unknown key" if len(loc) > 1 else "unknown table"
    if not loc:
        return message  # a check across tables, which names its own
    return f"[{'.'.join(loc[:-1]) or loc[0]}]{' ' + loc[-1] if len(loc) > 1 else ''}: {message}"
