from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
)

from shipyard.run import files

#: Minimum checkpoint TTL in hours, the SDK's floor: `TrainingClient.save_state` takes a TTL
#: "between 1 hour (3600) and 10 years" (tinker/lib/public_interfaces/training_client.py).
MIN_TTL_HOURS = 1.0
#: What a role may be named: it names the model's lane in the proxy's addresses.
ROLE = re.compile(r"[A-Za-z0-9_-]+")
#: A key TOML reads unquoted; `shown` quotes any other.
BARE = re.compile(r"[A-Za-z0-9_-]+")
#: Where each key a config written before a model was a table of its own now goes.
MOVED = {
    ("agent",): "[models.<role>] tables, one per model, and its harness to a key of the recipe",
    ("data", "group_size"): "a key of the recipe's own, under [recipe]",
    ("data", "test_group_size"): "a key of the recipe's own, under [recipe]",
    **{
        ("rollout", key): "a key of each [models.<role>]"
        for key in ("temperature", "top_p", "top_k", "max_tokens", "max_context", "fill_context")
    },
    ("rollout", "renderer"): "[models.<role>] renderers",
    **{
        ("recipe", key): "a key of each [models.<role>] the recipe trains"
        for key in ("learning_rate", "lora_rank", "restore_optimizer")
    },
}


class ConfigError(ValueError):
    """Raised for an invalid config. The message lists every problem, one per line."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


class _Table(BaseModel):
    """A table of the config, which refuses keys it does not know."""

    model_config = ConfigDict(extra="forbid")


class ModelTable(_Table):
    """A `[models.<role>]` table: a model the run reaches, and how its lane in the proxy samples
    it. The role is the table's name: what the recipe reads it by, and what the model's lane,
    spend and saves are filed under."""

    model: str
    #: Who serves the model: `tinker` or `together`, which the run can train, or another
    #: provider, such as `anthropic`, whose API it only calls.
    provider: str = "tinker"
    #: The checkpoint the model starts from: on Tinker, a training state (`.../weights/...`)
    #: for a model the recipe trains, weights saved for sampling (`.../sampler_weights/...`)
    #: for one it does not.
    from_checkpoint: str | None = None
    #: The cookbook renderers its lane holds, such as reasoning efforts
    #: (`qwen3_8_low_reasoning`); none for the model's own. A rollout picks one of them.
    renderers: list[str] = Field(default_factory=list)
    #: The rate `update` trains it at unless the recipe gives another. A model the recipe
    #: trains needs it.
    learning_rate: float | None = Field(default=None, gt=0)
    #: The LoRA rank of a new adapter; unset, 32 (`DEFAULT_LORA_RANK`). A model that goes on
    #: from a training state keeps its rank.
    lora_rank: int | None = Field(default=None, ge=1)
    #: With `from_checkpoint`, also restore its optimizer state.
    restore_optimizer: bool = False
    #: How many replicas of its sampler answer its calls, for a model on Together: each a copy
    #: of it on GPUs of its own, so that more answer a crowded rollout faster, and bill more.
    sampler_replicas: int = Field(default=1, ge=1)
    #: Most tokens one trial may sample across its calls; 0 for the model's context.
    max_context: int = Field(default=0, ge=0)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = Field(default=8192, ge=1)
    #: Give a call that would overflow the context whatever room is left, and turn the
    #: harness's own compaction off.
    fill_context: bool = False

    @field_validator("renderers", mode="before")
    @classmethod
    def _renderers(cls, value: Any) -> Any:
        """Accept a renderer or a list of them, each named once."""
        found = ([value] if value else []) if isinstance(value, str) else value
        if not isinstance(found, list) or not all(
            isinstance(one, str) and one.strip() for one in found
        ):
            raise ValueError("a renderer or a list of them")
        if len(set(found)) < len(found):
            raise ValueError("each renderer is named once")
        return found

    @property
    def sampling(self) -> dict[str, Any]:
        """The settings its lane samples every call with, over the harness's own."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "fill_context": self.fill_context,
        }


class Data(_Table):
    """The `[data]` table: the datasets to train and test on, and how they are batched."""

    #: Dataset or datasets to train on.
    train: str | list[str]
    #: Dataset or datasets to evaluate on, with the recipe's `evaluate`. A run that sets them
    #: ends with an evaluation.
    test: str | list[str] | None = None
    batch_size: int = Field(ge=1)
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
    """The `[rollout]` table: how trials run, and where the proxy the models are served in is."""

    sandbox: str = "docker"
    concurrency: int = Field(default=4, ge=1)
    timeout: float | None = None
    setup_timeout: float | None = None
    endpoint_url: str = ""
    host: str = "host.docker.internal"
    bind: str = "0.0.0.0"
    bind_port: int = Field(default=0, ge=0, le=65535)
    jobs_dir: str = "jobs"
    #: Per-trial secrets. Maps a variable of the trial's environment to a variable of this
    #: process's environment that holds a pool of values, separated by commas or whitespace.
    #: Each trial in flight holds one value alone, so no more trials run at once than there
    #: are values. Several pools pair by position.
    secrets: dict[str, str] = Field(default_factory=dict)


class Recipe(_Table):
    """The `[recipe]` table: the keys every run reads; a trainer declares its own in a subclass,
    such as the harnesses it runs and its group size."""

    steps: int = Field(ge=1)
    #: Evaluate on `[data] test` every this many steps from the start, the model the run starts
    #: from included: each evaluation runs beside the next step's update, which serves its new
    #: weights once it has finished. The last step is evaluated after it; 0 evaluates only then.
    evaluate_every: int = Field(default=0, ge=0)
    #: Grade a trial Harbor cut at its timeout by the verifier's reward, as any other. By
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


class Config[R: Recipe](_Table):
    """A validated `run.toml`, its `[recipe]` read as a trainer's `Recipe`."""

    #: Modules directory, relative to the config, which the run's modules start as.
    modules: str | None = None
    #: The models the run reaches, by role.
    models: dict[str, ModelTable]
    data: Data
    rollout: Rollout = Field(default_factory=Rollout)
    recipe: R
    checkpoints: Checkpoints = Field(default_factory=Checkpoints)
    logging: Logging = Field(default_factory=Logging)
    _home: Path = PrivateAttr(default_factory=Path)

    @field_validator("models")
    @classmethod
    def _roles(cls, value: dict[str, ModelTable]) -> dict[str, ModelTable]:
        """Refuse no model, and a role a lane's address could not hold."""
        if not value:
            raise ValueError("name a model, as [models.<role>]")
        bad = sorted(one for one in value if not ROLE.fullmatch(one))
        if bad:
            raise ValueError(
                f"{', '.join(bad)}: a role is letters, digits, '_' and '-', as it names the "
                "model's lane"
            )
        return value

    @property
    def datasets(self) -> list[str]:
        return _names(self.data.train)

    @property
    def test_sets(self) -> list[str]:
        return _names(self.data.test)

    @property
    def home(self) -> Path:
        """Directory the config was loaded from."""
        return self._home


def _names(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


class Said(str):
    """A value the run takes that TOML cannot hold as one value, such as each task's own
    timeout; `shown` writes it as a comment."""


#: The value the run takes for a key it decides elsewhere, such as an unset LoRA rank, by the
#: key's path in the config, as `("models", "agent", "lora_rank")`.
Resolved = dict[tuple[str, ...], Any]


def shown(config: Config, values: Resolved | None = None) -> str:
    """Return `config` as TOML, as the run reads it: every key of every table, defaults
    included, in the order the tables declare them, with the value the run takes, from
    `values` where it decides it elsewhere. A key with no value, or one TOML cannot hold, is a
    comment."""
    out: list[str] = []
    _shown(config, (), out, values or {})
    return "\n".join(out).strip() + "\n"


def _shown(table: BaseModel, path: tuple[str, ...], out: list[str], values: Resolved) -> None:
    keys: list[str] = []
    tables: list[tuple[tuple[str, ...], BaseModel]] = []
    for name in type(table).model_fields:
        value = getattr(table, name)
        if isinstance(value, _Table):
            tables.append(((*path, name), value))
        elif value and isinstance(value, dict) and all(_is_table(v) for v in value.values()):
            tables += [((*path, name, key), one) for key, one in value.items()]
        else:
            value = values.get((*path, name), value)
            if value is None or isinstance(value, Said):  # TOML has no null
                said = "none" if value is None else value
                keys.append(f"# {_key(name)} = {said}")
            else:
                keys.append(f"{_key(name)} = {_value(value)}")
    if path:
        out.append(f"[{'.'.join(_key(one) for one in path)}]")
    out.extend([*keys, ""] if keys or path else [])
    for at, one in tables:
        _shown(one, at, out, values)


def _is_table(value: Any) -> bool:
    return isinstance(value, _Table)


def _key(name: str) -> str:
    return name if BARE.fullmatch(name) else _value(name)


def _value(value: Any) -> str:
    """Write a value as TOML: a table inline, and `None` in one left out, as TOML has no null."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        found = [
            f"{_key(str(key))} = {_value(one)}" for key, one in value.items() if one is not None
        ]
        return "{ " + ", ".join(found) + " }" if found else "{}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(one) for one in value) + "]"
    if isinstance(value, Path):
        value = str(value)
    return tomli_w.dumps({"value": value}).removeprefix("value = ").rstrip("\n")


def config_file(path: Path) -> Path:
    """Return the `run.toml` for a config path: the file itself, or the one inside it."""
    path = Path(path)
    return path if path.is_file() else path / files.CONFIG


def load(
    path: Path, recipe: type[Recipe] = Recipe, overrides: Mapping[str, str] | None = None
) -> Config:
    """Load and validate the config at `path`, raising `ConfigError` with every problem."""
    file = config_file(path)
    raw = parsed(file, overrides)
    try:
        loaded = Config[recipe].model_validate(raw)  # type: ignore[valid-type]
    except ValidationError as error:
        raise ConfigError([_problem(one) for one in error.errors()]) from error
    loaded._home = file.parent
    return loaded


def parsed(path: Path, overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return the config at `path` as TOML, each `table.key` of `overrides` set in it."""
    file = config_file(path)
    try:
        raw = tomllib.loads(file.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError([f"cannot read {file}: {error.strerror or error}"]) from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError([f"{file.name} is not readable TOML: {error}"]) from error
    for name, text in (overrides or {}).items():
        *tables, key = name.split(".")
        table = raw
        for one in tables:
            table = table.setdefault(one, {})
            if not isinstance(table, dict):
                raise ConfigError([f"--{name}: {one} is a key, not a table"])
        table[key] = value_of(text)
    return raw


def value_of(text: str) -> Any:
    """Read an override as a TOML value, such as `2`, `true` or `["a", "b"]`; else as text."""
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def _problem(error: Any) -> str:
    """Say a validation error in the config's terms: `[table] key: problem`."""
    loc = [str(part) for part in error["loc"]]
    message = str(error["msg"]).removeprefix("Value error, ")
    if error["type"] == "missing":
        message = "field required" if len(loc) > 1 else "table required"
    elif error["type"] == "extra_forbidden":
        message = "unknown key" if len(loc) > 1 else "unknown table"
        moved = MOVED.get(tuple(loc))
        if moved:
            message += f": it moved, to {moved}"
    if not loc:
        return message  # a check across tables, which names its own
    return f"[{'.'.join(loc[:-1]) or loc[0]}]{' ' + loc[-1] if len(loc) > 1 else ''}: {message}"
