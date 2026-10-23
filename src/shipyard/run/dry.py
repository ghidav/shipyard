"""A dry run, for `shipyard run --dry`: the checks a run makes before it opens anything, and the
value it takes for each key it decides elsewhere, such as an unset LoRA rank or timeout. Nothing
opens, and nothing is left behind. The only calls out are each model's backend saying the model's
context length, which its lane reads the same way as it opens."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.models.task.config import TaskConfig as TaskToml

from shipyard.config import Config, Logging, Resolved, Said
from shipyard.jobs.harnesses import Harness
from shipyard.model import Model
from shipyard.models.backends import Session
from shipyard.models.clients.tinker import lora_rank
from shipyard.trainer import Trainer

#: The logger of the proxy's backends, which warns, with the error, of a context it cannot read.
BACKENDS = "harness_proxy.backends"


@dataclass
class Checked:
    """What a dry run found: the value the run takes for each key it decides elsewhere, by the
    key's path, and what it could not read that the run needs as it opens, which would stop it."""

    values: Resolved = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def checked(made: type[Trainer], config: Config) -> Checked:
    """Make the trainer in a scratch directory and run every check the run makes before it opens
    anything, raising what it would; then find the value the run takes for each key it decides
    elsewhere. Wandb would start a run as the trainer is made, so the check makes it without,
    once wandb is found."""
    if config.logging.wandb is not None and importlib.util.find_spec("wandb") is None:
        raise ValueError("[logging.wandb] needs wandb: install shipyard[wandb]")
    quiet = config.model_copy(update={"logging": Logging()})
    with tempfile.TemporaryDirectory(prefix="shipyard-dry-") as scratch:
        with made(quiet, Path(scratch) / "dry") as trainer:
            trainer.check()  # the backends it builds are bound to the models, and never opened
            tasks = [*trainer.generator.train, *trainer.generator.test]
            found = asyncio.run(models(trainer.models.all))
            found.values.update(rollout(config, tasks))
            return found


async def models(found: Sequence[Model]) -> Checked:
    """The values each model's table leaves to the run: the LoRA rank of one the recipe trains,
    the renderer its lane falls back on, and its context, which its backend says. A context its
    backend cannot say is a warning, with the backend's reason: the run stops without it."""
    values: Resolved = {}
    warnings: list[str] = []
    reads: dict[tuple[str, ...], tuple[Model, Any]] = {}
    for one in found:
        session, table, at = one.backend, one.table, ("models", one.role)
        if not isinstance(session, Session):
            continue  # a provider's API: it has no adapter, renderer or lane
        if table.lora_rank is None and one.training:
            values[(*at, "lora_rank")] = (
                Said("the training state's own") if table.from_checkpoint else lora_rank(None)
            )
        if not table.renderers:
            name = recommended(one.name)
            values[(*at, "renderers")] = [name] if name else Said("the model's own")
        if not table.max_context:
            reads[(*at, "max_context")] = (one, session.client.context_length())
    with _kept(BACKENDS) as records:
        pending = [read for _, read in reads.values()]
        lengths = await asyncio.gather(*pending, return_exceptions=True)
    # The backend logs why it could not read a context, naming the model.
    why = {str(one.args[0]): one.exc_info[1] for one in records if one.args and one.exc_info}
    for (at, (one, _)), length in zip(reads.items(), lengths, strict=True):
        if isinstance(length, int) and length:
            values[at] = length
            continue
        failed = length if isinstance(length, BaseException) else why.get(one.name)
        reason = f": {type(failed).__name__}: {failed}" if failed is not None else ""
        warnings.append(
            f"[models.{one.role}] max_context: {one.provider} did not say the context of "
            f"{one.name}{reason}"
        )
    return Checked(values, warnings)


def rollout(config: Config, tasks: Sequence[Path]) -> Resolved:
    """The values `[rollout]` leaves to Harbor: each task's own agent timeout, and each
    harness's setup timeout."""
    values: Resolved = {}
    if config.rollout.timeout is None:
        values[("rollout", "timeout")] = timeout(tasks)
    if config.rollout.setup_timeout is None:
        given = {one.bare: setup_timeout(one) for one in harnesses(config.recipe)}
        seconds = {one for one in given.values() if one is not None}
        if len(seconds) == 1:
            values[("rollout", "setup_timeout")] = seconds.pop()
        elif seconds:
            values[("rollout", "setup_timeout")] = Said(
                ", ".join(f"{name}'s {one:g} s" for name, one in given.items() if one)
            )
    return values


def recommended(model: str) -> str | None:
    """The renderer the cookbook recommends for `model`, which a lane with none uses; None for
    a model the cookbook does not know."""
    from tinker_cookbook.model_info import get_recommended_renderer_name

    try:
        return get_recommended_renderer_name(model)
    except Exception:  # noqa: BLE001 - a model the cookbook does not know
        return None


def timeout(tasks: Sequence[Path]) -> float | Said | None:
    """The agent timeout Harbor takes from the tasks' own task.toml: their one value, None for no
    limit, or, when they differ, what they range over."""
    found: list[float] = []
    unlimited = 0
    for task in tasks:
        try:
            config = TaskToml.model_validate_toml((task / "task.toml").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a task Harbor would refuse says so itself
            continue
        if config.agent.timeout_sec is None:
            unlimited += 1
        else:
            found.append(float(config.agent.timeout_sec))
    if not found and not unlimited:
        return Said("each task's own")
    if not found:
        return None
    low, high = min(found), max(found)
    if low == high and not unlimited:
        return low
    said = [f"{low:g} s" if low == high else f"{low:g} to {high:g} s"]
    if unlimited:
        said.append(f"no limit for {unlimited} task{'s' if unlimited > 1 else ''}")
    return Said(f"each task's own: {', '.join(said)}")


def harnesses(recipe: Any) -> Iterator[Harness]:
    """The harnesses the recipe's table names, in its keys and in their lists and tables."""
    for name in type(recipe).model_fields:
        value = getattr(recipe, name)
        if isinstance(value, dict):
            value = list(value.values())
        for one in value if isinstance(value, list | tuple) else [value]:
            if isinstance(one, Harness):
                yield one


def setup_timeout(harness: Harness) -> float | None:
    """The setup timeout Harbor gives `harness` when the run sets none: its class's own."""
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig

    try:
        made = AgentFactory.get_agent_class_from_config(AgentConfig(name=harness.bare))
    except Exception:  # noqa: BLE001 - a harness Harbor does not know is refused as it runs
        return None
    return getattr(made, "DEFAULT_SETUP_TIMEOUT_SEC", None)


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _kept(name: str) -> Iterator[list[logging.LogRecord]]:
    """Keep the warnings of a logger and its children, which the dry run then says, rather than
    print them."""
    logger, handler = logging.getLogger(name), _Records()
    propagates = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagates
