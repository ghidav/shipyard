from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from shipyard.models.clients.tinker import TINKER, TinkerClient
from shipyard.models.clients.together import TOGETHER, TogetherClient

if TYPE_CHECKING:
    from shipyard.config import Blueprint

#: The providers whose models the run trains, each through its client here.
TRAINED = frozenset({TINKER, TOGETHER})

#: The agent's training client, on whichever provider trains it.
Client = TinkerClient | TogetherClient


def client_for(config: Blueprint, directory: Path) -> Client:
    """Return the training client for the run's agent, by `[agent] provider`."""
    agent, recipe = config.agent, config.recipe
    run = Path(directory).name
    if agent.provider == TOGETHER:
        return TogetherClient(
            agent.model,
            run=run,
            from_checkpoint=agent.from_checkpoint,
            lora_rank=recipe.lora_rank,
            restore_optimizer=recipe.restore_optimizer,
        )
    return TinkerClient(
        agent.model,
        run=run,
        from_checkpoint=agent.from_checkpoint,
        lora_rank=recipe.lora_rank,
        restore_optimizer=recipe.restore_optimizer,
        anchor_ttl=round(config.checkpoints.ttl_hours * 3600),
    )
