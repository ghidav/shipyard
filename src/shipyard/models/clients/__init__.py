from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from shipyard.models.clients.tinker import TINKER, TinkerClient
from shipyard.models.clients.together import TOGETHER, TogetherClient

if TYPE_CHECKING:
    from shipyard.config import Blueprint

#: The agent's client, on whichever provider hosts it.
Client = TinkerClient | TogetherClient

#: The client of each provider whose models the run reaches through its own client and its
#: proxy, and can train.
CLIENTS: dict[str, type[Client]] = {TINKER: TinkerClient, TOGETHER: TogetherClient}


def client_for(config: Blueprint, directory: Path, *, trains: bool = True) -> Client:
    """Return the agent's client, by `[agent] provider`: one that `trains` it, at
    `[recipe] lora_rank`, or one that only serves it. Raise `ValueError` for an
    `[agent] from_checkpoint` the run cannot start from."""
    agent, recipe = config.agent, config.recipe
    return CLIENTS[agent.provider](
        agent.model,
        run=Path(directory).name,
        from_checkpoint=agent.from_checkpoint,
        lora_rank=recipe.lora_rank if trains else None,
        restore_optimizer=recipe.restore_optimizer if trains else False,
        trains=trains,
    )
