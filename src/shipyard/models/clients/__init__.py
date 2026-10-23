from __future__ import annotations

from typing import TYPE_CHECKING

from shipyard.models.clients.tinker import TINKER, TinkerClient
from shipyard.models.clients.together import TOGETHER, TogetherClient

if TYPE_CHECKING:
    from shipyard.config import ModelTable

#: A model's client, on whichever provider hosts it.
Client = TinkerClient | TogetherClient

#: The client of each provider whose models the run reaches through its own client and its
#: proxy, and can train.
CLIENTS: dict[str, type[Client]] = {TINKER: TinkerClient, TOGETHER: TogetherClient}


def client_for(table: ModelTable, run: str, *, trains: bool = True) -> Client:
    """Return a model's client, by its provider: one that `trains` it, at its `lora_rank`, or
    one that only serves it, with its `sampler_replicas` on Together. Raise `ValueError` for a
    `from_checkpoint` the run cannot start from."""
    return CLIENTS[table.provider](
        table.model,
        run=run,
        from_checkpoint=table.from_checkpoint,
        lora_rank=table.lora_rank if trains else None,
        restore_optimizer=table.restore_optimizer if trains else False,
        trains=trains,
        **({"sampler_replicas": table.sampler_replicas} if table.provider == TOGETHER else {}),
    )
