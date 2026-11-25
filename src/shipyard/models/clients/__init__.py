from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from shipyard.models.clients.tinker import TINKER, TinkerClient
from shipyard.models.clients.together import TOGETHER, TogetherClient, plan

if TYPE_CHECKING:
    from shipyard.config import ModelTable

#: A model's client, on whichever provider hosts it.
Client = TinkerClient | TogetherClient

#: The client of each provider whose models the run reaches through its own client and its
#: proxy, and can train.
CLIENTS: dict[str, type[Client]] = {TINKER: TinkerClient, TOGETHER: TogetherClient}


def client_for(table: ModelTable, run: str, *, trains: bool = True) -> Client:
    """Return a model's client, by its provider: one that `trains` it, at its `lora_rank` and
    from its `seed`, or one that only serves it, with its `sampler_replicas` on Together, and on
    Tinker on the server its `base_url` names. Raise `ValueError` for a `from_checkpoint` the
    run cannot start from, and for an `api_key_env` that names a variable not set."""
    server = {"base_url": table.base_url, "api_key_env": table.api_key_env}
    own = {
        TOGETHER: {"sampler_replicas": table.sampler_replicas},
        TINKER: {key: value for key, value in server.items() if value},
    }
    return CLIENTS[table.provider](
        table.model,
        run=run,
        from_checkpoint=table.from_checkpoint,
        lora_rank=table.lora_rank if trains else None,
        restore_optimizer=table.restore_optimizer if trains else False,
        trains=trains,
        seed=table.seed if trains and isinstance(table.seed, int) else None,
        **own[table.provider],
    )


def place(clients: Iterable[Client]) -> None:
    """Place the clients that open together, the ones of one run or of several: on Together,
    those on the same base model share its model resources. A client placed already keeps its
    place; Tinker's serves every model always, so its clients have none to take."""
    plan([one for one in clients if isinstance(one, TogetherClient)])


async def forgo(clients: Iterable[Client]) -> None:
    """Give up the place of each client that will not open, so that no model resource on
    Together waits for it, billing. One that opened and closed has left its place already."""
    for one in clients:
        if isinstance(one, TogetherClient):
            await one.close()
