"""The Tinker session a run opens: the service client, the capabilities read before a
session spends, and the close of a session that never became a trainer."""

from __future__ import annotations

import logging
from typing import Any

import tinker

logger = logging.getLogger(__name__)


def service_client(metadata: dict[str, str] | None = None) -> Any:
    """`tinker.ServiceClient`, which reads `TINKER_API_KEY` and `TINKER_BASE_URL` itself;
    `metadata` is the session's `user_metadata`, naming the run on Tinker's side."""
    # The transport is the SDK's own. tinker 0.32 takes no `http_client`: `ServiceClient`
    # warns and drops unknown kwargs (lib/public_interfaces/service_client.py), and
    # `InternalClientHolder` builds every `AsyncTinker` from its own kwargs on its own
    # thread (lib/internal_client_holder.py); pyqwest is switched off only by the server's
    # `ClientConfigResponse.use_pyqwest_transport`. The old `tinker_client` module's httpx
    # keepalive client (shipyard at 151b0b2) therefore has no way in, and is not ported.
    return tinker.ServiceClient(user_metadata=metadata)


async def server_has(service: Any, base_model: str) -> None:
    """Refuse a model the backend does not list or cannot train before a session spends;
    quiet when the capabilities cannot be read, which is no reason to refuse a run."""
    try:
        caps = await service.get_server_capabilities_async()
        models = list(caps.supported_models or [])
    except Exception:  # noqa: BLE001 - a courtesy read, never worth failing a run over
        logger.debug("could not read the server capabilities", exc_info=True)
        return
    if not models:
        return
    found = next((one for one in models if one.model_name == base_model), None)
    if found is None:
        names = sorted(str(one.model_name) for one in models)
        raise ValueError(f"the backend does not serve {base_model!r}; it serves {', '.join(names)}")
    if found.trainable is False:
        raise ValueError(
            f"{base_model!r} can be sampled but not trained; name the trainable variant "
            "under [model] name"
        )


async def quietly_closed(service: Any, why: BaseException) -> None:
    """Finish a session that never became a trainer; never the reason the caller sees."""
    try:
        await service.close("errored", f"{type(why).__name__}: {why}")
    except Exception:  # noqa: BLE001 - the failure being reported is the caller's
        logger.warning("could not finish the Tinker session after a failed open", exc_info=True)
