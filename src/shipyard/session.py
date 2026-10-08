"""The Tinker session a run opens: the service client, the capabilities read before a
session spends, and the close of a session that never became a trainer."""

from __future__ import annotations

import logging
from typing import Any

import tinker

logger = logging.getLogger(__name__)


def service_client(metadata: dict[str, str] | None = None) -> Any:
    """A `tinker.ServiceClient`, which reads `TINKER_API_KEY` and `TINKER_BASE_URL` from the
    environment. `metadata` becomes the session's `user_metadata`, naming the run on
    Tinker's side."""
    # The transport is the SDK's own. tinker 0.32 takes no `http_client`: `ServiceClient`
    # warns and drops unknown kwargs (lib/public_interfaces/service_client.py), and
    # `InternalClientHolder` builds every `AsyncTinker` from its own kwargs on its own
    # thread (lib/internal_client_holder.py). pyqwest is switched off only by the server's
    # `ClientConfigResponse.use_pyqwest_transport`.
    return tinker.ServiceClient(user_metadata=metadata)


async def server_has(service: Any, base_model: str) -> None:
    """Raise `ValueError` for a model the backend does not list or cannot train, before a
    session spends. Returns silently when the capabilities cannot be read."""
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
    """Finish a session that never became a trainer. A failure to close is logged, not
    raised, so it cannot hide the caller's own failure."""
    try:
        await service.close("errored", f"{type(why).__name__}: {why}")
    except Exception:  # noqa: BLE001 - the failure being reported is the caller's
        logger.warning("could not finish the Tinker session after a failed open", exc_info=True)
