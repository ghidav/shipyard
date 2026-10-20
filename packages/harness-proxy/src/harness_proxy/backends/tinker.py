from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

PREFIX = "tinker://"


async def connect(*, metadata: dict[str, str] | None, session: str | None) -> Any:
    """Return Tinker's service client, its session tagged with `metadata`."""
    import tinker

    return tinker.ServiceClient(user_metadata=metadata)


async def sampler(service: Any, base_model: str, weights: str | None) -> Any:
    """Return a sampling client on the `tinker://` weights, or on the base model for None."""
    if weights is None:
        return await service.create_sampling_client_async(base_model=base_model)
    return await service.create_sampling_client_async(model_path=weights)


async def context_of(service: Any, base_model: str) -> int | None:
    """Return the model's context length from `get_server_capabilities`; None on any error."""
    try:
        caps = await service.get_server_capabilities_async()
        for model in getattr(caps, "supported_models", None) or []:
            if getattr(model, "model_name", None) == base_model:
                return int(getattr(model, "max_context_length", 0) or 0) or None
    except Exception:  # noqa: BLE001 - a failed capability read must not fail a run
        logger.warning(
            "could not read the context length of %s: the budget and the context fit are "
            "not enforced",
            base_model,
            exc_info=True,
        )
    return None


async def close(service: Any) -> None:
    """Finish the proxy's Tinker session as a success: the proxy's own shutdown was clean."""
    await service.close("success")
