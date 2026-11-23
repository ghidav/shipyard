from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

PREFIX = "tinker://"


async def connect(
    *,
    metadata: dict[str, str] | None,
    session: str | None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> Any:
    """Return Tinker's service client, its session tagged with `metadata`: on the server at
    `base_url`, with the key the variable `api_key_env` holds, or on the one the environment
    names. `ValueError` if that variable is not set."""
    import tinker

    return tinker.ServiceClient(user_metadata=metadata, **reaching(base_url, api_key_env))


def reaching(base_url: str | None, api_key_env: str | None) -> dict[str, str]:
    """Return what `tinker.ServiceClient` takes to reach the server at `base_url` with the key
    the variable `api_key_env` holds; nothing for a None, which leaves it to the environment
    (TINKER_BASE_URL, TINKER_API_KEY). `ValueError` if the variable is not set: the SDK would
    send that server TINKER_API_KEY, Tinker's own key."""
    found: dict[str, str] = {}
    if base_url:
        found["base_url"] = base_url
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"{api_key_env} is not set: it holds the key of the Tinker server "
                f"{base_url or 'the run reaches'}"
            )
        found["api_key"] = key
    return found


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
    except Exception:  # noqa: BLE001 - the lane refuses, one way for any cause
        logger.warning(
            "could not read the context length of %s",
            base_model,
            exc_info=True,
        )
    return None


async def close(service: Any) -> None:
    """Finish the proxy's Tinker session as a success: the proxy's own shutdown was clean."""
    await service.close("success")
