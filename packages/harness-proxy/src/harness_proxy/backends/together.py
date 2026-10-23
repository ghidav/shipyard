from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from harness_proxy.chat import Refused

logger = logging.getLogger(__name__)

PREFIX = "together://"
#: Together's stop reasons, as Tinker names them.
STOPS = {"STOP_REASON_STOP": "stop", "STOP_REASON_LENGTH": "length"}


async def connect(*, metadata: dict[str, str] | None, session: str | None) -> Any:
    """Attach to the run's training session, whose generator the proxy samples."""
    if not session:
        raise ValueError(
            "the together backend samples from the run's own session: pass its id as `session`"
        )
    from together.post_training import SessionClient

    return await SessionClient.attach_async(session_id=session)


async def sampler(service: Any, base_model: str, weights: str | None) -> Sampler:
    """Return a client on the session's generator, which serves the weights published last."""
    return Sampler(service.generator)


async def context_of(service: Any, base_model: str) -> int | None:
    """Return the model's context length from Together's supported models; None on any error."""
    try:
        from together import AsyncTogether

        listed = await AsyncTogether().post_training.supported_models.get()
        for model in listed.data:
            if model.base_model == base_model:
                configs = [one.generator_config for one in model.compute_configs or ()]
                lengths = [one.context_length for one in configs if one is not None]
                return max(lengths, default=None)
    except Exception:  # noqa: BLE001 - the lane refuses, one way for any cause
        logger.warning(
            "could not read the context length of %s",
            base_model,
            exc_info=True,
        )
    return None


async def close(service: Any) -> None:
    """Let the session go and leave it running: the run that opened it stops it."""
    await service.detach_async()


@dataclass
class Sampler:
    """A sampling client on a Together generator, answering as Tinker's does."""

    generator: Any

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        try:
            given = wire_input(prompt)
        except ValueError as text_only:
            raise Refused(str(text_only)) from None
        result = await self.generator.sample_async(given, num_samples, wire_params(sampling_params))
        return response(result)


def wire_input(model_input: Any) -> dict[str, Any]:
    """Return a Tinker `ModelInput` as Together's; ValueError for an image: it takes text only."""
    chunks = []
    for chunk in model_input.chunks:
        if getattr(chunk, "type", "encoded_text") != "encoded_text":
            raise ValueError("Together takes text only, and this input holds an image")
        chunks.append({"encoded_text": {"tokens": [int(token) for token in chunk.tokens]}})
    return {"chunks": chunks}


def wire_params(params: Any) -> dict[str, Any]:
    """Return Tinker `SamplingParams` as Together's, dropping token-id stops with a warning."""
    found: dict[str, Any] = {
        "temperature": params.temperature,
        "top_k": params.top_k,
        "top_p": params.top_p,
    }
    if params.max_tokens is not None:
        found["max_tokens"] = params.max_tokens
    if params.seed is not None:
        found["seed"] = params.seed
    stops = [params.stop] if isinstance(params.stop, str) else list(params.stop or ())
    strings = [one for one in stops if isinstance(one, str)]
    if dropped := tuple(int(one) for one in stops if not isinstance(one, str)):
        _dropped(dropped)
    if strings:
        found["stop"] = strings
    return found


def response(result: Any) -> Any:
    """Return a Together `SampleResult` as Tinker's `SampleResponse`."""
    import tinker

    sequences = [
        tinker.types.SampledSequence(
            stop_reason=STOPS.get(str(one.stop_reason), "stop"),
            sequence_id=str(at),
            tokens_np=np.asarray([int(token) for token in one.tokens], dtype=np.int32),
            logprobs_np=_floats(one.logprobs),
        )
        for at, one in enumerate(result.sequences)
    ]
    cached = int(result.sequences[0].prompt_cache_hit_tokens or 0) if result.sequences else 0
    return tinker.types.SampleResponse(sequences=sequences, prompt_cache_hit_tokens=cached)


def _floats(values: Sequence[float] | None) -> np.ndarray | None:
    return None if values is None else np.asarray(values, dtype=np.float32)


@cache
def _dropped(stops: tuple[int, ...]) -> None:
    logger.warning(
        "Together stops on strings only, so the token-id stops %s are left to the model's "
        "end-of-sequence token",
        list(stops),
    )
