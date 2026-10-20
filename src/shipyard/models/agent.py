from __future__ import annotations

import asyncio
from collections.abc import Sequence as Values
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import tinker

from harness_proxy.wire import model_input
from shipyard.groups import Sequence
from shipyard.models.clients import Client, client_for
from shipyard.models.endpoints import ProviderEndpoint, ProxyEndpoint
from shipyard.run.costs import UNKNOWN, Spend

if TYPE_CHECKING:
    from shipyard.config import Blueprint

#: The agent's weights as the run began: the base model, or `[agent] from_checkpoint`, taken
#: before a resumed run loads its state. `Policy(INITIAL)` is them, frozen.
INITIAL = "initial"
#: Most scoring requests sent to one sampling client at once.
LOGPROB_CONCURRENCY = 16


class Frozen:
    """A model frozen in the agent's Tinker session: `INITIAL`, another model, or a checkpoint."""

    def __init__(self, load: str, model: str | None, spend: Spend) -> None:
        #: What it loads: `INITIAL`, a model's name, or a `tinker://` checkpoint.
        self.load = load
        #: The model its tokens are priced as; for a checkpoint or `INITIAL`, its base model,
        #: known once the session opens it.
        self.model = model
        self.provider: str | None = None
        self.spend = spend
        self.sampler: Any = None
        #: It is never served to a harness, and has no rollout budget.
        self.endpoint: Any = None
        self.budget: int | None = None

    def opened(self, sampler: Any, provider: str, model: str) -> None:
        """Take the sampling client the session opened, and who bills its tokens as what."""
        self.sampler, self.provider, self.model = sampler, provider, model

    async def sample(self, prompts: Values[str], **params: Any) -> list[str]:
        raise RuntimeError("a frozen policy has no endpoint to sample through")

    async def score(self, sequences: Values[Sequence]) -> list[list[float]]:
        """Return the logprob at each target of each sequence, through the sampling client."""
        if self.sampler is None:
            raise RuntimeError(
                "this policy cannot score yet: a frozen policy opens with the agent, when "
                "`train` starts"
            )
        found, tokens = await sampled(self.sampler, sequences)
        self.spend(self.provider or UNKNOWN, self.model or UNKNOWN, **tokens)
        return found


class Provider:
    """A provider's model, asked through LiteLLM. It cannot score."""

    def __init__(self, model: str, provider: str, spend: Spend) -> None:
        self.model, self.provider, self.spend = model, provider, spend
        #: What a harness calls when this model is the agent, and what `sample` asks.
        self.endpoint = ProviderEndpoint(model, provider)
        #: The provider's limits apply, not a budget of the run's.
        self.budget: int | None = None

    async def sample(self, prompts: Values[str], **params: Any) -> list[str]:
        """Return the model's reply to each prompt. `params` go to the provider as they are.
        Each call is billed as it ends; if any failed, the first failure is raised once every
        call has ended."""

        async def ask(prompt: str) -> str:
            reply, tokens = await self.endpoint.sample(
                [{"role": "user", "content": prompt}], **params
            )
            self.spend(self.provider, self.model, **tokens)
            return reply

        replies: list[str] = []
        for one in await asyncio.gather(*(ask(one) for one in prompts), return_exceptions=True):
            if isinstance(one, BaseException):
                raise one
            replies.append(one)
        return replies

    async def score(self, sequences: Values[Sequence]) -> list[list[float]]:
        raise RuntimeError(
            f"this policy cannot score: {self.provider}'s model gives no logprobs of a sequence"
        )


class Session:
    """The agent on Tinker or Together: its training client and the proxy its harnesses
    call."""

    def __init__(
        self,
        config: Blueprint,
        directory: Path,
        spend: Spend,
        *,
        weights: str | None = None,
    ) -> None:
        self.client: Client = client_for(config, directory)
        #: The proxy the agent's harnesses call.
        self.endpoint = ProxyEndpoint(config, directory)
        self.spend = spend
        #: The frozen models that open in this session.
        self.frozen: list[Frozen] = []
        #: The training state the session opens on, from the checkpoint a run resumes from;
        #: None opens the initial weights.
        self.weights = weights

    @property
    def provider(self) -> str:
        return self.client.provider

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def budget(self) -> int | None:
        """Most tokens one rollout may sample, as the proxy reports it once open."""
        return self.endpoint.budget

    async def __aenter__(self) -> Session:
        """Open the training client, the frozen models, a resumed state and the proxy, in order."""
        try:
            await self.client.open()
            begun: Any = None
            for one in self.frozen:
                if one.load == INITIAL:
                    begun = begun if begun is not None else await self.client.anchor()
                    one.opened(begun, self.provider, self.model)
                    continue
                sampler = await self.client.sampler(one.load)
                model = one.model or await sampler.get_base_model_async()
                one.opened(sampler, self.provider, model)
            if self.weights is not None:
                await self.client.load_state(self.weights)
            self.endpoint.backend = await self.client.serving()
            await self.endpoint.open()
            await self.serve("start")
        except BaseException as failed:
            await self.__aexit__(type(failed), failed, failed.__traceback__)
            raise
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        failed: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the proxy, and finish the Tinker session with how the run ended."""
        self.endpoint.close()
        if failed is None:
            await self.client.close("success")
            return
        status = "errored" if isinstance(failed, Exception) else "interrupted"
        await self.client.close(status, f"{type(failed).__name__}: {failed}")

    async def sample(self, prompts: Values[str], **params: Any) -> list[str]:
        raise NotImplementedError(
            "the agent on Tinker is sampled by its harness, through `rollout`, not from code"
        )

    async def score(self, sequences: Values[Sequence]) -> list[list[float]]:
        """Return the agent's logprob at each target of each sequence, from its training client."""
        # The trainer reads every token but the last, and predicts the next.
        self.count(train=sum(len(one.tokens) - 1 for one in sequences))
        return await self.client.compute_logprobs(sequences)

    async def serve(self, tag: str) -> str:
        """Publish the agent's weights as `tag`, point the proxy at them, and return their path."""
        path = await self.client.publish(tag)
        await self.endpoint.serve(path)
        return path

    async def save(
        self, tag: str, *, ttl_seconds: int | None, sampler: bool
    ) -> tuple[str, str | None]:
        """Save the agent's training state as `tag`, and with `sampler` its weights for sampling."""
        if not sampler:
            return await self.client.save_state(tag, ttl_seconds=ttl_seconds), None
        state, weights = await asyncio.gather(
            self.client.save_state(tag, ttl_seconds=ttl_seconds),
            self.client.save_weights_for_sampler(tag, ttl_seconds=ttl_seconds),
        )
        return state, weights

    def count(self, **tokens: int) -> None:
        """File tokens the agent spent, by kind, in the run's costs."""
        self.spend(self.provider, self.model, **tokens)


async def sampled(
    sampler: Any, sequences: Values[Sequence]
) -> tuple[list[list[float]], dict[str, int]]:
    """Return a sampling client's logprobs at each sequence's targets, and the tokens it took."""
    gate = asyncio.Semaphore(LOGPROB_CONCURRENCY)
    took = {"prefill": 0, "cache": 0, "sample": 0}

    async def one(sequence: Sequence) -> list[float]:
        prompt = model_input(sequence.tokens, sequence.images)
        async with gate:
            reply = await sampler.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=1),
                include_prompt_logprobs=True,
            )
        cache = min(int(reply.prompt_cache_hit_tokens or 0), prompt.length)
        took["prefill"] += prompt.length - cache
        took["cache"] += cache
        took["sample"] += sum(len(drawn.tokens) for drawn in reply.sequences)
        scored = list(reply.prompt_logprobs or [])
        return [float(scored[at] or 0.0) for at in sequence.targets]

    found = list(await asyncio.gather(*(one(sequence) for sequence in sequences)))
    return found, took
