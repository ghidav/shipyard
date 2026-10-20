from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from collections.abc import Sequence as Values
from typing import TYPE_CHECKING, Any

import tinker
from tinker import TensorData

from harness_proxy.backends.together import PREFIX, wire_input
from shipyard.models.clients.tinker import DEFAULT_LORA_RANK, shifted

if TYPE_CHECKING:
    from shipyard.groups import Sequence

logger = logging.getLogger(__name__)

#: The provider of a model Together post-training serves, which the run trains.
TOGETHER = "together"
#: Tinker's loss names, as Together's loss type and the field its settings go in.
LOSSES: dict[str, tuple[str, str | None]] = {
    "importance_sampling": ("LOSS_TYPE_IMPORTANCE_SAMPLING", None),
    "ppo": ("LOSS_TYPE_PPO", "ppo_params"),
    "cispo": ("LOSS_TYPE_CISPO", "cispo_params"),
    "dro": ("LOSS_TYPE_DRO", "dro_params"),
    "cross_entropy": ("LOSS_TYPE_CROSS_ENTROPY", "cross_entropy_params"),
}
#: Seconds the run waits for GPUs before giving up, as Together's own Tinker client does.
PROVISION_SECONDS = 3600.0

#: Provisions a model resource for a base model, and returns it once it is ready.
Provision = Callable[[str], Awaitable[Any]]


class TogetherClient:
    """The agent's Together session, on a model resource provisioned on `open`."""

    def __init__(
        self,
        model: str,
        *,
        run: str | None = None,
        from_checkpoint: str | None = None,
        lora_rank: int | None = None,
        restore_optimizer: bool = False,
        provision: Provision | None = None,
    ) -> None:
        self.model = model
        self.provider = TOGETHER
        self.run = run
        self.from_checkpoint = from_checkpoint
        self.lora_rank = lora_rank
        self.restore_optimizer = restore_optimizer
        self.resources: Any = None
        #: The training session, opened when first needed: a resume opens it on its checkpoint.
        self.session: Any = None
        #: Whether an optimizer step moved the weights since they were last published.
        self.moved = False
        self._provision = provision if provision is not None else provisioned
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """Provision a model resource for the model, with a generator the proxy samples."""
        self.resources = await self._provision(self.model)

    async def serving(self) -> dict[str, str]:
        """Return what the proxy samples this session through: the backend and the session."""
        session = await self._session()
        return {"backend": TOGETHER, "session": str(session.session_id)}

    async def anchor(self) -> Any:
        raise NotImplementedError(FROZEN)

    async def sampler(self, model: str) -> Any:
        raise NotImplementedError(FROZEN)

    async def forward_backward(
        self, datums: Values[tinker.Datum], loss_fn: str, loss_config: Mapping[str, float] | None
    ) -> Awaitable[list[list[float]]]:
        """Run one training pass, and return what awaits its logprobs, once the pass is done."""
        session = await self._session()
        result = await self._one(
            session.trainer.forward_backward_async(
                samples=[sample(one) for one in datums],
                loss=loss(loss_fn, loss_config),
                return_loss_fn_outputs=True,
            )
        )
        return _ready([_floats(row.tensors["logprobs"].data) for row in result.loss_fn_outputs])

    async def optim_step(self, adam: tinker.AdamParams, rate: float) -> Awaitable[None]:
        """Run an optimizer step with `adam`'s settings at learning rate `rate`."""
        session = await self._session()
        await self._one(session.trainer.optim_step_async(adam_params=adam_params(adam, rate)))
        self.moved = True
        return _ready(None)

    async def compute_logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
        """Return each sequence's logprobs at its targets, from one forward pass."""
        samples = []
        for one in sequences:
            read, target_tokens = shifted(one.tokens, one.images)
            ones = TensorData(data=[1.0] * read.length, dtype="float32")
            inputs = {"target_tokens": tensor(target_tokens), "weights": tensor(ones)}
            samples.append({"model_input": wire_input(read), "loss_fn_inputs": inputs})
        session = await self._session()
        result = await self._one(
            session.trainer.forward_async(
                samples=samples,
                loss={"type": "LOSS_TYPE_CROSS_ENTROPY"},
                return_loss_fn_outputs=True,
            )
        )
        out: list[list[float]] = []
        for one, row in zip(sequences, result.loss_fn_outputs, strict=True):
            scored = _floats(row.tensors["logprobs"].data)
            out.append([scored[at - 1] if at > 0 else 0.0 for at in one.targets])
        return out

    async def publish(self, tag: str) -> str:
        """Point the generator at the weights the updates moved to, and return their name."""
        session = await self._session()
        if self.moved:
            await self._one(session.trainer.weights_sync_async())
            self.moved = False
        return f"{PREFIX}{session.session_id}/{tag}"

    async def save_weights_for_sampler(self, name: str, *, ttl_seconds: int | None) -> str:
        """Save an inference checkpoint, which Together keeps and can deploy; return its id."""
        session = await self._session()
        return str((await self._one(session.create_inference_checkpoint_async())).id)

    async def save_state(self, name: str, *, ttl_seconds: int | None) -> str:
        """Save a training checkpoint, which Together keeps until deleted; return its id."""
        session = await self._session()
        return str((await self._one(session.create_training_checkpoint_async())).id)

    async def load_state(self, path: str) -> None:
        """Go on from training checkpoint `path` in a new session on the same resource."""
        if self.session is not None:
            await self.session.stop_async()
        self.session = await self.resources.create_session_async(
            display_name=self.run, resume_from_checkpoint_id=path, load_optimizer=True
        )

    async def close(self, status: str = "success", detail: str | None = None) -> None:
        """Stop the session and the GPU resource, which bills until it stops; then do nothing."""
        held = (("session", self.session), ("model resource", self.resources))
        self.session = self.resources = None
        for what, one in held:
            if one is None:
                continue
            try:
                await one.stop_async()
            except Exception:  # noqa: BLE001 - a resource that will not stop is not a run failure
                logger.warning(
                    "could not stop the Together %s: it bills until it stops on its own after "
                    "an hour idle",
                    what,
                    exc_info=True,
                )

    async def _session(self) -> Any:
        """Return the session, opened on first use: a new adapter, or `from_checkpoint`'s."""
        if self.session is None:
            if self.from_checkpoint:
                self.session = await self.resources.create_session_async(
                    display_name=self.run,
                    resume_from_checkpoint_id=self.from_checkpoint,
                    load_optimizer=self.restore_optimizer,
                )
            else:
                rank = DEFAULT_LORA_RANK if self.lora_rank is None else int(self.lora_rank)
                self.session = await self.resources.create_session_async(
                    display_name=self.run, lora_config={"rank": rank}
                )
        return self.session

    async def _one[T](self, work: Awaitable[T]) -> T:
        """Await one training operation, after the one before it."""
        async with self._lock:
            return await work


#: Why a frozen policy cannot open next to a Together agent.
FROZEN = (
    "a Together session samples only its own weights, so a frozen policy, such as INITIAL or a "
    "teacher, needs an agent on Tinker"
)


async def provisioned(model: str) -> Any:
    """Provision a LoRA resource for `model` with one generator; ValueError if Together lacks it."""
    from together import AsyncTogether
    from together.post_training import ModelResourcesClient

    listed = (await AsyncTogether().post_training.supported_models.get()).data
    found = next((one for one in listed if one.base_model == model), None)
    if found is None:
        names = ", ".join(sorted(one.base_model for one in listed))
        raise ValueError(f"[agent] model: Together does not train {model}; it trains {names}")
    return await ModelResourcesClient.create_async(
        base_model=model,
        compute_config={"num_generator_replicas": 1},
        timeout=PROVISION_SECONDS,
    )


def sample(datum: tinker.Datum) -> dict[str, Any]:
    """Return a Tinker datum as Together's training sample, its mask left out as on Tinker."""
    inputs = datum.loss_fn_inputs
    return {
        "model_input": wire_input(datum.model_input),
        "loss_fn_inputs": {key: tensor(value) for key, value in inputs.items() if key != "mask"},
    }


def tensor(value: TensorData) -> dict[str, Any]:
    """Return a Tinker tensor as Together's: its flat data and its dtype."""
    return {"data": list(value.data), "dtype": value.dtype}


def loss(name: str, config: Mapping[str, float] | None) -> dict[str, Any]:
    """Return a Tinker loss and its settings as Together's loss config."""
    if name not in LOSSES:
        raise ValueError(f"no Together loss for {name!r}: it takes {', '.join(sorted(LOSSES))}")
    kind, field = LOSSES[name]
    found: dict[str, Any] = {"type": kind}
    if config:
        if field is None:
            raise ValueError(f"the {name} loss takes no settings")
        found[field] = dict(config)
    return found


def adam_params(adam: tinker.AdamParams, rate: float) -> dict[str, float]:
    """Return Tinker's Adam settings at `rate` as Together's, gradient clipping always sent."""
    return {
        "learning_rate": rate,
        "beta1": adam.beta1,
        "beta2": adam.beta2,
        "eps": adam.eps,
        "weight_decay": adam.weight_decay,
        "grad_clip_norm": adam.grad_clip_norm,
    }


def _floats(values: Any) -> list[float]:
    return [float(one) for one in values]


async def _ready[T](value: T) -> T:
    return value
