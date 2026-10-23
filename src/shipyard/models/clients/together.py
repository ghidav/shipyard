from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping
from collections.abc import Sequence as Values
from typing import TYPE_CHECKING, Any

import tinker
from tinker import TensorData

from harness_proxy.backends.together import PREFIX, wire_input
from shipyard.models.clients.tinker import lora_rank, sent, shifted

if TYPE_CHECKING:
    from shipyard.groups import Sequence

logger = logging.getLogger(__name__)

#: The provider of a model Together post-training serves, which the run trains.
TOGETHER = "together"
#: Each of Tinker's `LOSSES` as Together's loss type, and the field its settings go in.
LOSS_TYPES: dict[str, tuple[str, str | None]] = {
    "importance_sampling": ("LOSS_TYPE_IMPORTANCE_SAMPLING", None),
    "ppo": ("LOSS_TYPE_PPO", "ppo_params"),
    "cispo": ("LOSS_TYPE_CISPO", "cispo_params"),
    "dro": ("LOSS_TYPE_DRO", "dro_params"),
    "cross_entropy": ("LOSS_TYPE_CROSS_ENTROPY", None),
}
#: Seconds the run waits for GPUs before giving up, as Together's own Tinker client does.
PROVISION_SECONDS = 3600.0
#: Seconds the run waits for one training operation: no limit, as Together's own Tinker client
#: waits, where its SDK gives up after five minutes, which an update over long trajectories
#: outlasts.
OPERATION_SECONDS: float | None = None

#: Provisions a model resource for a base model with that many sampler replicas, and returns it
#: once it is ready.
Provision = Callable[[str, int], Awaitable[Any]]


class TogetherClient:
    """A model's Together session, on a model resource provisioned on `open`. It opens the same
    whether the recipe trains the model or not: the proxy samples it through the session."""

    #: The backend it reaches, as a job's row names it.
    kind = TOGETHER
    #: Whether what it saves lives only the TTL it is given: Together keeps its checkpoints
    #: until they are deleted.
    expires = False

    def __init__(
        self,
        model: str,
        *,
        run: str | None = None,
        from_checkpoint: str | None = None,
        lora_rank: int | None = None,
        restore_optimizer: bool = False,
        trains: bool = True,
        sampler_replicas: int = 1,
        provision: Provision | None = None,
    ) -> None:
        self.model = model
        #: Who its tokens are billed to.
        self.provider = TOGETHER
        #: Whether the run trains the model; its session opens either way.
        self.trains = trains
        self.run = run
        self.from_checkpoint = from_checkpoint
        self.lora_rank = lora_rank
        self.restore_optimizer = restore_optimizer
        #: How many replicas of the sampler its resource holds: Together's generator replicas.
        self.sampler_replicas = sampler_replicas
        self.resources: Any = None
        #: The training session, opened when first needed: a resume opens it on its checkpoint.
        self.session: Any = None
        #: Whether an optimizer step moved the weights since they were last published.
        self.moved = False
        self._provision = provision if provision is not None else provisioned
        self._lock = asyncio.Lock()

    async def context_length(self) -> int | None:
        """Return the model's context length as Together's list of models says it, read as the
        proxy's lane reads it when it opens, which provisions nothing; None when it does not
        say. The read needs no session: the list is the account's."""
        from harness_proxy.backends.together import context_of

        return await context_of(None, self.model)

    async def open(self) -> None:
        """Provision a model resource for the model, with the sampler replicas the proxy samples,
        and have the interpreter stop it when it exits, if the run has not."""
        self.resources = await self._provision(self.model, self.sampler_replicas)
        hold(self)

    async def serving(self) -> dict[str, str]:
        """Return what the proxy samples this session through: the backend and the session."""
        session = await self._session()
        return {"backend": TOGETHER, "session": str(session.session_id)}

    async def fit(
        self,
        parts: Values[Values[tinker.Datum]],
        adam: tinker.AdamParams,
        rate: float,
        loss_fn: str,
        loss_config: Mapping[str, float] | None,
    ) -> list[list[float]]:
        """Run a training pass and an optimizer step on each part in turn, at learning rate
        `rate`, and return each datum's logprobs; Together runs one operation at a time."""
        session = await self._session()
        chosen, params = loss(loss_fn, loss_config), adam_params(adam, rate)
        trained: list[list[float]] = []
        for part in parts:
            result = await self._one(
                session.trainer.forward_backward_async(
                    samples=[sample(one, loss_fn) for one in part],
                    loss=chosen,
                    return_loss_fn_outputs=True,
                    timeout=OPERATION_SECONDS,
                )
            )
            trained += [_floats(row.tensors["logprobs"].data) for row in result.loss_fn_outputs]
            await self._one(
                session.trainer.optim_step_async(adam_params=params, timeout=OPERATION_SECONDS)
            )
            self.moved = True
        return trained

    async def compute_logprobs(
        self, sequences: Values[Sequence]
    ) -> tuple[list[list[float]], dict[str, int]]:
        """Return each sequence's logprobs at its targets, from one forward pass, and the tokens
        that pass read."""
        samples, read_tokens = [], 0
        for one in sequences:
            read, target_tokens = shifted(one.tokens, one.images)
            read_tokens += int(read.length)
            ones = TensorData(data=[1.0] * read.length, dtype="float32")
            inputs = {"target_tokens": tensor(target_tokens), "weights": tensor(ones)}
            samples.append({"model_input": wire_input(read), "loss_fn_inputs": inputs})
        session = await self._session()
        result = await self._one(
            session.trainer.forward_async(
                samples=samples,
                loss={"type": "LOSS_TYPE_CROSS_ENTROPY"},
                return_loss_fn_outputs=True,
                timeout=OPERATION_SECONDS,
            )
        )
        out: list[list[float]] = []
        for one, row in zip(sequences, result.loss_fn_outputs, strict=True):
            scored = _floats(row.tensors["logprobs"].data)
            out.append([scored[at - 1] if at > 0 else 0.0 for at in one.targets])
        return out, {"train": read_tokens}

    async def publish(self, tag: str) -> str:
        """Point the generator at the weights the updates moved to, and return their name."""
        session = await self._session()
        if self.moved:
            await self._one(session.trainer.weights_sync_async(timeout=OPERATION_SECONDS))
            self.moved = False
        return f"{PREFIX}{session.session_id}/{tag}"

    async def save_weights_for_sampler(self, name: str, *, ttl_seconds: int | None) -> str:
        """Save an inference checkpoint, which Together keeps and can deploy; return its id."""
        session = await self._session()
        saving = session.create_inference_checkpoint_async(timeout=OPERATION_SECONDS)
        return str((await self._one(saving)).id)

    async def save_state(self, name: str, *, ttl_seconds: int | None) -> str:
        """Save a training checkpoint, which Together keeps until deleted; return its id."""
        session = await self._session()
        saving = session.create_training_checkpoint_async(timeout=OPERATION_SECONDS)
        return str((await self._one(saving)).id)

    async def load_state(self, path: str) -> None:
        """Go on from training checkpoint `path` in a new session on the same resource."""
        if self.session is not None:
            await self.session.stop_async()
        self.session = await self.resources.create_session_async(
            display_name=self.run, resume_from_checkpoint_id=path, load_optimizer=True
        )

    async def close(self, status: str = "success", detail: str | None = None) -> None:
        """Stop the session and the GPU resource, which bills until it stops: the resource by
        force, which Together otherwise refuses while a session still runs an operation, as
        when the run fails in an update. One that will not stop stays held, and the
        interpreter tries it again when it exits; once both have stopped, do nothing."""
        for name, what in (("session", "session"), ("resources", "model resource")):
            one = getattr(self, name)
            if one is None:
                continue
            try:
                await (one.stop_async(force=True) if name == "resources" else one.stop_async())
            except Exception:  # noqa: BLE001 - a resource that will not stop is not a run failure
                logger.warning(
                    "could not stop the Together %s: the interpreter tries again when it exits",
                    what,
                    exc_info=True,
                )
                continue
            setattr(self, name, None)
        if self.session is None and self.resources is None:
            RUNNING.discard(self)

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
                self.session = await self.resources.create_session_async(
                    display_name=self.run, lora_config={"rank": lora_rank(self.lora_rank)}
                )
        return self.session

    async def _one[T](self, work: Awaitable[T]) -> T:
        """Await one training operation, after the one before it."""
        async with self._lock:
            return await work


#: Together clients whose session or GPU resource may still run: the interpreter stops each when
#: it exits, if the run has not, as a crash or a second interrupt can leave them billing.
RUNNING: set[TogetherClient] = set()
#: Whether the interpreter's exit stops `RUNNING`.
_hooked = False


def hold(client: TogetherClient) -> None:
    """Hold `client` until it closes: the interpreter stops it when it exits, and SIGTERM, when
    nothing handles it, exits the interpreter so that it does."""
    global _hooked
    RUNNING.add(client)
    if not _hooked:
        at_exit(stop_running)
        _hooked = True
    exit_on_sigterm()


def at_exit(stop: Callable[[], None]) -> None:
    """Run `stop` as the interpreter exits, before concurrent.futures shuts its executors down:
    an ordinary atexit callback runs after, when the HTTP client a stop needs can no longer
    resolve a host. Together's own Tinker client does the same."""
    threading._register_atexit(stop)  # type: ignore[attr-defined]


def stop_running() -> None:
    """Stop the session and GPU resource of every Together client the run left open."""
    for client in list(RUNNING):
        try:
            asyncio.run(client.close("interrupted", "the process exited before the run closed it"))
        except Exception:  # noqa: BLE001 - the interpreter is exiting; say so and go on
            logger.warning("could not stop a Together client as the process exited", exc_info=True)


def exit_on_sigterm() -> None:
    """Make SIGTERM exit the interpreter, so its exit stops the GPUs, unless something already
    handles it: `shipyard run` does, by unwinding the run, which closes every client."""
    if threading.current_thread() is not threading.main_thread():
        return
    if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        return
    signal.signal(signal.SIGTERM, _exit)


def _exit(signum: int, frame: Any) -> None:
    sys.exit(128 + signum)


async def provisioned(model: str, sampler_replicas: int = 1) -> Any:
    """Provision a LoRA resource for `model` with `sampler_replicas` generators; ValueError if
    Together lacks it."""
    from together import AsyncTogether
    from together.post_training import ModelResourcesClient

    listed = (await AsyncTogether().post_training.supported_models.get()).data
    found = next((one for one in listed if one.base_model == model), None)
    if found is None:
        names = ", ".join(sorted(one.base_model for one in listed))
        raise ValueError(f"Together does not train {model}; it trains {names}")
    return await ModelResourcesClient.create_async(
        base_model=model,
        compute_config={"num_generator_replicas": sampler_replicas},
        timeout=PROVISION_SECONDS,
    )


def sample(datum: tinker.Datum, loss: str) -> dict[str, Any]:
    """Return a Tinker datum as Together's training sample for `loss`, sent as on Tinker."""
    read = sent(datum, loss)
    return {
        "model_input": wire_input(read.model_input),
        "loss_fn_inputs": {key: tensor(value) for key, value in read.loss_fn_inputs.items()},
    }


def tensor(value: TensorData) -> dict[str, Any]:
    """Return a Tinker tensor as Together's: its flat data and its dtype."""
    return {"data": list(value.data), "dtype": value.dtype}


def loss(name: str, config: Mapping[str, float] | None) -> dict[str, Any]:
    """Return a Tinker loss and its settings, as `check_loss` passed them, as Together's."""
    kind, field = LOSS_TYPES[name]
    return {"type": kind, **({field: dict(config)} if config and field else {})}


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
