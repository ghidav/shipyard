from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping
from collections.abc import Sequence as Values
from functools import partial
from typing import TYPE_CHECKING, Any

import tinker
from tinker import TensorData

from harness_proxy.backends.together import PREFIX, wire_input
from shipyard.models.clients.tinker import lora_rank, sent, shifted

if TYPE_CHECKING:
    import httpx

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
#: Seconds between looks at the session while an operation runs, which fails once Together has
#: stopped the session: it stops one, recording no error, when the project's balance runs out,
#: and an operation on a stopped session never ends (27 Oct: a run waited 6.5 h on one).
SESSION_CHECK_SECONDS = 60.0
#: A session's statuses once it has ended or is ending, as Together's SDK counts them.
ENDED = frozenset(
    {
        "TRAINING_SESSION_STATUS_STOPPING",
        "TRAINING_SESSION_STATUS_STOPPED",
        "TRAINING_SESSION_STATUS_ERROR",
        "TRAINING_SESSION_STATUS_EXPIRED",
    }
)
#: How many times an operation whose data lost its connection as it uploaded is sent again, and
#: the seconds before each. Together's SDK uploads a large operation's data from this machine
#: before it sends the operation, and its own five retries come within seconds, which a congested
#: uplink outlasts (24 Oct: every upload of an update's second training pass dropped, and the run
#: failed).
UPLOAD_RETRIES = 3
UPLOAD_RETRY_SECONDS = 60.0
#: How a publish syncs the generator: in the background, so the trainer goes on at once.
#: Together refuses `PIPELINE`, which would switch a call in flight, on LoRA.
BACKGROUND_PUBLISH = "WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH"

#: The adapters one model resource holds, a session each. Together refuses the sixth ("is at
#: capacity (5/5 adapter slots in use)", measured 25 Nov on Qwen/Qwen3.5-9B) and publishes the
#: number nowhere, so the run plans with this one.
ADAPTER_SLOTS = 5

#: Provisions a model resource for a base model with that many sampler replicas, and returns it
#: once it is ready.
Provision = Callable[[str, int], Awaitable[Any]]


class SessionStopped(RuntimeError):
    """Together stopped the session an operation ran on, so the operation would never end."""


class Resource:
    """A Together model resource: the GPUs a base model is trained and sampled on, which the
    clients placed on it share, each with a session of its own or a share of the untrained one.
    The first of them to open provisions it, with the sampler replicas they ask for between
    them, and it stops, by force, once the last of them has closed."""

    def __init__(self, model: str) -> None:
        self.model = model
        #: The clients placed on it, until each closes.
        self.clients: list[TogetherClient] = []
        #: The model resources of its base model, itself among them, which start together.
        self.group: list[Resource] = [self]
        #: Together's handle on it, once it is provisioned and until it stops.
        self.handle: Any = None
        self._starting = asyncio.Lock()

    @property
    def sampler_replicas(self) -> int:
        """The sampler replicas it holds: the ones each client placed on it asks for."""
        return sum(one.sampler_replicas for one in self.clients)

    async def join(self, client: TogetherClient) -> Any:
        """Return its handle, provisioned first if it has none, with the others of its group, so
        that the wait for GPUs is one for them all. A failure of another's is left to the
        clients placed on that one, which meet it as they join."""
        started = [one._started(client._provision) for one in self.group]
        found = await asyncio.gather(*started, return_exceptions=True)
        mine = found[self.group.index(self)]
        if isinstance(mine, BaseException):
            raise mine
        return mine

    async def leave(self, client: TogetherClient) -> None:
        """Take `client` off it, and stop it once no client is left on it."""
        if client in self.clients:
            self.clients.remove(client)
        if not self.clients:
            await self.stop()

    async def stop(self) -> None:
        """Stop it by force, which Together otherwise refuses while a session still runs an
        operation, as when a run fails in an update. One that will not stop stays held, and the
        interpreter tries it again when it exits."""
        if self.handle is None:
            return
        try:
            await self.handle.stop_async(force=True)
        except Exception:  # noqa: BLE001 - a resource that will not stop is not a run failure
            logger.warning(
                "could not stop the Together model resource: the interpreter tries again when "
                "it exits",
                exc_info=True,
            )
            return
        self.handle = None
        HELD.discard(self)

    async def _started(self, provision: Provision) -> Any:
        """Return its handle, provisioned first if it has none and a client is still placed on
        it; one whose clients all closed as it started stops at once."""
        async with self._starting:
            if self.handle is None and self.clients:
                self.handle = await provision(self.model, self.sampler_replicas)
                HELD.add(self)
                hooked()
        if not self.clients:
            await self.stop()
        return self.handle


class Seat:
    """The session the clients of an untrained model share on a model resource: an adapter no
    one trains scores and samples as the base model does, whoever asks (measured 25 Nov: two
    fresh sessions scored a sequence alike), so the runs of a process open one between them.
    The first of them to need it opens it, and the last to close stops it."""

    def __init__(self) -> None:
        self.session: Any = None
        #: The clients that share it, until each closes.
        self.clients: list[TogetherClient] = []
        #: Held for each training operation on it: Together runs one at a time on a session.
        self.lock = asyncio.Lock()
        self._opening = asyncio.Lock()

    async def open(self, client: TogetherClient) -> Any:
        """Return the session, opened by `client` if no one has yet."""
        async with self._opening:
            if self.session is None:
                self.session = await client.resources.create_session_async(
                    display_name=client.run, lora_config={"rank": lora_rank(client.lora_rank)}
                )
            return self.session

    async def leave(self, client: TogetherClient) -> None:
        """Take `client` off it, and stop the session once no client is left on it."""
        if client in self.clients:
            self.clients.remove(client)
        if not self.clients and self.session is not None:
            await self.session.stop_async()
            self.session = None


def plan(clients: Values[TogetherClient]) -> list[Resource]:
    """Place each client that has no place yet on a model resource, and return the resources
    made for them. The clients of a base model share its resources: each one the run trains,
    or serves from a checkpoint, is an adapter of its own, and the untrained ones are one
    between them. A resource holds `ADAPTER_SLOTS` adapters, so more are spread evenly over as
    many resources as they need, the untrained one on the first."""
    made: list[Resource] = []
    by_model: dict[str, list[TogetherClient]] = {}
    for one in clients:
        if one.place is None:
            by_model.setdefault(one.model, []).append(one)
    for model, found in by_model.items():
        untrained = [one for one in found if one.shares]
        adapters = ([untrained] if untrained else []) + [[one] for one in found if not one.shares]
        group = [Resource(model) for _ in range(-(-len(adapters) // ADAPTER_SLOTS))]
        for resource in group:
            resource.group = group
        seat = Seat() if untrained else None
        for at, sharing in enumerate(adapters):
            resource = group[at % len(group)]
            for one in sharing:
                one.place = resource
                resource.clients.append(one)
                if one.shares and seat is not None:
                    one.seat = seat
                    seat.clients.append(one)
        made += group
    return made


class TogetherClient:
    """A model's Together session, on a model resource it shares with the clients placed with
    it, provisioned as the first of them opens. It opens the same whether the recipe trains the
    model or not: the proxy samples it through the session."""

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
        seed: int | None = None,
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
        #: The seed of a new adapter's first weights; None for Together's own.
        self.seed = seed
        #: How many replicas of the sampler it brings to its resource: Together's generator
        #: replicas, which the resource holds for all the clients placed on it.
        self.sampler_replicas = sampler_replicas
        #: The model resource it is placed on, which `plan` gives it, alone or with the clients
        #: that open with it; None until then, and `open` places a client that has none alone.
        self.place: Resource | None = None
        #: The session it shares with the other untrained clients of its base model, if any.
        self.seat: Seat | None = None
        #: Together's handle on its resource, from when it opens until it closes.
        self.resources: Any = None
        #: The training session, opened when first needed: a resume opens it on its checkpoint.
        self.session: Any = None
        #: Whether an optimizer step moved the weights since they were last published.
        self.moved = False
        #: The version of the weights published last, which names the weights that answered each
        #: call; None for weights published without a sync, as at the start.
        self.version: int | None = None
        #: The syncs the session's generator took: the version it stamps each reply with. Together
        #: counts syncs there, where `weights_sync` answers the optimizer steps taken (measured 24
        #: Oct: after 10 updates of 2 optimizer steps, a reply said 10 and the last sync 20).
        self.syncs = 0
        self._provision = provision if provision is not None else provisioned
        self._lock = asyncio.Lock()

    async def context_length(self) -> int | None:
        """Return the model's context length as Together's list of models says it, read as the
        proxy's lane reads it when it opens, which provisions nothing; None when it does not
        say. The read needs no session: the list is the account's."""
        from harness_proxy.backends.together import context_of

        return await context_of(None, self.model)

    @property
    def shares(self) -> bool:
        """Whether it serves the base model's own weights, untrained, so that one session does
        for every client that does."""
        return not self.trains and not self.from_checkpoint

    async def open(self) -> None:
        """Join its model resource, provisioned now if this is the first client on it to open,
        and have the interpreter stop what is left when it exits, if the run has not."""
        if self.place is None:
            plan([self])
        assert self.place is not None
        self.resources = await self.place.join(self)
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
            result = await self._one_with_data(
                partial(
                    session.trainer.forward_backward_async,
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
        result = await self._one_with_data(
            partial(
                session.trainer.forward_async,
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
        """Point the generator at the weights the updates moved to, and return their name. The
        sync runs in the background, so the trainer goes on at once. The calls in flight end on
        the weights they began with, and the calls that come meanwhile wait until the last of them
        ends, then take the new ones (measured 24 Oct). Each reply names the version that wrote
        it, counting syncs: keep the new weights' as `version`."""
        session = await self._session()
        self.version = None
        if self.moved:
            await self._one(
                session.trainer.weights_sync_async(
                    weight_sync_type=BACKGROUND_PUBLISH, timeout=OPERATION_SECONDS
                )
            )
            self.syncs += 1
            self.version = self.syncs
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
        self.syncs = 0  # a new session's generator counts its syncs anew (unmeasured on a resume)

    async def close(self, status: str = "success", detail: str | None = None) -> None:
        """Stop the session, or leave the one it shares, which stops with the last client on
        it; then leave the model resource, which bills until it stops, and stops, by force,
        with the last client placed on it. A session that will not stop stays held, as a
        resource does, and the interpreter tries again when it exits. A client that never
        opened gives up its place; one that closed already does nothing."""
        try:
            if self.seat is not None:
                await self.seat.leave(self)
            elif self.session is not None:
                await self.session.stop_async()
        except Exception:  # noqa: BLE001 - a session that will not stop is not a run failure
            logger.warning(
                "could not stop the Together session: the interpreter tries again when it exits",
                exc_info=True,
            )
        else:
            self.session = None
        if self.place is not None:
            await self.place.leave(self)
        self.resources = None
        if self.session is None:
            RUNNING.discard(self)

    async def _session(self) -> Any:
        """Return the session, opened on first use: the one it shares, a new adapter, from its
        seed if it has one, or `from_checkpoint`'s."""
        if self.session is None:
            if self.seat is not None:
                self.session = await self.seat.open(self)
            elif self.from_checkpoint:
                self.session = await self.resources.create_session_async(
                    display_name=self.run,
                    resume_from_checkpoint_id=self.from_checkpoint,
                    load_optimizer=self.restore_optimizer,
                )
            else:
                seeded = {} if self.seed is None else {"seed": self.seed}
                self.session = await self.resources.create_session_async(
                    display_name=self.run,
                    lora_config={"rank": lora_rank(self.lora_rank), **seeded},
                )
        return self.session

    async def _one[T](self, work: Awaitable[T]) -> T:
        """Await one training operation, after the one before it on its session, its own or the
        one it shares, looking at the session every `SESSION_CHECK_SECONDS`: SessionStopped once
        Together has stopped it."""
        async with self.seat.lock if self.seat is not None else self._lock:
            session, running = self.session, asyncio.ensure_future(work)
            try:
                while not (await asyncio.wait({running}, timeout=SESSION_CHECK_SECONDS))[0]:
                    await check_session(session)
                return running.result()
            finally:
                running.cancel()

    async def _one_with_data[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one training operation whose data the SDK uploads before it sends the operation,
        after the one before it; again, `UPLOAD_RETRIES` times at most, `UPLOAD_RETRY_SECONDS`
        after each upload that loses its connection. Such an operation was never sent, so it
        never runs twice."""
        from together import APIConnectionError

        retries = 0
        while True:
            try:
                return await self._one(operation())
            except APIConnectionError as error:
                if retries == UPLOAD_RETRIES or not unsent(error.request):
                    raise
            retries += 1
            logger.warning(
                "an operation's data lost its connection as it uploaded: sending it again in "
                "%.0f s (%d of %d)",
                UPLOAD_RETRY_SECONDS,
                retries,
                UPLOAD_RETRIES,
            )
            await asyncio.sleep(UPLOAD_RETRY_SECONDS)


#: Together clients whose session may still run: the interpreter closes each when it exits, if
#: the run has not, as a crash or a second interrupt can leave them billing.
RUNNING: set[TogetherClient] = set()
#: The model resources that are up: the interpreter stops each still up when it exits, as one
#: that would not stop is, and one whose clients never opened to close.
HELD: set[Resource] = set()
#: Whether the interpreter's exit stops `RUNNING` and `HELD`.
_hooked = False


def hold(client: TogetherClient) -> None:
    """Hold `client` until it closes: the interpreter closes it when it exits."""
    RUNNING.add(client)
    hooked()


def hooked() -> None:
    """Have the interpreter stop what Together still runs when it exits, and SIGTERM, when
    nothing handles it, exit the interpreter so that it does."""
    global _hooked
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
    """Close every Together client the run left open, which stops its session and, with the
    last client on it, its model resource; then stop every model resource still up."""
    for client in list(RUNNING):
        try:
            asyncio.run(client.close("interrupted", "the process exited before the run closed it"))
        except Exception:  # noqa: BLE001 - the interpreter is exiting; say so and go on
            logger.warning("could not stop a Together client as the process exited", exc_info=True)
    for one in list(HELD):
        asyncio.run(one.stop())


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


async def check_session(session: Any) -> None:
    """Raise SessionStopped if Together has stopped `session`. A look that fails tells nothing:
    the operation's own polling fails too if Together stays out of reach."""
    from together import APIError

    try:
        status = (await session.retrieve_async()).status
    except APIError:
        logger.warning("could not look at the Together session: waiting on", exc_info=True)
        return
    if status in ENDED:
        raise SessionStopped(
            f"Together's session {session.session_id} is {status}: the operation it ran will "
            "never end. Together stops a session when the project's balance runs out."
        )


def unsent(request: httpx.Request) -> bool:
    """Whether a request that lost its connection left its operation unsent: the SDK uploads an
    operation's data, asking for an upload URL and then putting the data there, before it sends
    the operation, and downloads a result's data only after the operation ran."""
    path = request.url.path
    putting = request.method == "PUT" and "/payloads/" in path
    return putting or path.endswith("/payloads/upload-url")


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
