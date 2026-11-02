"""Tests for models on Together post-training: the run provisions a model resource for each,
opens a session on it, which trains the model when the recipe does, and its proxy samples that
session's generator."""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import tinker
from together import APIConnectionError

from proxies import MODEL
from shipyard import Config
from shipyard.config import ModelTable
from shipyard.model import Model
from shipyard.models.clients.together import (
    UPLOAD_RETRIES,
    SessionStopped,
    TogetherClient,
    adam_params,
    loss,
    sample,
    stop_running,
)
from shipyard.run import files
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.runs import GRPO, ONE_STEP, run_toml, train
from tests.service import FakeService
from tests.test_connect import agent_keys, connected, table

#: The config's agent, on Together.
ON_TOGETHER = agent_keys(provider="together")


#: Seconds Together's SDK waits for an operation unless told otherwise.
SDK_WAIT = 300.0


@dataclass
class FakeTogetherTrainer:
    """Fake Together trainer: each operation it ran, the seconds each was to be waited on, and
    logprobs of -0.3 for every slot."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    waits: list[float | None] = field(default_factory=list)

    async def forward_backward_async(
        self,
        *,
        samples: Any,
        loss: Any,
        return_loss_fn_outputs: bool,
        timeout: float | None = SDK_WAIT,
    ) -> Any:
        self.calls.append(("forward_backward", (list(samples), loss, return_loss_fn_outputs)))
        self.waits.append(timeout)
        return SimpleNamespace(loss_fn_outputs=[_scored(one) for one in samples])

    async def forward_async(
        self,
        *,
        samples: Any,
        loss: Any,
        return_loss_fn_outputs: bool,
        timeout: float | None = SDK_WAIT,
    ) -> Any:
        self.calls.append(("forward", (list(samples), loss)))
        self.waits.append(timeout)
        return SimpleNamespace(loss_fn_outputs=[_scored(one) for one in samples])

    async def optim_step_async(self, *, adam_params: Any, timeout: float | None = SDK_WAIT) -> None:
        self.calls.append(("optim_step", adam_params))
        self.waits.append(timeout)

    async def weights_sync_async(
        self,
        *,
        weight_sync_type: str = "WEIGHT_SYNC_TYPE_SYNCHRONOUS",
        timeout: float | None = SDK_WAIT,
    ) -> Any:
        """Sync the generator, answering the version its weights now are, as Together does: the
        optimizer steps taken, as a string."""
        self.calls.append(("weights_sync", weight_sync_type))
        self.waits.append(timeout)
        steps = sum(name == "optim_step" for name, _ in self.calls)
        return SimpleNamespace(weights_version=str(steps))


def _scored(one: Any) -> Any:
    slots = sum(len(chunk["encoded_text"]["tokens"]) for chunk in one["model_input"]["chunks"])
    return SimpleNamespace(tensors={"logprobs": SimpleNamespace(data=[-0.3] * slots)})


@dataclass
class FakeTogetherSession:
    """Fake Together session; a `busy` one is still running an operation, so it will not stop.
    Looked at, it answers its `status`, but the first `unreachable` looks lose their connection;
    `looks` counts them."""

    session_id: str
    trainer: FakeTogetherTrainer
    saved: list[str] = field(default_factory=list)
    stopped: bool = False
    busy: bool = False
    status: str = "TRAINING_SESSION_STATUS_RUNNING"
    unreachable: int = 0
    looks: int = 0

    async def retrieve_async(self) -> Any:
        self.looks += 1
        if self.unreachable:
            self.unreachable -= 1
            raise APIConnectionError(request=httpx.Request("GET", SESSION))
        return SimpleNamespace(status=self.status)

    async def create_training_checkpoint_async(self, *, timeout: float | None = SDK_WAIT) -> Any:
        self.saved.append("training")
        self.trainer.waits.append(timeout)
        return SimpleNamespace(id=f"training-{len(self.saved)}")

    async def create_inference_checkpoint_async(self, *, timeout: float | None = SDK_WAIT) -> Any:
        self.saved.append("inference")
        self.trainer.waits.append(timeout)
        return SimpleNamespace(id=f"inference-{len(self.saved)}")

    async def stop_async(self) -> None:
        if self.busy:
            raise TimeoutError(f"Timed out waiting for session {self.session_id} to stop")
        self.stopped = True


@dataclass
class FakeResources:
    """Fake Together model resource: its sampler replicas, and the sessions opened on it with
    what each asked for. Like Together's, it stops while a session runs only when forced."""

    model: str
    sampler_replicas: int = 1
    trainer: FakeTogetherTrainer = field(default_factory=FakeTogetherTrainer)
    sessions: list[tuple[dict[str, Any], FakeTogetherSession]] = field(default_factory=list)
    stopped: bool = False

    async def create_session_async(self, **asked: Any) -> FakeTogetherSession:
        session = FakeTogetherSession(f"s{len(self.sessions) + 1}", self.trainer)
        self.sessions.append((asked, session))
        return session

    async def stop_async(self, *, force: bool = False) -> None:
        if not force and any(not session.stopped for _, session in self.sessions):
            raise RuntimeError("has 1 active session(s); stop sessions first or use force=true")
        self.stopped = True


@pytest.fixture
def provisioned(monkeypatch: pytest.MonkeyPatch) -> list[FakeResources]:
    """Provision fake Together resources, and return them in the order they were made."""
    made: list[FakeResources] = []

    async def provision(model: str, sampler_replicas: int) -> FakeResources:
        made.append(FakeResources(model, sampler_replicas))
        return made[-1]

    monkeypatch.setattr("shipyard.models.clients.together.provisioned", provision)
    return made


async def test_a_run_on_together_trains_its_session_and_its_proxy_samples_it(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    provisioned: list[FakeResources],
) -> None:
    """The proxy samples the session the run opened. An update trains it, syncs the generator
    and names the new weights; the checkpoints are Together's, and both GPUs bills stop."""
    proxy, service = fakes
    harbor(Alternating())
    home = run_toml(tmp_path, **ONE_STEP, **ON_TOGETHER)
    directory = await train(home, tmp_path / "runs")
    (resources,) = provisioned
    ((asked, session),) = resources.sessions
    assert resources.model == MODEL and service.calls == [], "Tinker is never asked"
    assert resources.sampler_replicas == 1, "one sampler replica unless the model asks for more"
    assert asked == {"display_name": directory.name, "lora_config": {"rank": 32}}
    ((_, settings, _),) = proxy.placed
    (lane,) = settings["lanes"]
    assert (lane["backend"], lane["session"]) == ("together", "s1")
    assert set(proxy.weights) == {"together://s1/start"}, "the weights published first"
    assert [name for name, _ in resources.trainer.calls] == [
        "forward_backward",
        "optim_step",
        "weights_sync",
    ]
    # The trainer goes on while the generator takes the new weights.
    assert resources.trainer.calls[2][1] == "WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH"
    assert proxy.published[-1] == ("agent", "together://s1/sample-1", 1), "its version named"
    samples, chosen, outputs = resources.trainer.calls[0][1]
    assert chosen == {"type": "LOSS_TYPE_IMPORTANCE_SAMPLING"} and outputs is True
    inputs = {key for one in samples for key in one["loss_fn_inputs"]}
    assert inputs == {"target_tokens", "logprobs", "advantages"}, "the mask stays local"
    adam = resources.trainer.calls[1][1]
    assert adam["learning_rate"] == 2e-5 and adam["grad_clip_norm"] == 0.0
    (job,) = files.read(directory / files.JOBS)
    assert job["backend"] == "together"
    rows = list(files.read(directory / files.CHECKPOINTS))
    assert [(row["tag"], row["sampler_weights"]) for row in rows] == [
        ("step-1", {}),
        ("final", {"agent": "inference-3"}),
    ]
    assert all(row["training_state"]["agent"].startswith("training-") for row in rows)
    assert [row["ttl_hours"] for row in rows] == [None, None], "Together keeps them until deleted"
    assert session.stopped and resources.stopped, "the session and its GPUs stop with the run"
    assert len(resources.trainer.waits) == 6 and set(resources.trainer.waits) == {None}, (
        "every operation, checkpoints too, is waited on as long as it takes, not the SDK's 300 s"
    )
    spent = files.read_json(directory / files.COSTS)["tokens"]["together"][MODEL]
    assert spent["train"] > 0 and spent["sample"] > 0


async def test_a_model_on_tinker_opens_a_session_of_its_own_beside_a_together_agent(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    provisioned: list[FakeResources],
) -> None:
    _, service = fakes

    class Taught(GRPO):  # type: ignore[misc, valid-type]
        def __init__(self, config: Config, directory: Path) -> None:
            super().__init__(config, directory)
            self.teacher = self.models["teacher"]

    harbor(Alternating())
    home = run_toml(tmp_path, table("teacher", "Qwen/Qwen3-32B"), **ONE_STEP, **ON_TOGETHER)
    await train(home, tmp_path / "runs", Taught)
    assert ("sampling", "Qwen/Qwen3-32B", None) in service.calls and len(provisioned) == 1
    assert [status for status, _ in service.closed] == ["success"], "its session ends with the run"
    home = run_toml(tmp_path, table("teacher", "Qwen/Qwen3-33B"), **ONE_STEP, **ON_TOGETHER)
    with pytest.raises(ValueError, match="does not serve 'Qwen/Qwen3-33B'"):
        await train(home, tmp_path / "runs", Taught)
    assert len(provisioned) == 1, "no GPU is provisioned for a run its teacher refuses"


async def test_two_models_on_together_open_two_sessions_each_on_a_resource_of_its_own(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], provisioned: list[FakeResources]
) -> None:
    proxy, service = fakes
    judge = table("judge", "Qwen/Qwen3-32B", provider="together")
    made, models = connected(tmp_path, judge, **ON_TOGETHER)
    agent, judged = models.all
    trained, served = made.sessions
    assert (agent.backend, judged.backend) == (trained, served)
    assert (trained.trains, served.trains) == (True, False)
    async with made:
        assert [one.model for one in provisioned] == ["Qwen/Qwen3-32B", MODEL], (
            "a resource each, the untrained model's first"
        )
        for session, resource in ((served, provisioned[0]), (trained, provisioned[1])):
            ((_, opened),) = resource.sessions
            assert session.client.session is opened, "each samples its own session"
        ((_, settings, _),) = proxy.placed
        lanes = [(one["name"], one["model"], one["backend"]) for one in settings["lanes"]]
        assert lanes == [("agent", MODEL, "together"), ("judge", "Qwen/Qwen3-32B", "together")]
        told = {name: lane.told for name, lane in made.proxy.lanes.items()}
        assert told == {"agent": "together://s1/start", "judge": "together://s1/start"}, (
            "each serves its starting weights"
        )
    assert all(one.stopped for one in provisioned), "both GPUs stop with the run"
    assert service.calls == [], "Tinker is never asked"


async def test_a_model_on_together_gets_the_sampler_replicas_it_asks_for_and_no_other_does(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], provisioned: list[FakeResources]
) -> None:
    made, _ = connected(tmp_path, **agent_keys(provider="together", sampler_replicas=2))
    async with made:
        assert [one.sampler_replicas for one in provisioned] == [2]
    with pytest.raises(ValueError, match=r"\[models\.teacher\] sampler_replicas: .* tinker runs"):
        Model("teacher", ModelTable(model="Qwen/Qwen3-32B", sampler_replicas=2))


async def test_a_resumed_together_run_opens_its_session_on_the_checkpoint() -> None:
    """A session cannot load a checkpoint once open, so the resumed one is the first opened."""
    resources = FakeResources(MODEL)

    async def provide(model: str, sampler_replicas: int) -> FakeResources:
        return resources

    client = TogetherClient(MODEL, run="r", provision=provide)
    await client.open()
    await client.load_state("training-7")
    assert await client.serving() == {"backend": "together", "session": "s1"}
    assert [asked for asked, _ in resources.sessions] == [
        {"display_name": "r", "resume_from_checkpoint_id": "training-7", "load_optimizer": True}
    ]


async def test_a_publishs_version_counts_the_syncs_as_replies_do_not_the_optimizer_steps() -> None:
    """Together stamps each reply with the syncs its generator took, where a sync answers the
    optimizer steps taken: with two steps an update, the two part from the first update on."""
    resources = FakeResources(MODEL)

    async def provide(model: str, sampler_replicas: int) -> FakeResources:
        return resources

    client = TogetherClient(MODEL, run="r", provision=provide)
    await client.open()
    session = await client._session()
    versions = []
    for _ in range(2):
        for _ in range(2):  # an update of two substeps
            await session.trainer.optim_step_async(adam_params=None)
        client.moved = True
        await client.publish("sample")
        versions.append(client.version)
    assert versions == [1, 2], "replies say 1 and 2, where the syncs answered 2 and 4"
    await client.publish("again")
    assert client.version is None, "nothing moved: no sync, no version"


#: The requests Together's SDK makes for a training pass that carries data: the upload's, before
#: it sends the pass; the pass's own; and the download of its result's, after the pass ran.
SESSION = "https://api.together.xyz/v1/rl/training-sessions/s1"
UPLOAD_URL = ("POST", f"{SESSION}/payloads/upload-url")
UPLOAD = ("PUT", "https://together-rl.r2.cloudflarestorage.com/prod/payloads/s1/p1")
OPERATION = ("POST", f"{SESSION}/operations/forward-backward")
POLL = ("GET", f"{SESSION}/operations/forward-backward/o1")
DOWNLOAD_URL = ("POST", f"{SESSION}/payloads/p2/download-url")
DOWNLOAD = ("GET", "https://together-rl.r2.cloudflarestorage.com/prod/payloads/s1/p2")


@dataclass
class Dropping(FakeTogetherTrainer):
    """A trainer whose first `drops` training passes lose their connection at `request`, as
    Together's SDK raises it; `tried` counts the passes sent, dropped or not."""

    drops: int = 1
    request: tuple[str, str] = UPLOAD
    tried: int = 0

    async def forward_backward_async(
        self,
        *,
        samples: Any,
        loss: Any,
        return_loss_fn_outputs: bool,
        timeout: float | None = SDK_WAIT,
    ) -> Any:
        self.tried += 1
        if self.drops:
            self.drops -= 1
            raise APIConnectionError(request=httpx.Request(*self.request))
        return await super().forward_backward_async(
            samples=samples,
            loss=loss,
            return_loss_fn_outputs=return_loss_fn_outputs,
            timeout=timeout,
        )


async def _fit(client: TogetherClient) -> list[list[float]]:
    """Train `client` on one datum with PPO."""
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3], dtype="int64"),
            "logprobs": tinker.TensorData(data=[0.0, -0.5], dtype="float32"),
            "advantages": tinker.TensorData(data=[0.0, 1.0], dtype="float32"),
        },
    )
    adam = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)
    return await client.fit([[datum]], adam, 1e-4, "ppo", None)


@pytest.mark.parametrize("dropped", [UPLOAD_URL, UPLOAD])
async def test_a_training_pass_whose_data_did_not_upload_is_sent_again_after_a_pause(
    together_exit: list[Any], monkeypatch: pytest.MonkeyPatch, dropped: tuple[str, str]
) -> None:
    """The SDK uploads a pass's data before it sends the pass, so a pass whose upload dropped
    never reached Together: the client sends it again, `UPLOAD_RETRIES` times at most."""
    monkeypatch.setattr("shipyard.models.clients.together.UPLOAD_RETRY_SECONDS", 0.0)
    trainer = Dropping(drops=UPLOAD_RETRIES, request=dropped)
    client = await _opened(FakeResources(MODEL, trainer=trainer))
    assert await _fit(client) == [[-0.3, -0.3]]
    assert trainer.tried == UPLOAD_RETRIES + 1
    assert [name for name, _ in trainer.calls] == ["forward_backward", "optim_step"]
    trainer.drops, trainer.tried = UPLOAD_RETRIES + 1, 0
    with pytest.raises(APIConnectionError):
        await _fit(client)
    assert trainer.tried == UPLOAD_RETRIES + 1, "one drop more than it retries fails the update"


@pytest.mark.parametrize("dropped", [OPERATION, POLL, DOWNLOAD_URL, DOWNLOAD])
async def test_a_training_pass_that_may_have_reached_together_is_not_sent_again(
    together_exit: list[Any], monkeypatch: pytest.MonkeyPatch, dropped: tuple[str, str]
) -> None:
    """A pass whose own request, or its result's download, lost the connection may have run:
    sent again, its gradients would count twice."""
    monkeypatch.setattr("shipyard.models.clients.together.UPLOAD_RETRY_SECONDS", 0.0)
    trainer = Dropping(drops=1, request=dropped)
    client = await _opened(FakeResources(MODEL, trainer=trainer))
    with pytest.raises(APIConnectionError):
        await _fit(client)
    assert trainer.tried == 1 and trainer.calls == []


@dataclass
class Held(FakeTogetherTrainer):
    """A trainer whose training passes run until `released`, as one never ends on a session
    Together stopped."""

    released: asyncio.Event = field(default_factory=asyncio.Event)

    async def forward_backward_async(self, **sent: Any) -> Any:
        await self.released.wait()
        return await super().forward_backward_async(**sent)


async def test_an_operation_fails_once_together_stops_its_session_and_goes_on_while_it_runs(
    together_exit: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Together stops a session, recording no error, when the project's balance runs out, and an
    operation on it never ends (27 Oct: a run waited 6.5 h on one). An operation on a session
    that runs goes on as long as it takes, though a look at the session fails."""
    monkeypatch.setattr("shipyard.models.clients.together.SESSION_CHECK_SECONDS", 0.01)
    stopped = FakeResources(MODEL, trainer=Held())
    client = await _opened(stopped)
    ((_, session),) = stopped.sessions
    session.status = "TRAINING_SESSION_STATUS_STOPPED"
    with pytest.raises(SessionStopped, match="s1 is TRAINING_SESSION_STATUS_STOPPED"):
        await asyncio.wait_for(_fit(client), 5)
    held = Held()
    running = FakeResources(MODEL, trainer=held)
    client = await _opened(running)
    ((_, session),) = running.sessions
    session.unreachable = 2
    asyncio.get_running_loop().call_later(0.2, held.released.set)
    assert await asyncio.wait_for(_fit(client), 5) == [[-0.3, -0.3]]
    assert session.looks > 2, "it looked on, past the looks that failed"


def test_together_takes_tinkers_losses_and_adam_settings_as_its_own() -> None:
    clipped = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}
    assert loss("ppo", clipped) == {"type": "LOSS_TYPE_PPO", "ppo_params": clipped}
    assert loss("importance_sampling", None) == {"type": "LOSS_TYPE_IMPORTANCE_SAMPLING"}
    assert loss("cross_entropy", None) == {"type": "LOSS_TYPE_CROSS_ENTROPY"}
    adam = adam_params(tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8), 1e-4)
    assert adam["learning_rate"] == 1e-4 and adam["grad_clip_norm"] == 0.0
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3], dtype="int64"),
            "logprobs": tinker.TensorData(data=[0.0, -0.5], dtype="float32"),
            "advantages": tinker.TensorData(data=[0.0, 1.0], dtype="float32"),
            "mask": tinker.TensorData(data=[0.0, 1.0], dtype="float32"),
        },
    )
    weighed = {"data": [0.0, 1.0], "dtype": "float32"}
    assert sample(datum, "cross_entropy") == {
        "model_input": {"chunks": [{"encoded_text": {"tokens": [1, 2]}}]},
        "loss_fn_inputs": {"target_tokens": {"data": [2, 3], "dtype": "int64"}, "weights": weighed},
    }, "each target weighed by its advantage, as on Tinker; the mask stays local"
    assert set(sample(datum, "ppo")["loss_fn_inputs"]) == {
        "target_tokens",
        "logprobs",
        "advantages",
    }


@dataclass
class Refusing(FakeResources):
    """A model resource whose first stops fail, as a network that drops them would."""

    refusals: int = 1

    async def stop_async(self, *, force: bool = False) -> None:
        if self.refusals:
            self.refusals -= 1
            raise ConnectionError("the stop did not reach Together")
        self.stopped = True


async def _opened(resources: FakeResources) -> TogetherClient:
    """Return a client opened on `resources`, its session started."""

    async def provide(model: str, sampler_replicas: int) -> FakeResources:
        return resources

    client = TogetherClient(MODEL, run="r", provision=provide)
    await client.open()
    await client.serving()
    return client


async def test_a_client_the_run_never_closed_is_stopped_as_the_interpreter_exits(
    together_exit: list[Any],
) -> None:
    resources = FakeResources(MODEL)
    client = await _opened(resources)
    ((_, session),) = resources.sessions
    assert together_exit == [stop_running], "the hook is registered once a resource runs"
    await asyncio.to_thread(stop_running)  # as the interpreter's exit runs it, with no loop
    assert session.stopped and resources.stopped, "the session and the GPUs stop at exit"
    assert client.session is None and client.resources is None


async def test_a_closed_client_is_left_alone_at_exit_and_one_that_would_not_stop_is_tried_again(
    together_exit: list[Any],
) -> None:
    closed, refusing = FakeResources(MODEL), Refusing(MODEL)
    first, second = await _opened(closed), await _opened(refusing)
    assert together_exit == [stop_running], "one hook, however many clients"
    await first.close()
    await second.close()
    assert closed.stopped and not refusing.stopped, "the second's stop did not get through"
    assert refusing.sessions[0][1].stopped, "its session did"
    await asyncio.to_thread(stop_running)
    assert refusing.stopped, "the exit tries it again"
    stops = closed.stopped
    closed.stopped = False
    await asyncio.to_thread(stop_running)
    assert stops and not closed.stopped, "a client that closed is not stopped again"


async def test_a_run_that_fails_in_an_update_still_stops_its_gpus(
    together_exit: list[Any],
) -> None:
    """Together will not stop a session still running an operation, nor, unforced, the resource
    it runs on: the run stops the resource by force."""
    resources = FakeResources(MODEL)
    client = await _opened(resources)
    ((_, session),) = resources.sessions
    session.busy = True
    await client.close("failed")
    assert resources.stopped and not session.stopped, "the GPUs stop though the session would not"


async def test_sigterm_exits_the_interpreter_unless_something_already_handles_it() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    await _opened(FakeResources(MODEL))
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler) and handler is not signal.SIG_DFL
    with pytest.raises(SystemExit) as exited:
        handler(signal.SIGTERM, None)
    assert exited.value.code == 128 + signal.SIGTERM, "a clean exit, so the exit hook runs"

    def own(signum: int, frame: Any) -> None:
        return None

    signal.signal(signal.SIGTERM, own)
    await _opened(FakeResources(MODEL))
    assert signal.getsignal(signal.SIGTERM) is own, "a handler already set, as the CLI's, is kept"
