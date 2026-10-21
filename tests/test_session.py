"""Tests for `shipyard.models.backends.Session`: what it trains and saves, with a fake
training client in place of a provider's SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import tinker
import torch

from shipyard.groups import Sequence, Trial
from shipyard.jobs.harnesses import harness
from shipyard.models.backends import Saved, Session
from shipyard.run.directory import open_run
from tests.fakes import FakeProxy
from tests.runs import blueprint
from tests.service import FakeService
from tests.test_endpoints import agent

#: The trial a test's sequences belong to; the session reads only their tokens and advantages.
TRIAL = Trial("j", Path("t"), harness("pi"), Path("t"))
ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)


@dataclass
class FakeClient:
    """A training client that trains nothing: it keeps what it was handed and saved, and gives
    every slot the logprob `trained`."""

    trained: float = -0.5
    provider: str = "tinker"
    model: str = "Qwen/Qwen3-8B"
    #: Whether what it saves expires, as Tinker's does.
    expires: bool = True
    handed: list[tuple[Any, ...]] = field(default_factory=list)
    saves: list[tuple[str, str, int | None]] = field(default_factory=list)

    async def fit(
        self, parts: Any, adam: Any, rate: float, loss_fn: str, loss_config: Any
    ) -> list[list[float]]:
        self.handed.append((parts, adam, rate, loss_fn, loss_config))
        return [[self.trained] * one.model_input.length for part in parts for one in part]

    async def save_state(self, name: str, *, ttl_seconds: int | None) -> str:
        self.saves.append(("state", name, ttl_seconds))
        return f"state/{name}"

    async def save_weights_for_sampler(self, name: str, *, ttl_seconds: int | None) -> str:
        self.saves.append(("sampler", name, ttl_seconds))
        return f"sampler/{name}"


def _session(tmp_path: Path, spent: list[Any], **replacements: str) -> tuple[Session, FakeClient]:
    directory, config = open_run(blueprint(tmp_path, **replacements), root=tmp_path / "runs")
    session = Session(
        config,
        directory,
        lambda *billed, **tokens: spent.append((billed, tokens)),
        agent(config, directory),
    )
    session.client = client = FakeClient()  # type: ignore[assignment]
    return session, client


def _sequence(tokens: int, advantage: float) -> Sequence:
    """A chain of `tokens` tokens whose last two were sampled, both at `advantage`."""
    made = Sequence(tuple(range(1, tokens + 1)), ((tokens - 2, tokens),), None, (), TRIAL)
    made.advantages = torch.full((2,), advantage)
    return made


async def test_fit_hands_the_client_each_part_as_datums_and_files_the_tokens_read(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    spent: list[Any] = []
    session, client = _session(tmp_path, spent)
    one, two, three = _sequence(4, 1.0), _sequence(6, -1.0), _sequence(4, 0.5)
    logprobs = {one: [-0.5, -0.5], two: (-0.5, -0.5), three: [-0.5, -0.5]}
    trained = await session.fit(
        [[one, two], [three]], logprobs=logprobs, adam=ADAM, rate=1e-5, loss="ppo", loss_config=None
    )
    ((parts, adam, rate, loss, config),) = client.handed
    assert [[datum.model_input.length for datum in part] for part in parts] == [[3, 5], [3]]
    assert (adam, rate, loss, config) == (ADAM, 1e-5, "ppo", None)
    first = parts[0][0].loss_fn_inputs
    assert first["logprobs"].tolist() == [0.0, -0.5, -0.5], "each at the target it scored"
    assert first["advantages"].tolist() == [0.0, 1.0, 1.0]
    assert first["mask"].tolist() == [0.0, 1.0, 1.0]
    assert spent == [(("tinker", "Qwen/Qwen3-8B"), {"train": 11})], "every slot read, once"
    assert trained.tokens == 11
    assert (trained.kl_v1, trained.kl_v2) == (pytest.approx(0.0), pytest.approx(0.0)), (
        "trained where it sampled, so nothing drifted"
    )


async def test_save_keeps_a_checkpoint_for_its_ttl_unless_it_is_kept_for_good(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    ttl = {"[recipe]": "[checkpoints]\nttl_hours = 2\n\n[recipe]"}
    session, client = _session(tmp_path, [], **ttl)
    assert await session.save("step-2", keep=False) == Saved("state/step-2", None, 2.0)
    assert await session.save("final", keep=True) == Saved("state/final", "sampler/final", None)
    assert client.saves == [
        ("state", "step-2", 2 * 3600),
        ("state", "final", None),
        ("sampler", "final", None),
    ]
