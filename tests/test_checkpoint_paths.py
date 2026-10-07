"""`[model] from_checkpoint` is a state path for a training run and a sampler path for
anything that only samples: `check` names the other column, the proxy of a training run
starts on the base model, and the KL anchor samples weights published from the state."""

from __future__ import annotations

from pathlib import Path

from shipyard.config import check, load
from shipyard.serving import endpoint_settings, start_weights
from tests.trials import write_blueprint

BLUEPRINTS = Path(__file__).parent / "blueprints"
STATE = "tinker://abc:train:0/weights/final"
SAMPLER = "tinker://abc:train:0/sampler_weights/final"


def _with(tmp_path: Path, kind: str, checkpoint: str) -> Path:
    text = (BLUEPRINTS / kind / "run.toml").read_text(encoding="utf-8")
    text = text.replace("[model]\n", f'[model]\nfrom_checkpoint = "{checkpoint}"\n', 1)
    return write_blueprint(tmp_path / kind, text)


def _blocked(home: Path) -> list[str]:
    return [found.text for found in check(home) if found.level == "blocked"]


def test_a_training_run_names_a_state_path(tmp_path: Path) -> None:
    assert not any("from_checkpoint" in text for text in _blocked(_with(tmp_path, "dapo", STATE)))
    (wrong,) = [
        t for t in _blocked(_with(tmp_path / "s", "dapo", SAMPLER)) if "from_checkpoint" in t
    ]
    assert "state_path" in wrong


def test_a_sampling_run_names_a_sampler_path(tmp_path: Path) -> None:
    home = _with(tmp_path, "evaluate", SAMPLER)
    assert not any("from_checkpoint" in text for text in _blocked(home))
    (wrong,) = [
        t for t in _blocked(_with(tmp_path / "w", "evaluate", STATE)) if "from_checkpoint" in t
    ]
    assert "sampler_path" in wrong and wrong.startswith("[model] from_checkpoint: evaluate")


def test_the_proxy_of_a_training_run_starts_on_the_base_model(tmp_path: Path) -> None:
    training = load(_with(tmp_path, "dapo", STATE))
    sampling = load(_with(tmp_path, "evaluate", SAMPLER))
    assert training.trains and not sampling.trains
    assert start_weights(training) is None and endpoint_settings(training)["weights"] is None
    assert start_weights(sampling) == SAMPLER == endpoint_settings(sampling)["weights"]
