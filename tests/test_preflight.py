"""`check`'s findings beyond the schema: the facts, the datasets under `tasks/`, the
harness and sandbox, and where the proxy stands (here, behind a tunnel, or elsewhere)."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipyard import config, preflight
from shipyard.config import Finding, check
from shipyard.proxy import tunnel
from tests.trials import fixture_tasks, write_blueprint

BLUEPRINTS = Path(__file__).parent / "blueprints"

#: An evaluate blueprint `check` passes from a working directory holding the fixtures.
HOSTED = """
[model]
name = "m"
provider = "openrouter"
[data]
dataset = "fixture"
batch_size = 2
[rollout]
harness = "pi@0.85.1"
[recipe]
kind = "evaluate"
"""


# ------------------------------------------------------------------ the findings

#: The probe's finding without a key, the one every offline check ends on.
UNASKED = "could not ask tinker whether it serves Qwen/Qwen3-8B: TINKER_API_KEY is unset"


def test_check_is_only_ok_for_the_evaluate_fixture_over_its_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    findings = check(BLUEPRINTS / "evaluate")
    assert findings and all(f.level == "ok" for f in findings)
    assert [f.text for f in findings] == [
        "recipe evaluate",
        "model some-hosted-model, provider openrouter",
        "dataset fixture",
        "harness pi@0.85.1, sandbox modal",
        "dataset fixture: 2 tasks under tasks/fixture",
    ]


@pytest.mark.parametrize("kind", ["dapo", "gepa"])
def test_check_names_the_facts_then_says_where_the_proxy_stands(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tinker-served fixtures state their facts, miss their datasets here, and are
    served on this machine for their docker sandbox; the probe cannot ask without a key."""
    monkeypatch.chdir(tmp_path)
    findings = check(BLUEPRINTS / kind)
    texts = [f.text for f in findings]
    assert texts[:4] == [
        f"recipe {kind}",
        "model Qwen/Qwen3-8B, provider tinker (served by this run)",
        "datasets aime-train, math-500" if kind == "gepa" else "dataset aime-train",
        "harness pi@0.85.1, sandbox docker",
    ]
    assert [f.level for f in findings[:4]] == ["ok"] * 4
    assert findings[-2] == Finding(
        "ok", "serving Qwen/Qwen3-8B on this machine for a docker sandbox"
    )
    assert findings[-1] == Finding("warning", UNASKED)
    missing = [f for f in findings if f.level == "blocked" and "no dataset at" in f.text]
    assert len(missing) == (2 if kind == "gepa" else 1)
    assert str(Path("tasks") / "aime-train") in missing[0].text
    assert [f for f in findings if f.level == "blocked"] == missing


def test_check_blocks_a_missing_dataset_with_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    text = HOSTED.replace('dataset = "fixture"', 'dataset = ["fixture", "absent"]')
    findings = check(write_blueprint(tmp_path, text))
    blocked = [f for f in findings if f.level == "blocked"]
    assert len(blocked) == 1
    assert "no dataset at tasks/absent" in blocked[0].text
    assert "harbor datasets download" in blocked[0].text
    assert Finding("ok", "dataset fixture: 2 tasks under tasks/fixture") in findings


def test_check_blocks_an_unknown_sandbox_and_a_blank_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    text = HOSTED.replace('harness = "pi@0.85.1"', 'harness = "  "\nsandbox = "cloud9"')
    blocked = [f.text for f in check(write_blueprint(tmp_path, text)) if f.level == "blocked"]
    assert len(blocked) == 2
    assert blocked[0].startswith("[rollout] harness: name the harness")
    assert blocked[1].startswith("[rollout] sandbox: no Harbor environment called 'cloud9'")
    assert "docker, podman" in blocked[1] and "modal" in blocked[1]
    assert set(config.SANDBOXES) >= {"docker", "modal", "podman", "e2b", "daytona"}


SERVED = HOSTED.replace('provider = "openrouter"', 'provider = "tinker"').replace(
    'name = "m"', 'name = "Qwen/Qwen3-8B"'
)


def _served(tmp_path: Path, **rollout: object) -> list[Finding]:
    """The served evaluate blueprint over the fixture dataset, with `[rollout]` keys added."""
    extra = (
        "".join(
            f"{key} = {value if not isinstance(value, str) else repr(value)}\n"
            for key, value in rollout.items()
        )
        .replace("True", "true")
        .replace("False", "false")
    )
    base = SERVED.replace('harness = "pi@0.85.1"\n', "") if "harness" in rollout else SERVED
    return check(write_blueprint(tmp_path, base.replace("[rollout]\n", f"[rollout]\n{extra}")))


def test_check_serves_a_local_sandbox_on_this_machine_and_warns_it_could_not_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    findings = _served(tmp_path, sandbox="docker")
    assert [f.level for f in findings] == ["ok"] * 6 + ["warning"]
    assert Finding("ok", "model Qwen/Qwen3-8B, provider tinker (served by this run)") in findings
    assert findings[-2:] == [
        Finding("ok", "serving Qwen/Qwen3-8B on this machine for a docker sandbox"),
        Finding("warning", UNASKED),
    ]
    assert _served(tmp_path / "p", sandbox="podman")[-2].text.endswith("for a podman sandbox")


def test_check_names_the_tunnel_for_a_sandbox_elsewhere_or_blocks_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shipyard.proxy.tunnel.found", lambda: ("/opt/bin/cloudflared", []))
    findings = _served(tmp_path, sandbox="modal")
    assert (
        Finding(
            "ok",
            "serving Qwen/Qwen3-8B behind a cloudflared tunnel (found at /opt/bin/cloudflared)",
        )
        in findings
    )
    # The probe's warning, and the one that says only streams are kept alive.
    assert [f.level for f in findings] == ["ok"] * 6 + ["warning"] * 2
    assert findings[-2] == Finding("warning", preflight.NOT_KEPT_ALIVE)
    docker = ["/usr/local/bin/docker", "run", "--rm", tunnel.IMAGE, "tunnel"]
    monkeypatch.setattr("shipyard.proxy.tunnel.found", lambda: (docker[0], docker))
    assert Finding(
        "ok",
        "serving Qwen/Qwen3-8B behind a cloudflared tunnel (found at docker at "
        f"/usr/local/bin/docker, image {tunnel.IMAGE})",
    ) in _served(tmp_path / "d", sandbox="modal")
    monkeypatch.setattr("shipyard.proxy.tunnel.found", lambda: None)
    findings = _served(tmp_path / "none", sandbox="modal")
    assert [f.level for f in findings].count("blocked") == 1
    assert (
        Finding(
            "blocked",
            "sandbox modal cannot reach a proxy in this process and cloudflared is not installed; "
            "install it or set [rollout] endpoint_url",
        )
        in findings
    )


def test_check_warns_of_long_replies_over_a_tunnel_and_of_an_unprofiled_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shipyard.proxy.tunnel.found", lambda: ("/opt/bin/cloudflared", []))
    warned = [
        f.text for f in _served(tmp_path, sandbox="modal", max_tokens=32768) if f.level == "warning"
    ]
    assert any(text.startswith("[rollout] max_tokens = 32768") for text in warned)
    local = _served(tmp_path / "l", sandbox="docker", max_tokens=32768)
    assert not any("max_tokens" in f.text for f in local), "a sandbox here, no warning"
    monkeypatch.setenv("SHIPYARD_PROXY_TOKEN", "k")
    named = _served(
        tmp_path / "n", sandbox="docker", endpoint_url="https://p.example", max_tokens=32768
    )
    assert not any("max_tokens" in f.text for f in named), "a proxy elsewhere, the sandbox here"
    far = _served(
        tmp_path / "f", sandbox="modal", endpoint_url="https://p.example", max_tokens=32768
    )
    assert any("max_tokens = 32768" in f.text for f in far), "the sandbox elsewhere is the slow leg"
    findings = _served(tmp_path / "h", harness="swe-agent@1.0")
    assert (
        Finding("warning", "no profile for harness swe-agent@1.0; generic OpenAI wiring")
        in findings
    )
    assert not any("no profile" in f.text for f in _served(tmp_path / "t", harness="terminus-2"))


def test_check_on_a_remote_proxy_needs_the_token_in_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    findings = _served(tmp_path, sandbox="modal", endpoint_url="https://p.example")
    assert Finding("ok", "rollouts reach the proxy at https://p.example") in findings
    blocked = [f.text for f in findings if f.level == "blocked"]
    assert len(blocked) == 1 and "SHIPYARD_PROXY_TOKEN is unset" in blocked[0]
    monkeypatch.setenv("SHIPYARD_PROXY_TOKEN", "k")
    findings = _served(tmp_path / "k", sandbox="modal", endpoint_url="https://p.example")
    assert not [f for f in findings if f.level == "blocked"]
    assert findings[-1] == Finding("warning", UNASKED)


def test_the_probe_asks_the_backend_only_under_a_key_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TINKER_API_KEY", "sk")
    monkeypatch.setattr("shipyard.preflight.supported_models", lambda: ["Qwen/Qwen3-8B", "other/m"])
    assert _served(tmp_path, sandbox="docker")[-1] == Finding("ok", "tinker serves Qwen/Qwen3-8B")
    monkeypatch.setenv("TINKER_BASE_URL", "https://api.fireworks.ai/inference/v1")
    monkeypatch.setattr("shipyard.preflight.supported_models", lambda: ["other/m"])
    assert _served(tmp_path / "f", sandbox="docker")[-1] == Finding(
        "blocked", "tinker@api.fireworks.ai does not serve Qwen/Qwen3-8B; it serves other/m"
    )

    def broken() -> list[str]:
        raise ConnectionError("no route to host")

    monkeypatch.setattr("shipyard.preflight.supported_models", broken)
    assert _served(tmp_path / "b", sandbox="docker")[-1] == Finding(
        "warning",
        "could not ask tinker@api.fireworks.ai whether it serves Qwen/Qwen3-8B: no route to host",
    )


# ------------------------------------------------------------------ the reflector's key


def test_a_claude_code_reflector_without_an_anthropic_key_is_warned_about() -> None:
    unkeyed = preflight.key_finding("claude-code", "claude-sonnet-5-5", {})
    assert unkeyed is not None and unkeyed.level == "warning"
    assert "ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN" in unkeyed.text
    assert preflight.key_finding("claude-code", None, {"ANTHROPIC_AUTH_TOKEN": "t"}) is None
    assert preflight.key_finding("claude-code@2.1.0", None, {"ANTHROPIC_API_KEY": "k"}) is None


def test_a_reflector_harness_with_no_key_names_or_no_harbor_name_is_quiet() -> None:
    assert preflight.key_finding("pi@0.85.1", "Qwen/Qwen3-8B", {}) is None
    assert preflight.key_finding("no-such-harness", "m", {}) is None
    unkeyed = preflight.key_finding("pi@0.85.1", "anthropic/claude-sonnet-5-5", {})
    assert unkeyed is not None and "ANTHROPIC_API_KEY" in unkeyed.text
