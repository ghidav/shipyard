"""Tests for `check` findings beyond the schema: datasets, harness, sandbox, and proxy placement."""

from __future__ import annotations

from pathlib import Path

import pytest
from tinker.types import SupportedModel

from shipyard.cli import preflight
from shipyard.cli.preflight import Finding
from shipyard.proxy import tunnel
from tests.trials import BLUEPRINTS, check, fixture_tasks, write_blueprint

#: Provider-hosted evaluate blueprint that passes `check` from a directory with the fixtures.
HOSTED = """
[agent]
model = "m"
provider = "openrouter"
harness = "pi@0.85.1"

[data]
test = "fixture"
batch_size = 2
[rollout]
[recipe]
kind = "evaluate"
"""


# ------------------------------------------------------------------ the findings

#: The model probe's warning without an API key; every offline `check` ends with it.
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
        "test fixture",
        "harness pi@0.85.1, sandbox modal",
        "dataset fixture: 2 tasks under tasks/fixture",
    ]


@pytest.mark.parametrize("kind", ["dapo", "gepa"])
def test_check_names_the_facts_then_says_where_the_proxy_stands(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Tinker-served fixtures have no datasets in this directory and use a docker
    sandbox, so the proxy runs on this machine. Without an API key the probe only warns."""
    monkeypatch.chdir(tmp_path)
    findings = check(BLUEPRINTS / kind)
    texts = [f.text for f in findings]
    assert texts[:4] == [
        f"recipe {kind}",
        "model Qwen/Qwen3-8B, provider tinker (served by this run)",
        "train aime-train, math-500" if kind == "gepa" else "train aime-train",
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
    text = HOSTED.replace('test = "fixture"', 'test = ["fixture", "absent"]')
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
    text = HOSTED.replace('harness = "pi@0.85.1"', 'harness = "  "').replace(
        "[rollout]\n", '[rollout]\nsandbox = "cloud9"\n'
    )
    blocked = [f.text for f in check(write_blueprint(tmp_path, text)) if f.level == "blocked"]
    assert len(blocked) == 2
    assert blocked[0].startswith("[agent] harness: name the harness")
    assert blocked[1].startswith("[rollout] sandbox: no Harbor environment called 'cloud9'")
    assert "docker, podman" in blocked[1] and "modal" in blocked[1]
    assert set(preflight.SANDBOXES) >= {"docker", "modal", "podman", "e2b", "daytona"}


SERVED = HOSTED.replace('provider = "openrouter"', 'provider = "tinker"').replace(
    'model = "m"', 'model = "Qwen/Qwen3-8B"'
)


def _served(tmp_path: Path, **rollout: object) -> list[Finding]:
    """Run `check` on the Tinker-served evaluate blueprint with extra `[rollout]` keys. A
    `harness` keyword replaces the `[agent]` harness instead."""
    base = SERVED
    if "harness" in rollout:
        base = base.replace('harness = "pi@0.85.1"', f"harness = {rollout.pop('harness')!r}")
    extra = (
        "".join(
            f"{key} = {value if not isinstance(value, str) else repr(value)}\n"
            for key, value in rollout.items()
        )
        .replace("True", "true")
        .replace("False", "false")
    )
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


def test_compaction_switched_off_by_hand_without_fill_context_is_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a harness profile's compaction key set to the profile's value warns. The
    warning points to `fill_context`, which sets the same keys and gives an overflowing
    call the remaining context instead of the 400 a compacting harness expects. A key set
    to any other value leaves compaction on."""
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    env = 'env = { DISABLE_COMPACT = "1", OTHER = "x" }'
    claude = f'harness = "claude-code"\n{env}'
    found = check(write_blueprint(tmp_path / "c", SERVED.replace('harness = "pi@0.85.1"', claude)))
    assert (
        Finding(
            "warning",
            "[agent] env sets DISABLE_COMPACT, which switches the harness's compaction off. "
            "The proxy still refuses a call that overflows the context, as a harness that "
            "compacts expects. Set [rollout] fill_context = true instead. It sets the same keys "
            "and gives such a call what is left",
        )
        in found
    )
    mixed = 'harness = "claude-code"\nenv = { DISABLE_COMPACT = "0", DISABLE_AUTO_COMPACT = "1" }'
    said = check(write_blueprint(tmp_path / "m", SERVED.replace('harness = "pi@0.85.1"', mixed)))
    assert [f.text.split(",")[0] for f in said if "compaction" in f.text] == [
        "[agent] env sets DISABLE_AUTO_COMPACT"
    ]
    on = 'harness = "claude-code"\nenv = { DISABLE_COMPACT = "0" }'
    kept = check(write_blueprint(tmp_path / "o", SERVED.replace('harness = "pi@0.85.1"', on)))
    assert not any("compaction" in f.text for f in kept)
    filled = SERVED.replace('harness = "pi@0.85.1"', f"{claude}\nfill_context = true")
    assert not any("compaction" in f.text for f in check(write_blueprint(tmp_path / "f", filled)))
    pi = SERVED.replace("[rollout]\n", f"[rollout]\n{env}\n")
    assert not any("compaction" in f.text for f in check(write_blueprint(tmp_path / "p", pi)))
    hosted = HOSTED.replace('harness = "pi@0.85.1"', claude)
    assert not any("compaction" in f.text for f in check(write_blueprint(tmp_path / "h", hosted)))


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
    served = _catalog(
        SupportedModel(model_name="Qwen/Qwen3-8B"), SupportedModel(model_name="other/m")
    )
    monkeypatch.setattr("shipyard.cli.preflight.supported_models", lambda: served)
    assert _served(tmp_path, sandbox="docker")[-1] == Finding("ok", "tinker serves Qwen/Qwen3-8B")
    monkeypatch.setenv("TINKER_BASE_URL", "https://api.fireworks.ai/inference/v1")
    other = _catalog(SupportedModel(model_name="other/m"))
    monkeypatch.setattr("shipyard.cli.preflight.supported_models", lambda: other)
    assert _served(tmp_path / "f", sandbox="docker")[-1] == Finding(
        "blocked", "tinker@api.fireworks.ai does not serve Qwen/Qwen3-8B; it serves other/m"
    )

    def broken() -> dict[str, SupportedModel]:
        raise ConnectionError("no route to host")

    monkeypatch.setattr("shipyard.cli.preflight.supported_models", broken)
    assert _served(tmp_path / "b", sandbox="docker")[-1] == Finding(
        "warning",
        "could not ask tinker@api.fireworks.ai whether it serves Qwen/Qwen3-8B: no route to host",
    )


# ------------------------------------------------------------------------ the teacher

#: The opd fixture's teacher.
TEACHER = "Qwen/Qwen3-32B"


def _catalog(*models: SupportedModel) -> dict[str, SupportedModel]:
    return {str(one.model_name): one for one in models}


def _taught(tmp_path: Path, teacher: str = TEACHER, rollout: str = "") -> list[Finding]:
    """Run `check` on the opd fixture with another teacher and extra `[rollout]` keys, and
    return the findings that name the teacher or its tokenizer."""
    text = (BLUEPRINTS / "opd" / "run.toml").read_text(encoding="utf-8")
    text = text.replace(f'teacher = "{TEACHER}"', f'teacher = "{teacher}"')
    text = text.replace("[rollout]\n", f"[rollout]\n{rollout}")
    found = check(write_blueprint(tmp_path, text))
    return [f for f in found if "teacher" in f.text or "tokeniz" in f.text]


def test_a_teacher_is_named_and_checked_only_under_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _taught(tmp_path / "a") == [
        Finding("ok", f"teacher {TEACHER}, distill 1.0"),
        Finding("warning", f"could not check the teacher {TEACHER}: TINKER_API_KEY is unset"),
    ]
    found = _taught(tmp_path / "b", rollout="temperature = 0.7\n")
    assert (
        Finding(
            "warning",
            "[rollout] temperature = 0.7: the teacher scores at temperature 1, so its term is "
            "measured on tokens the policy sampled at another temperature",
        )
        in found
    )


def test_a_teacher_must_be_served_sampleable_long_enough_and_tokenize_as_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TINKER_API_KEY", "sk")
    models = _catalog(
        SupportedModel(model_name="Qwen/Qwen3-8B", max_context_length=32768),
        SupportedModel(model_name=TEACHER, max_context_length=32768, sampleable=True),
        SupportedModel(model_name="Qwen/Qwen3-Short", max_context_length=8192),
        SupportedModel(model_name="Qwen/Qwen3-Busy", sampleable=False),
        SupportedModel(model_name="meta-llama/Llama-3.1-70B", max_context_length=32768),
    )
    monkeypatch.setattr("shipyard.cli.preflight.supported_models", lambda: models)
    monkeypatch.setattr(
        "shipyard.cli.preflight.vocabulary", lambda model: {"qwen" if "Qwen" in model else "x": 0}
    )
    assert [f for f in _taught(tmp_path / "ok") if f.level != "ok"] == []
    assert Finding("ok", f"the teacher {TEACHER} tokenizes as Qwen/Qwen3-8B does") in _taught(
        tmp_path / "said"
    )
    blocked = {
        "Qwen/Qwen3-9B": "[recipe] teacher: tinker does not serve Qwen/Qwen3-9B",
        "Qwen/Qwen3-Busy": "[recipe] teacher: tinker lists Qwen/Qwen3-Busy but cannot sample it",
        "Qwen/Qwen3-Short": "[recipe] teacher: Qwen/Qwen3-Short reads at most 8192 tokens and a "
        "rollout may reach 32768; set [rollout] max_context = 8192",
        "meta-llama/Llama-3.1-70B": "[recipe] teacher: meta-llama/Llama-3.1-70B tokenizes "
        "differently from Qwen/Qwen3-8B, so it would score other tokens than the policy sampled",
    }
    for at, (teacher, said) in enumerate(blocked.items()):
        found = _taught(tmp_path / str(at), teacher)
        assert [f for f in found if f.level == "blocked"] == [Finding("blocked", said)]
    short = _taught(tmp_path / "capped", "Qwen/Qwen3-Short", "max_context = 8192\n")
    assert [f for f in short if f.level == "blocked"] == []


def test_a_checkpoint_teacher_is_checked_as_its_base_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TINKER_API_KEY", "sk")
    models = _catalog(
        SupportedModel(model_name="Qwen/Qwen3-8B"), SupportedModel(model_name=TEACHER)
    )
    monkeypatch.setattr("shipyard.cli.preflight.supported_models", lambda: models)
    monkeypatch.setattr("shipyard.cli.preflight.vocabulary", lambda model: {"same": 0})
    asked: list[str] = []

    def base(path: str) -> str:
        asked.append(path)
        if "gone" in path:
            raise RuntimeError("404 not found")
        return TEACHER

    monkeypatch.setattr("shipyard.cli.preflight.checkpoint_base", base)
    path = "tinker://r/sampler_weights/final"
    assert Finding("ok", f"tinker serves the teacher {TEACHER}") in _taught(tmp_path / "a", path)
    gone = "tinker://gone/sampler_weights/final"
    assert [f for f in _taught(tmp_path / "b", gone) if f.level == "blocked"] == [
        Finding("blocked", f"[recipe] teacher: cannot read {gone}: 404 not found")
    ]
    assert asked == [path, gone]


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
