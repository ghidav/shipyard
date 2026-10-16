"""The GEPA reflector, run as a Harbor trial.

A synthetic task holds the component's files and one trace per outcome. The agent writes the
rewrite to `/logs/artifacts/<marker>`, and the artifact is read back as a `Module` that the
kind's `check` accepts or rejects."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard import types
from shipyard.modules import KINDS, Module, read, write
from shipyard.sampling.admit import verdict
from shipyard.sampling.endpoint import ProviderEndpoint
from shipyard.training.recipes.gepa.fitness import Outcome
from shipyard.training.recipes.gepa.propose import Proposer, Reflection
from shipyard.types import Agent, Job, Policy, Settings

if TYPE_CHECKING:
    from shipyard.training.trainer import Trainer

logger = logging.getLogger(__name__)

#: Container directory the reflector writes the new version to. Harbor collects it into every
#: trial's `artifacts/logs/artifacts/` without a declaration (harbor/models/trial/paths.py,
#: `EnvironmentPaths.artifacts_dir`).
PUBLISH = "/logs/artifacts"
COLLECTED = ("artifacts", "logs", "artifacts")
#: Setup timeout in seconds. The reflector's harness installs itself into an image that has
#: none, and a Node harness took 366 s on `python:3.12-slim`, past Harbor's default of 360 s.
SETUP_TIMEOUT = 1800.0
#: Trial settings for a reflection: no verifier, and none of the `[rollout]` env, kwargs,
#: timeout or secrets.
REFLECTION = Settings(verify=False, setup_timeout=SETUP_TIMEOUT, secrets=False)
#: The container's workdir, and the names under it of the component's directory, the traces
#: directory and the kind's reference.
WORKDIR = "/app"
CURRENT = "current"
TRACES = "traces"
REFERENCE = "reference.md"
TASK_NAME = "shipyard/reflection"
#: Closing tags a reflector sometimes leaves on the last line of a file.
CLOSING_TAGS = ("</content>", "</file>", "</document>")


def reflect(
    trainer: Trainer,
    *,
    harness: str,
    model: str | None,
    image: str,
    incremental: bool = False,
) -> Proposer:
    """Return a `Proposer` that runs one Harbor trial per proposal on the trainer's engine.

    Each trial runs `harness` with `model` in `image` and is recorded by the trainer. With
    `incremental`, the prompt asks for the smallest change. Every failure is returned as a
    decline, so one failed container cannot end a search. Raise `ValueError` if `harness` is
    empty."""
    if not harness.strip():
        raise ValueError(
            "the reflector needs a harness: Harbor reads an unnamed agent as its oracle, and "
            "the reflection task has no reference solution"
        )

    async def reflector(reflection: Reflection) -> Module | None:
        home = Path(tempfile.mkdtemp(prefix="shipyard-reflection-"))
        try:
            task = task_for(reflection, home, image=image, incremental=incremental)
            rolled = await _job(trainer, task, harness=harness, model=model, about=reflection)
            return written(rolled, reflection.module)
        except Exception:
            logger.exception("the reflection trial for %r failed", reflection.component)
            return None
        finally:
            shutil.rmtree(home, ignore_errors=True)

    return reflector


async def _job(
    trainer: Trainer,
    task: Path,
    *,
    harness: str,
    model: str | None,
    about: Reflection,
) -> Job:
    """Run the reflection task as a one-trial job with its own agent on a provider endpoint,
    recorded with purpose "reflection" rather than as a rollout."""
    job = Job(
        trainer.reserve_name(),
        purpose="reflection",
        about={"component": about.component, "candidate": about.candidate.digest},
    )
    reflector = Agent(Policy(ProviderEndpoint(model)), [types.harness(harness)])
    job.trials = await trainer.engine.run(job.name, [task], 1, agent=reflector, settings=REFLECTION)
    trainer.record_job(job, party=party_of(model))
    return job


def party_of(model: str | None) -> str:
    """Return the party billed for a reflection when Harbor recorded no provider: the
    model's provider prefix, or "unknown"."""
    if model and "/" in model:
        return model.split("/", 1)[0]
    return "unknown"


def task_for(reflection: Reflection, home: Path, *, image: str, incremental: bool) -> Path:
    """Write the reflection as a Harbor task under `home` and return the task directory.

    The task holds `task.toml` naming the prebuilt image (so `environment/` is uploaded into
    the workdir), the component's files under `current/<name>/`, one trace per outcome, the
    kind's reference and the instruction."""
    module = reflection.module
    kind = KINDS[module.kind]
    task = home / "reflection"
    environment = task / "environment"
    (environment / TRACES).mkdir(parents=True)
    write(module.files, into=environment / CURRENT / module.name)
    for index, outcome in enumerate(reflection.outcomes):
        trace = environment / TRACES / f"{index:02d}-{_slug(outcome.task)}.txt"
        trace.write_text(_trace(outcome), encoding="utf-8")
    (task / "task.toml").write_text(
        f'[task]\nname = "{TASK_NAME}"\n'
        'description = "Rewrite one component of a gepa candidate."\n\n'
        f'[environment]\ndocker_image = "{image}"\nworkdir = "{WORKDIR}"\n',
        encoding="utf-8",
    )
    reference = kind.reference()
    if reference:
        (environment / REFERENCE).write_text(reference, encoding="utf-8")
    beside = sorted(path for path in module.files if path != kind.marker)
    (task / "instruction.md").write_text(
        instruction(
            name=module.name,
            marker=kind.marker,
            beside=beside,
            reference=bool(reference),
            incremental=incremental,
            alone=kind.alone,
        ),
        encoding="utf-8",
    )
    return task


def _trace(outcome: Outcome) -> str:
    score = "unmeasured" if outcome.reward is None else f"{outcome.reward:.3f}"
    body = f"task: {outcome.task}\nscore: {score}\n\n"
    if outcome.inputs.strip():
        body += f"--- asked ---\n{outcome.inputs.strip()}\n\n"
    if outcome.transcript.strip():
        body += f"--- transcript ---\n{outcome.transcript.strip()}\n\n"
    return body + f"--- observed ---\n{outcome.feedback}\n"


def instruction(
    *,
    name: str,
    marker: str,
    beside: list[str],
    reference: bool,
    incremental: bool,
    alone: bool = False,
) -> str:
    """Return the reflector's prompt: the goal, where the files are, that nothing else is
    available, what to keep and where to write the answer. When `alone`, the marker is the
    only file written back."""
    current, traces = f"{WORKDIR}/{CURRENT}/{name}", f"{WORKDIR}/{TRACES}"
    return (
        f"Improve `{current}/{marker}` so that an assistant given it does better on tasks "
        f"like the ones in `{traces}/`.\n\n"
        f"Each file in `{traces}/` is one attempt under the current version: the task it "
        "was given, what it scored, the tail of the assistant's own transcript, and what "
        "the grader said. A score of `unmeasured` means the environment failed, which no "
        "rewrite can repair.\n\n"
        + (
            f"Read `{WORKDIR}/{REFERENCE}` first. It is what this kind of component is "
            "written against: what already exists for you to call, and what you must not "
            "redefine. Nothing checks a rewrite against it.\n\n"
            if reference
            else ""
        )
        + (
            f"The directory also holds {', '.join(f'`{one}`' for one in beside)}, which the "
            "text may refer to. They are delivered as they are and are not yours to rewrite."
            "\n\n"
            if beside
            else ""
        )
        + (
            "Make the smallest change that addresses what the traces show: edit the current "
            "text in place, keep its structure and every part that is working, and do not "
            "rewrite it from scratch.\n\n"
            if incremental
            else ""
        )
        + "Those directories are the whole of what you have; there is nothing else here to "
        "find, so do not go looking.\n\n"
        "You are reading traces the assistant will never see. Anything you learn from them "
        "is lost unless you write it into the file, and so is anything already in there that "
        "is working: a rewrite that drops a working instruction to make room for a new one "
        "trades one failure for another.\n\n"
        f"Write the new `{marker}` to `{PUBLISH}/{marker}`. "
        + (
            "It is the only file delivered. "
            if alone
            else f"Whatever else you want kept goes beside it under `{PUBLISH}`, at the same "
            "relative paths it has now; a file you do not write is a file the next assistant "
            "will not have. "
        )
        + "What lands there is given verbatim to the next assistant."
    )


def written(job: Job, module: Module) -> Module | None:
    """Read the rewrite the trial published as a module of the same name and kind.

    Return None, logging the cause, when the artifacts hold no marker or the kind's `check`
    rejects the rewrite. A marker inside exactly one subdirectory is read; if two
    subdirectories hold one, neither is."""
    if not job.trials:
        return None
    trial = job.trials[0].dir
    kind = KINDS[module.kind]
    marker = kind.marker
    published = trial.joinpath(*COLLECTED)
    found = published / marker
    if not found.is_file() and published.is_dir():
        nested = [d for d in sorted(published.iterdir()) if d.is_dir() and (d / marker).is_file()]
        if len(nested) == 1:
            published, found = nested[0], nested[0] / marker
    if not found.is_file():
        _say_why_not(trial, published, marker)
        return None
    files = {marker: found.read_text(encoding="utf-8")} if kind.alone else read(published)
    files[marker] = unwrapped(files[marker])
    rewritten = Module(module.name, module.kind, files)
    refused = kind.check(rewritten)
    if refused:
        logger.warning("rewrite of %r declined: %s", module.name, refused)
        return None
    return rewritten


def _say_why_not(trial: Path, published: Path, marker: str) -> None:
    """Log why a trial published no rewrite: at warning level when the trial failed or wrote
    the wrong files, and at info level when it wrote nothing, which counts as a decline."""
    ended = verdict(trial).ended
    if ended:
        logger.warning("the reflection trial failed: %s; see %s", ended, trial)
    elif left := (sorted(p.name for p in published.iterdir()) if published.is_dir() else []):
        logger.warning("the reflection trial wrote no %r; it left %s; see %s", marker, left, trial)
    else:
        logger.info("the reflection trial wrote no %r under %s", marker, "/".join(COLLECTED))


def unwrapped(text: str) -> str:
    """Strip any wrapper the reflector put around a file: closing tags alone on the last
    lines, a fence pair around the whole text, or a lone fence on the last line. A fence that
    closes a block the text opened is kept."""
    lines = text.splitlines()
    kept = list(lines)
    while kept and kept[-1].strip() in CLOSING_TAGS:
        kept.pop()
    fences = [index for index, line in enumerate(kept) if line.strip().startswith("```")]
    if len(kept) >= 2 and fences and fences[0] == 0 and fences[-1] == len(kept) - 1:
        if len(fences) % 2 == 0:
            kept = kept[1:-1]
    elif fences and fences[-1] == len(kept) - 1 and len(fences) % 2 == 1:
        kept = kept[:-1]
    if kept == lines:
        return text
    return "\n".join(kept) + ("\n" if kept else "")


def _slug(task: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in task).strip("-")
