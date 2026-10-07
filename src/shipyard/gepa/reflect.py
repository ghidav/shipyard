"""The reflector as a Harbor trial: a synthetic task holding the component's files and one
trace per outcome, an agent told to write the rewrite to `/logs/artifacts/<marker>`, and
the artifact read back as a `Module` that the kind's `check` admits or declines."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard.admit import verdict
from shipyard.cost import billed_by
from shipyard.gepa.fitness import Outcome
from shipyard.gepa.propose import Proposer, Reflection
from shipyard.modules import KINDS, Module, read, write
from shipyard.rollout import Rollouts, rollout

if TYPE_CHECKING:
    from shipyard.run import Run

logger = logging.getLogger(__name__)

#: Where the reflector writes the new version: Harbor's conventional publish directory,
#: collected into every trial's `artifacts/logs/artifacts/` without being declared
#: (harbor/models/trial/paths.py, `EnvironmentPaths.artifacts_dir`).
PUBLISH = "/logs/artifacts"
COLLECTED = ("artifacts", "logs", "artifacts")
#: The reflector's harness installs itself into an image that carries none: Harbor's
#: 360 s is six short of what a Node harness takes on `python:3.12-slim`.
SETUP_TIMEOUT = 1800.0
#: In the container's workdir: the component's directory, the traces, the kind's contract.
WORKDIR = "/app"
CURRENT = "current"
TRACES = "traces"
REFERENCE = "reference.md"
TASK_NAME = "shipyard/reflection"
#: Closing tags a reflector leaves on the last line of a file that are not the file.
CLOSING_TAGS = ("</content>", "</file>", "</document>")


def reflect(
    run: Run, *, harness: str, model: str | None, image: str, incremental: bool = False
) -> Proposer:
    """A `Proposer` running one trial per proposal under `run`: the reflection `harness`
    answering from `model` in `image`, asked for the smallest change when `incremental`.
    Every failure is a decline, since losing a search to one container would be worse."""
    if not harness.strip():
        raise ValueError(
            "the reflector needs a harness: Harbor reads an unnamed agent as its oracle, and "
            "the reflection task has no reference solution"
        )

    async def reflector(reflection: Reflection) -> Module | None:
        home = Path(tempfile.mkdtemp(prefix="shipyard-reflection-"))
        try:
            task = task_for(reflection, home, image=image, incremental=incremental)
            rolled = await _job(run, task, harness=harness, model=model, about=reflection)
            return written(rolled, reflection.module)
        except Exception:
            logger.exception("the reflection trial for %r failed", reflection.component)
            return None
        finally:
            shutil.rmtree(home, ignore_errors=True)

    return reflector


async def _job(
    run: Run, task: Path, *, harness: str, model: str | None, about: Reflection
) -> Rollouts:
    """One job under the run that is not a rollout of the policy: through `rollout` and
    not `Run.sample`, so nothing points it at the run's proxy, delivers the candidate to
    it or bills it as sampling; recorded as a `reflection` under the provider's party."""
    cfg = run.config
    job = run.reserve_job()
    rolled = await rollout(
        [task],
        job,
        harness=harness,
        model=model,
        sandbox=cfg.rollout.sandbox,
        concurrency=1,
        rollouts=1,
        jobs_dir=run.jobs_dir,
        verify=False,
        setup_timeout=SETUP_TIMEOUT,
        run_trial=run.run_trial,
    )
    run.record_job(
        rolled,
        purpose="reflection",
        party=billed_by(rolled) or party_of(model),
        component=about.component,
        candidate=about.candidate.digest,
    )
    return rolled


def party_of(model: str | None) -> str:
    """Whose bill a reflection is when Harbor recorded no provider: the model's prefix."""
    if model and "/" in model:
        return model.split("/", 1)[0]
    return "unknown"


def task_for(reflection: Reflection, home: Path, *, image: str, incremental: bool) -> Path:
    """The reflection as a Harbor task under `home`: `task.toml` naming the prebuilt
    image (so `environment/` is uploaded into the workdir), the component's files under
    `current/<name>/`, a trace per outcome, the kind's reference, and the instruction."""
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
        ),
        encoding="utf-8",
    )
    return task


def _trace(outcome: Outcome) -> str:
    """One attempt as the reflector reads it: the task and its score, what was asked,
    the tail of the transcript, and what the grader said."""
    score = "unmeasured" if outcome.reward is None else f"{outcome.reward:.3f}"
    body = f"task: {outcome.task}\nscore: {score}\n\n"
    if outcome.inputs.strip():
        body += f"--- asked ---\n{outcome.inputs.strip()}\n\n"
    if outcome.transcript.strip():
        body += f"--- transcript ---\n{outcome.transcript.strip()}\n\n"
    return body + f"--- observed ---\n{outcome.feedback}\n"


def instruction(
    *, name: str, marker: str, beside: list[str], reference: bool, incremental: bool
) -> str:
    """What the reflector is asked: the goal, where everything is, that those directories
    are all there is, to keep what works, and where to write the answer."""
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
        f"Write the new `{marker}` to `{PUBLISH}/{marker}`. Whatever else you want kept goes "
        f"beside it under `{PUBLISH}`, at the same relative paths it has now; a file you do "
        "not write is a file the next assistant will not have. What lands there is given "
        "verbatim to the next assistant."
    )


def written(rolled: Rollouts, module: Module) -> Module | None:
    """The rewrite the trial published, as a module of the same name and kind, or None:
    no marker under the artifacts (one directory holding it is read, two are not), or
    a rewrite the kind's `check` refuses, each logged with its cause."""
    if not rolled.trials:
        return None
    trial = Path(rolled.trials[0])
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
    files = read(published)
    files[marker] = unwrapped(files[marker])
    rewritten = Module(module.name, module.kind, files)
    refused = kind.check(rewritten)
    if refused:
        logger.warning("rewrite of %r declined: %s", module.name, refused)
        return None
    return rewritten


def _say_why_not(trial: Path, published: Path, marker: str) -> None:
    """A trial that died is said at warning, one that wrote the wrong thing too; one that
    read the traces and wrote nothing has declined, which the search counts."""
    ended = verdict(trial).ended
    if ended:
        logger.warning("the reflection trial failed: %s; see %s", ended, trial)
    elif left := (sorted(p.name for p in published.iterdir()) if published.is_dir() else []):
        logger.warning("the reflection trial wrote no %r; it left %s; see %s", marker, left, trial)
    else:
        logger.info("the reflection trial wrote no %r under %s", marker, "/".join(COLLECTED))


def unwrapped(text: str) -> str:
    """The file without a wrapper the reflector put around it: a closing tag alone on the
    last line, a fence pair around the whole text, or a lone fence on the last line. A
    fence closing a block the text opened is the text's own."""
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
    """A task reference as something that can be a filename."""
    return "".join(character if character.isalnum() else "-" for character in task).strip("-")
