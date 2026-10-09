"""Modules: the text a rollout carries into its container. A module is read from a
directory, written back to one, and delivered the way Harbor takes it. This version has
two kinds, skills and prompts. A tool (`server.py`) or harness (`agent.py`) kind would
implement the `Kind` protocol."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

#: Marks a directory as a skill. The directory name is the module name.
SKILL_FILE = "SKILL.md"
#: Marks a directory as a prompt.
PROMPT_FILE = "PROMPT.md"
#: The template a prompt is delivered as, under the job's modules directory.
PROMPT_TEMPLATE = "prompt/template.j2"
#: The end of a Jinja raw block, which a prompt's text may not contain.
END_RAW = re.compile(r"{%-?\s*endraw\s*-?%}")
#: Markers of kinds this version does not deliver, so `seed` can report them as
#: unsupported.
RESERVED = {"server.py": "tool", "agent.py": "harness"}
#: Files a module directory may hold that nobody wrote. `read` skips them.
INCIDENTAL = ("__pycache__", ".DS_Store")


class Inadmissible(ValueError):
    """Raised for a module, or a directory of modules, that cannot be delivered as written."""


class Unsupported(Inadmissible):
    """Raised for a module of a kind this version does not deliver."""

    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        super().__init__(
            f"{name!r} is a {kind} module, and {kind} modules are not supported in this version"
        )


@dataclass(frozen=True)
class Module:
    """One component: its name, its kind, and every file of its directory by relative path."""

    name: str
    kind: str
    files: Mapping[str, str]

    @property
    def text(self) -> str:
        """The marker file's content."""
        return self.files[KINDS[self.kind].marker]

    @property
    def digest(self) -> str:
        """sha256[:16] over every (path, content) in path order. Changing any file changes it."""
        return _digest(sorted(self.files.items()))


@dataclass(frozen=True)
class Candidate:
    """What one rollout carries: components by name, with one digest over all of them."""

    components: Mapping[str, Module]

    @property
    def digest(self) -> str:
        return _digest(
            (name, module.kind, module.digest) for name, module in sorted(self.components.items())
        )

    def with_component(self, name: str, module: Module) -> Candidate:
        """A new candidate with that component replaced or added. The module is renamed to
        `name`, because a kind delivers modules by their name."""
        return Candidate({**self.components, name: replace(module, name=name)})

    def __len__(self) -> int:
        return len(self.components)


class Kind(Protocol):
    """What a kind of module defines: its marker, which modules are admissible, how a set
    of them reaches the container, and what a reflector is told about the kind."""

    marker: str

    def check(self, module: Module) -> str | None: ...

    # Returns the agent-config fields that carry the modules written under `into`, which
    # `merge_fields` merges onto the rollout's own. That suffices for a kind Harbor uploads
    # itself, like `skills`. A kind that places files in the container (a tool's
    # `server.py`) would need a separate route for them.
    def deliver(self, modules: Sequence[Module], into: Path) -> dict[str, Any]: ...

    def reference(self) -> str | None: ...


class Skill:
    """A document the harness reads before it starts work: `SKILL.md` with frontmatter,
    plus any files beside it."""

    marker = SKILL_FILE

    def check(self, module: Module) -> str | None:
        """Require frontmatter with `name` and `description`. Harbor does not check it, and
        the harness needs it to surface the skill."""
        text = module.text.strip()
        if not text:
            return "the text is empty"
        lines = text.splitlines()
        if lines[0].strip() != "---":
            return (
                "it does not open with a `---` frontmatter block, so the harness will "
                "upload the file and never surface the skill"
            )
        try:
            closed = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
        except StopIteration:
            return "the frontmatter block is never closed by a second `---`"
        header = lines[1:closed]
        missing = [
            key
            for key in ("name", "description")
            if not any(
                line.startswith(f"{key}:") and line[len(key) + 1 :].strip() for line in header
            )
        ]
        if missing:
            return f"the frontmatter has no {' or '.join(missing)}"
        return None

    def deliver(self, modules: Sequence[Module], into: Path) -> dict[str, Any]:
        """Write each module's files under `<into>/skills/<name>/` and return that root as
        the only entry of the agent config's `skills`."""
        # `AgentConfig.skills` takes a root of skill directories, each uploaded into the
        # container before the agent's setup. The root is absolute because Harbor reads a
        # relative path that does not exist as a git source to fetch.
        root = (into / "skills").resolve()
        for module in modules:
            write(module.files, into=root / module.name)
        return {"skills": [str(root)]}

    def reference(self) -> str | None:
        """None: a skill has no contract beyond its document."""
        return None


class Prompt:
    """Text placed before the task's instruction in the harness's first message: `PROMPT.md`.
    Harbor's `prompt_template_path` delivers it, so it reaches any harness Harbor installs,
    whether or not that harness reads skills."""

    marker = PROMPT_FILE

    def check(self, module: Module) -> str | None:
        """Require one file of text with no Jinja raw-block end, since the text is delivered
        inside one and nothing else is."""
        if set(module.files) != {PROMPT_FILE}:
            return f"a prompt is its {PROMPT_FILE} alone, and nothing beside it is delivered"
        if not module.text.strip():
            return "the text is empty"
        if END_RAW.search(module.text):
            return "it contains `{% endraw %}`, which would end the block it is delivered in"
        return None

    def deliver(self, modules: Sequence[Module], into: Path) -> dict[str, Any]:
        """Write the template Harbor renders each task's instruction through: the text kept
        raw, then the instruction. Return it as the agent's `prompt_template_path`."""
        if len(modules) > 1:
            names = ", ".join(sorted(module.name for module in modules))
            raise Inadmissible(f"a candidate carries one prompt, and this one has {names}")
        template = (into / PROMPT_TEMPLATE).resolve()
        template.parent.mkdir(parents=True, exist_ok=True)
        text = "{% raw %}" + preamble(modules[0]) + "{% endraw %}{{ instruction }}"
        template.write_text(text, encoding="utf-8")
        return {"kwargs": {"prompt_template_path": str(template)}}

    def reference(self) -> str | None:
        return (
            f"`{PROMPT_FILE}` is given to the assistant at the start of every task, in its first "
            "message, before the task's own instruction. It is the same text for every task, "
            "so it should hold what helps across them, written to the assistant.\n"
        )


def preamble(module: Module) -> str:
    """What a prompt puts before the instruction: its text, then a blank line."""
    return module.text.strip() + "\n\n"


#: The kinds this version delivers, by name. A new kind is a class plus a row here.
KINDS: dict[str, Kind] = {"skill": Skill(), "prompt": Prompt()}
#: Each kind's marker mapped to its name.
MARKERS = {kind.marker: name for name, kind in KINDS.items()}


def seed(directory: Path) -> Candidate:
    """The candidate a directory holds: one component per subdirectory, named after it and
    holding a kind's marker, each checked. A reserved marker, no marker, an empty directory
    or a failed check raises `Inadmissible` naming the component and the reason."""
    directory = Path(directory)
    known = " or ".join(MARKERS)
    if not directory.is_dir():
        raise Inadmissible(
            f"no modules at {directory}: a directory holding one subdirectory per "
            f"component, each with a {known} inside it"
        )
    components = {
        home.name: _component(home)
        for home in sorted(directory.iterdir())
        if home.is_dir() and not home.name.startswith(".")
    }
    if not components:
        raise Inadmissible(f"no components under {directory}: no subdirectory holds a {known}")
    return Candidate(components)


def _component(home: Path) -> Module:
    """The checked module a component directory holds; its marker tells the kind."""
    for marker, kind in RESERVED.items():
        if (home / marker).is_file():
            raise Unsupported(kind, home.name)
    claimed = [kind for marker, kind in MARKERS.items() if (home / marker).is_file()]
    if not claimed:
        raise Inadmissible(f"{home.name!r} holds no {' or '.join(MARKERS)}, so its kind is unknown")
    if len(claimed) > 1:
        raise Inadmissible(f"{home.name!r} is marked as {' and '.join(claimed)}; a module is one")
    module = Module(home.name, claimed[0], read(home))
    refused = KINDS[module.kind].check(module)
    if refused:
        raise Inadmissible(f"{home.name!r} cannot be delivered as a {module.kind}: {refused}")
    return module


def keep(candidate: Candidate, into: Path) -> None:
    """Write the candidate in the layout `seed` reads, `<into>/<name>/<files>`, so that a
    winner can serve as a seed."""
    for name, module in candidate.components.items():
        write(module.files, into=Path(into) / name)


def deliver(candidate: Candidate, into: Path) -> dict[str, Any]:
    """Write every module where Harbor finds it and return the merged agent-config fields.
    Each kind's handler is called once with its modules. `into` is emptied first, so a
    container receives the files of one candidate only."""
    into = Path(into)
    if into.exists():
        shutil.rmtree(into)
    fields: dict[str, Any] = {}
    for kind, handler in KINDS.items():
        mine = [module for module in candidate.components.values() if module.kind == kind]
        if mine:
            merge_fields(fields, handler.deliver(mine, into))
    return fields


def merge_fields(fields: dict[str, Any], more: Mapping[str, Any]) -> None:
    """Merge `more` into `fields` in place. Lists are concatenated, dicts merge key by key
    with `more` winning, and anything else is replaced."""
    for key, value in more.items():
        had = fields.get(key)
        if isinstance(had, list) and isinstance(value, list):
            fields[key] = had + value
        elif isinstance(had, dict) and isinstance(value, Mapping):
            fields[key] = {**had, **value}
        else:
            fields[key] = value


def read(home: Path) -> dict[str, str]:
    """Every file a module directory holds, keyed by its path relative to it. Incidental
    files and dot-files are skipped at every depth. A file that is not UTF-8 text raises
    `Inadmissible`."""
    files: dict[str, str] = {}
    for found in sorted(Path(home).rglob("*")):
        relative = found.relative_to(home)
        if not found.is_file() or any(
            part in INCIDENTAL or part.startswith(".") for part in relative.parts
        ):
            continue
        try:
            files[relative.as_posix()] = found.read_text(encoding="utf-8")
        except UnicodeDecodeError as refused:
            raise Inadmissible(
                f"{relative.as_posix()!r} in {Path(home).name!r} is not UTF-8 text, and a "
                "module is delivered as text"
            ) from refused
    return files


def write(files: Mapping[str, str], *, into: Path) -> None:
    """Write a module's files under `into`. A path that would leave it raises
    `Inadmissible`, because these names come from a reflector's filesystem."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    root = into.resolve()
    for relative, body in files.items():
        found = (root / relative).resolve()
        if found != root and root not in found.parents:
            raise Inadmissible(f"{relative!r} is not inside the module's directory")
        found.parent.mkdir(parents=True, exist_ok=True)
        found.write_text(body, encoding="utf-8")


def _digest(parts: Iterable[tuple[str, ...]]) -> str:
    """sha256[:16] over the strings in order, each followed by a NUL byte, so no two layouts
    collide."""
    hasher = hashlib.sha256()
    for part in parts:
        for piece in part:
            hasher.update(piece.encode("utf-8"))
            hasher.update(b"\0")
    return hasher.hexdigest()[:16]
