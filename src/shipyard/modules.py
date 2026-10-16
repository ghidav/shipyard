"""Modules: the text a rollout carries into its container, such as a prompt and skills.

A modules directory has one layout, `PROMPT.md` and `skills/<name>/SKILL.md`, which `seed`
reads and `keep` writes. Each kind of module is delivered through the Harbor mechanism made
for it. This version supports two kinds, prompts and skills; a tool or harness kind would
implement the `Kind` protocol and claim its reserved directory."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

#: The prompt file at the top of a modules directory, and its component name.
PROMPT_FILE = "PROMPT.md"
PROMPT = "prompt"
#: Directory holding the skills, one subdirectory per skill, named after it.
SKILLS_DIR = "skills"
#: Marker file of a skill's directory.
SKILL_FILE = "SKILL.md"
#: The expected layout, as error messages state it.
LAYOUT = f"{PROMPT_FILE} and {SKILLS_DIR}/<name>/{SKILL_FILE}"
#: Path of the template a prompt is delivered as, relative to the job's modules directory.
PROMPT_TEMPLATE = "prompt/template.j2"
#: Matches the end of a Jinja raw block, which a prompt's text may not contain.
END_RAW = re.compile(r"{%-?\s*endraw\s*-?%}")
#: Top-level directories reserved for kinds this version does not deliver, so `seed` can
#: report them as unsupported.
RESERVED = {"tools": "tool", "harness": "harness"}
#: Generated files a module directory may contain, which `read` skips.
INCIDENTAL = ("__pycache__", ".DS_Store")


class Inadmissible(ValueError):
    """Raised for a module, or a directory of modules, that cannot be delivered as written."""


class Unsupported(Inadmissible):
    """Raised for a module of a kind this version does not deliver."""

    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        super().__init__(
            f"{name!r} is for {kind} modules, and {kind} modules are not supported in this version"
        )


@dataclass(frozen=True)
class Module:
    """One component: its name, its kind, and the text of every file in its directory, keyed
    by relative path."""

    name: str
    kind: str
    files: Mapping[str, str]

    @property
    def text(self) -> str:
        """Content of the module's marker file."""
        return self.files[KINDS[self.kind].marker]

    @property
    def digest(self) -> str:
        """First 16 hex digits of a SHA-256 over every (path, content) pair, in path order."""
        return _digest(sorted(self.files.items()))


@dataclass(frozen=True)
class Candidate:
    """The modules one rollout carries, keyed by component name, with one digest over all
    of them."""

    components: Mapping[str, Module]

    @property
    def digest(self) -> str:
        return _digest(
            (name, module.kind, module.digest) for name, module in sorted(self.components.items())
        )

    def with_component(self, name: str, module: Module) -> Candidate:
        """Return a copy with component `name` added or replaced by `module`.

        The module is renamed to `name`, because a kind delivers modules by name."""
        return Candidate({**self.components, name: replace(module, name=name)})

    def __len__(self) -> int:
        return len(self.components)


class Kind(Protocol):
    """A kind of module. It defines its marker file, where its modules live in the layout,
    whether the marker is the whole module, which modules are admissible, how its modules
    reach the container, and what a reflector is told about the kind."""

    marker: str
    #: Whether the marker is the whole module, so nothing beside it is read, kept or
    #: delivered.
    alone: bool

    # The module's directory under the modules directory, "" for the top.
    def home(self, name: str) -> str: ...

    def check(self, module: Module) -> str | None: ...

    # Returns the agent-config fields that carry the modules written under `into`, each kind
    # its own keys, which the engine lays over the rollout's own. That suffices for a kind
    # Harbor uploads itself, like `skills`. A kind that places files in the container (a
    # tool's `server.py`) would need a separate route for them.
    def deliver(self, modules: Sequence[Module], into: Path) -> dict[str, Any]: ...

    def reference(self) -> str | None: ...


class Skill:
    """A skill: a `SKILL.md` document with frontmatter, plus any files beside it, which the
    harness reads before it starts work."""

    marker = SKILL_FILE
    alone = False

    def home(self, name: str) -> str:
        return f"{SKILLS_DIR}/{name}"

    def check(self, module: Module) -> str | None:
        """Return why the skill cannot be delivered, or None.

        Requires frontmatter with `name` and `description`: Harbor does not check it, and
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
        """Write each module's files under `<into>/skills/<name>/`, and return that root as
        the only entry of the agent config's `skills`."""
        # `AgentConfig.skills` takes a root of skill directories, each uploaded into the
        # container before the agent's setup. The root is absolute because Harbor reads a
        # relative path that does not exist as a git source to fetch.
        root = (into / "skills").resolve()
        for module in modules:
            write(module.files, into=root / module.name)
        return {"skills": [str(root)]}

    def reference(self) -> str | None:
        """Return None: a skill needs no explanation beyond its own document."""
        return None


class Prompt:
    """A prompt: the text of `PROMPT.md`, placed before the task's instruction in the
    harness's first message.

    Harbor's `prompt_template_path` delivers it, so it reaches any harness Harbor installs,
    whether or not that harness reads skills."""

    marker = PROMPT_FILE
    alone = True

    def home(self, name: str) -> str:
        return ""

    def check(self, module: Module) -> str | None:
        """Return why the prompt cannot be delivered, or None.

        A prompt must be `PROMPT.md` alone, since nothing else is delivered, with non-empty
        text that does not end a Jinja raw block, since it is delivered inside one."""
        if set(module.files) != {PROMPT_FILE}:
            return f"a prompt is its {PROMPT_FILE} alone, and nothing beside it is delivered"
        if not module.text.strip():
            return "the text is empty"
        if END_RAW.search(module.text):
            return "it contains `{% endraw %}`, which would end the block it is delivered in"
        return None

    def deliver(self, modules: Sequence[Module], into: Path) -> dict[str, Any]:
        """Write the Jinja template Harbor renders each task's instruction through: the
        prompt text in a raw block, then the instruction. Return its path as the agent's
        `prompt_template_path`. Raises `Inadmissible` for more than one prompt."""
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
    """Return the text a prompt puts before the instruction: its text, then a blank line."""
    return module.text.strip() + "\n\n"


#: Module kinds this version delivers, by name. A new kind is a class plus an entry here.
KINDS: dict[str, Kind] = {"skill": Skill(), "prompt": Prompt()}


def seed(directory: Path) -> Candidate:
    """Read the candidate a modules directory holds, checking each module.

    `PROMPT.md` becomes the component `prompt`, and each `skills/<name>/` holding `SKILL.md`
    becomes the component `<name>`. Raises `Inadmissible`, naming the entry and the reason,
    for anything else at the top level, a reserved directory, a skill without its marker,
    an empty directory, or a module that fails its kind's check."""
    directory = Path(directory)
    if not directory.is_dir():
        raise Inadmissible(f"no modules at {directory}: a directory holding {LAYOUT}")
    components: dict[str, Module] = {}
    for entry in _entries(directory):
        if entry.name in RESERVED:
            raise Unsupported(RESERVED[entry.name], entry.name)
        if entry.name == PROMPT_FILE and entry.is_file():
            components[PROMPT] = _checked(Module(PROMPT, "prompt", {PROMPT_FILE: _text(entry)}))
        elif entry.name == SKILLS_DIR and entry.is_dir():
            for home in _entries(entry):
                if not (home / SKILL_FILE).is_file():
                    raise Inadmissible(
                        f"'{SKILLS_DIR}/{home.name}' holds no {SKILL_FILE}; each entry under "
                        f"{SKILLS_DIR}/ is a skill's directory"
                    )
                if home.name in components:
                    raise Inadmissible(
                        f"'{SKILLS_DIR}/{home.name}' takes the name of the {PROMPT_FILE} component"
                    )
                components[home.name] = _checked(Module(home.name, "skill", read(home)))
        else:
            raise Inadmissible(
                f"{entry.name!r} is not part of the layout: a modules directory holds {LAYOUT}"
            )
    if not components:
        raise Inadmissible(f"no modules under {directory}: it holds neither {LAYOUT}")
    return Candidate(components)


def _entries(directory: Path) -> list[Path]:
    return [
        entry
        for entry in sorted(directory.iterdir())
        if not entry.name.startswith(".") and entry.name not in INCIDENTAL
    ]


def _text(file: Path) -> str:
    try:
        return file.read_text(encoding="utf-8")
    except UnicodeDecodeError as refused:
        raise Inadmissible(
            f"{file.name!r} is not UTF-8 text, and a module is delivered as text"
        ) from refused


def _checked(module: Module) -> Module:
    refused = KINDS[module.kind].check(module)
    if refused:
        raise Inadmissible(f"{module.name!r} cannot be delivered as a {module.kind}: {refused}")
    return module


def keep(candidate: Candidate, into: Path) -> None:
    """Write the candidate in the layout `seed` reads, so that a winner can seed a later
    search. For a kind whose marker is the whole module, only the marker is written."""
    for module in candidate.components.values():
        kind = KINDS[module.kind]
        files = {kind.marker: module.text} if kind.alone else module.files
        write(files, into=Path(into) / kind.home(module.name))


def deliver(candidate: Candidate, into: Path) -> dict[str, Any]:
    """Write every module where Harbor finds it, and return the merged agent-config fields.

    Each kind's handler is called once with all its modules. `into` is emptied first, so a
    container receives the files of one candidate only."""
    into = Path(into)
    if into.exists():
        shutil.rmtree(into)
    fields: dict[str, Any] = {}
    for kind, handler in KINDS.items():
        mine = [module for module in candidate.components.values() if module.kind == kind]
        if mine:
            fields.update(handler.deliver(mine, into))
    return fields


def read(home: Path) -> dict[str, str]:
    """Read every file in a module directory, keyed by relative path.

    Incidental files and dot-files are skipped at every depth. Raises `Inadmissible` for a
    file that is not UTF-8 text."""
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
    """Write a module's files under `into`.

    Raises `Inadmissible` for a path that would escape `into`, because these paths come
    from a reflector's filesystem."""
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
    """Return the first 16 hex digits of a SHA-256 over the strings in order. Each string is
    followed by a NUL byte, so that no two layouts collide."""
    hasher = hashlib.sha256()
    for part in parts:
        for piece in part:
            hasher.update(piece.encode("utf-8"))
            hasher.update(b"\0")
    return hasher.hexdigest()[:16]
