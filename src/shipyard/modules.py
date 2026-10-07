"""Modules: the text a rollout carries into its container, read off a directory, kept
back to one, and delivered the way Harbor takes it. One kind in this version, skills;
the `Kind` protocol is where a tool (`server.py`) or a harness (`agent.py`) kind goes."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

#: What marks a directory as a skill; the directory's own name names the module.
SKILL_FILE = "SKILL.md"
#: The markers of kinds this version does not deliver, named so `seed` can say so
#: rather than "unknown marker".
RESERVED = {"server.py": "tool", "agent.py": "harness"}
#: What a module directory holds that nobody wrote; skipped when it is read.
INCIDENTAL = ("__pycache__", ".DS_Store")


class Inadmissible(ValueError):
    """A module, or a directory of them, that cannot be delivered as written."""


class Unsupported(Inadmissible):
    """A module of a kind this version does not deliver."""

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
        """The marker file's content: the document a skill is."""
        return self.files[KINDS[self.kind].marker]

    @property
    def digest(self) -> str:
        """sha256[:16] over every (path, content) in path order: any file changed changes it."""
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
        """A new candidate with that component replaced or added, the module named by it:
        a kind delivers by the module's name, and the two must agree."""
        return Candidate({**self.components, name: replace(module, name=name)})

    def __len__(self) -> int:
        return len(self.components)


class Kind(Protocol):
    """What a kind of module knows: its marker, whether a module is admissible, how a set
    of them reaches the container, and what a reflector is told about the kind."""

    marker: str

    def check(self, module: Module) -> str | None: ...

    # The agent-config fields that carry the modules written under `into`, merged onto
    # the rollout's own by `merge_fields`. Enough for a kind Harbor uploads itself, as it
    # does `skills`; a kind that must place files in the container (a tool's `server.py`)
    # needs a route for them beside the fields, which no kind here has asked for yet.
    def deliver(self, modules: Sequence[Module], into: Path) -> dict[str, Any]: ...

    def reference(self) -> str | None: ...


class Skill:
    """A document the harness reads before it starts work: `SKILL.md` with a frontmatter,
    and whatever the directory keeps beside it."""

    marker = SKILL_FILE

    def check(self, module: Module) -> str | None:
        """Frontmatter with `name` and `description`, which Harbor does not check and the
        harness needs to surface the skill."""
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
        """Each module's files under `<into>/skills/<name>/`, and that root as the one
        entry of the agent config's `skills`."""
        # `AgentConfig.skills` takes a root of skill directories, each uploaded into the
        # container before the agent's setup; absolute, since Harbor reads a relative
        # path that does not exist as a git source to fetch.
        root = (into / "skills").resolve()
        for module in modules:
            write(module.files, into=root / module.name)
        return {"skills": [str(root)]}

    def reference(self) -> str | None:
        """None: a skill is a document, and the document is the whole of the contract."""
        return None


#: The kinds this version delivers, by name: a new kind is a class and a row here.
KINDS: dict[str, Kind] = {"skill": Skill()}
#: Each kind's marker to its name, off that table.
MARKERS = {kind.marker: name for name, kind in KINDS.items()}


def seed(directory: Path) -> Candidate:
    """Every component a directory spells, one subdirectory each, named by it and holding
    a kind's marker, each checked. A reserved marker, no marker, an empty directory or a
    failed check raises `Inadmissible` naming the component and the reason."""
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
    """The module a component directory holds, its kind told by the marker, checked."""
    for marker, kind in RESERVED.items():
        if (home / marker).is_file():
            raise Unsupported(kind, home.name)
    claimed = [kind for marker, kind in MARKERS.items() if (home / marker).is_file()]
    if not claimed:
        raise Inadmissible(
            f"{home.name!r} holds no {' or '.join(MARKERS)}, so nothing says what kind of "
            "module it is"
        )
    module = Module(home.name, claimed[0], read(home))
    refused = KINDS[module.kind].check(module)
    if refused:
        raise Inadmissible(f"{home.name!r} cannot be delivered as a {module.kind}: {refused}")
    return module


def keep(candidate: Candidate, into: Path) -> None:
    """The candidate in the layout `seed` reads, `<into>/<name>/<files>`: a winner is a seed."""
    for name, module in candidate.components.items():
        write(module.files, into=Path(into) / name)


def deliver(candidate: Candidate, into: Path) -> dict[str, Any]:
    """Every module written where Harbor finds it, each kind called once with its modules,
    their agent-config fields merged. `into` is emptied first, so no container is handed
    one candidate's text beside another's."""
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
    """Agent-config fields onto others, in place: lists join, dicts merge key by key
    with `more` winning, anything else is replaced."""
    for key, value in more.items():
        had = fields.get(key)
        if isinstance(had, list) and isinstance(value, list):
            fields[key] = had + value
        elif isinstance(had, dict) and isinstance(value, Mapping):
            fields[key] = {**had, **value}
        else:
            fields[key] = value


def read(home: Path) -> dict[str, str]:
    """Every file a module directory holds, by its path relative to it, the incidental
    and the dot-files skipped at every depth; a file that is not UTF-8 text is refused."""
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
    """A module's directory, written whole; a path that would leave it is refused, since
    these names come off a reflector's own filesystem."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    root = into.resolve()
    for relative, body in files.items():
        found = (root / relative).resolve()
        if found != root and root not in found.parents:
            raise Inadmissible(f"{relative!r} is not inside the module's own directory")
        found.parent.mkdir(parents=True, exist_ok=True)
        found.write_text(body, encoding="utf-8")


def _digest(parts: Iterable[tuple[str, ...]]) -> str:
    """sha256[:16] over strings in order, each terminated, so no two layouts collide."""
    hasher = hashlib.sha256()
    for part in parts:
        for piece in part:
            hasher.update(piece.encode("utf-8"))
            hasher.update(b"\0")
    return hasher.hexdigest()[:16]
