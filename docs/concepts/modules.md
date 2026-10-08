# Modules

A module is text that a rollout carries into its container. In v1 the only kind of module is the skill:
a document the harness reads before it starts work. A candidate is a set of named modules with one
digest. A rollout carries at most one candidate.

## A module is a directory

A modules directory holds one subdirectory per module. The subdirectory's name is the module's name. A
marker file inside it names the module's kind:

| Marker | Kind | In v1 |
|---|---|---|
| `SKILL.md` | skill | delivered |
| `server.py` | tool | refused |
| `agent.py` | harness | refused |

Every file in the subdirectory travels with the module, at any depth, except dot-files, `__pycache__`
and `.DS_Store`. Every file must be UTF-8 text. A blueprint with one skill:

```
blueprints/05-gepa-docker/
├── run.toml
└── modules/
    └── solving/
        └── SKILL.md
```

The digest is a sha256 prefix computed over every file of every module.

## Skills

A `SKILL.md` must open with a frontmatter block that names the skill and describes it:

```markdown
---
name: solving
description: How to approach a task in this environment before writing anything.
---

Read the instruction twice. State what "done" means in one line before acting.
Run the existing tests first when there are any, and read their output before editing.
Make the smallest change that makes the verifier pass, then re-run it.
```

The first line must be `---`, and a second `---` must close the block. Between them, `name:` and
`description:` must both have values. A harness surfaces a skill only through its frontmatter. Without
one, the file is uploaded and never used. `shipyard check` refuses such a skill:

```
blocked  [recipe] modules: 'solving' cannot be delivered as a skill: it does not open with a `---` frontmatter block, so the harness will upload the file and never surface the skill
blocked  [recipe] modules: 'solving' cannot be delivered as a skill: the frontmatter has no name
```

## How a skill reaches the container

Before each job, shipyard writes every skill to `jobs/<job>/modules/skills/<name>/`. It then names that
root in the `skills` field of Harbor's agent config. Harbor uploads each skill directory into the
container before the harness's setup runs. The job's row in `jobs.jsonl` records the candidate's digest
under `modules`.

The harness must declare skill support in Harbor. Harbor refuses a trial whose harness does not, and
[admission](admission.md) masks that trial. `pi`, `claude-code`, `opencode` and `terminus-2` declare
it; `check` does not test this.

## Carrying modules

`[recipe] modules` names a directory relative to the blueprint. A gradient recipe or `evaluate` carries
it into every rollout of the run:

```toml
[recipe]
kind = "evaluate"
modules = "modules"
```

`shipyard check --verbose` counts the components and prints the digest. For `blueprints/05-gepa-docker`:

```
ok  modules: 1 component(s) under blueprints/05-gepa-docker/modules (f33ca85e3cd4ecf1)
```

The run reads the directory once, when it opens, and copies it to `runs/<id>/modules/carried/`.
Edits to the blueprint's modules during the run do not reach later jobs. The digest on each
job row matches that copy.

`gepa` uses the same key in a different way: there it names the seed of the search. See [gepa](gepa.md).

## Refusals

`check` blocks a modules directory that cannot be delivered and names the module and the reason. Tools
and harnesses are refused by name:

```
blocked  [recipe] modules: 'search' is a tool module, and tool modules are not supported in this version
blocked  [recipe] modules: 'myagent' is a harness module, and harness modules are not supported in this version
```

These cases are refused too:

- A subdirectory with no marker.
- A directory with no subdirectories, or a path that does not exist.
- A file that is not UTF-8 text.

## The Kind protocol

Each kind of module is a class with four members:

| Member | What it does |
|---|---|
| `marker` | Names the file that marks a module of this kind. |
| `check(module)` | Returns why the module cannot be delivered, or `None`. |
| `deliver(modules, into)` | Writes the modules under `into` and returns the agent-config fields that carry them. |
| `reference()` | Returns what the gepa reflector is told about the kind, or `None` for skills. |

The kinds live in the `KINDS` table in `shipyard.modules`. A new kind is a class plus a row in that
table.
