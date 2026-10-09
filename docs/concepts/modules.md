# Modules

A module is text that a rollout carries into its container. In v1 there are two kinds. A skill is a
document the harness can open while it works. A prompt is text placed before every task's
instruction. A candidate is a set of named modules with one digest. A rollout carries at most one
candidate.

## The layout

A modules directory has one layout. A prompt sits at the top, and each skill has a directory under
`skills/`:

```
blueprints/my-run/
├── run.toml
└── modules/
    ├── PROMPT.md
    └── skills/
        └── solving/
            ├── SKILL.md
            └── examples/
                └── worked.md
```

| Path | Kind | Component | In v1 |
|---|---|---|---|
| `PROMPT.md` | prompt | `prompt` | delivered |
| `skills/<name>/`, holding `SKILL.md` | skill | `<name>` | delivered |
| `tools/` | tool | | refused |
| `harness/` | harness | | refused |

Both parts are optional, but the directory must hold at least one module. Nothing else may sit at the
top. A skill takes every file in its directory with it, at any depth, except dot-files, `__pycache__`
and `.DS_Store`. Every file must be UTF-8 text.

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

## Prompts

A prompt is `PROMPT.md` at the top of the modules directory. Its text goes at the start of the
harness's first message, before the task's own instruction, in every task the candidate is carried
into.

A harness decides whether to open a skill, and a model may never do so. A prompt is always read,
because it is part of the task. It also reaches any harness Harbor installs, including those that take
no skills.

A candidate carries at most one prompt, beside any number of skills. When gepa rewrites a prompt,
only `PROMPT.md` is kept.

## How a prompt reaches the container

Before each job, shipyard writes the prompt as a Jinja template to
`jobs/<job>/modules/prompt/template.j2`: the text inside a raw block, then `{{ instruction }}`. It then
names the template in the agent's `prompt_template_path` kwarg. Harbor renders every task's
instruction through it, so the harness receives the text, a blank line, and then the instruction. The
raw block keeps any `{{ }}` or `{% %}` in the text as plain text.

When `[rollout.kwargs]` names its own `prompt_template_path` and the modules hold a prompt, the
prompt's template wins. `check` warns about this:

```
warning  [rollout.kwargs] prompt_template_path: the prompt in the modules is delivered as the prompt template, so this one is not used where the prompt is carried
```

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

`check` blocks a modules directory that cannot be delivered and names the entry and the reason. The
directories kept for tools and harnesses are refused by name:

```
blocked  [recipe] modules: 'tools' is for tool modules, and tool modules are not supported in this version
blocked  [recipe] modules: 'harness' is for harness modules, and harness modules are not supported in this version
```

These cases are refused too:

- Anything else at the top, such as a skill's directory outside `skills/`:
  `'solving' is not part of the layout: a modules directory holds PROMPT.md and skills/<name>/SKILL.md`.
- A directory under `skills/` without a `SKILL.md`.
- A skill named `prompt` beside a `PROMPT.md`, since both would be the component `prompt`.
- A prompt with an empty text, or with `{% endraw %}` in it.
- A directory holding no module, or a path that does not exist.
- A file that is not UTF-8 text.

## The Kind protocol

Each kind of module is a class with six members:

| Member | What it does |
|---|---|
| `marker` | Names the file that marks a module of this kind. |
| `alone` | Whether the marker is the whole module, so nothing beside it is read, kept or delivered. |
| `home(name)` | Returns where a module of this kind lives in the layout, `""` for the top. |
| `check(module)` | Returns why the module cannot be delivered, or `None`. |
| `deliver(modules, into)` | Writes the modules under `into` and returns the agent-config fields that carry them. |
| `reference()` | Returns what the gepa reflector is told about the kind, or `None` when there is nothing to add. |

The kinds live in the `KINDS` table in `shipyard.modules`. A new kind is a class plus a row in that
table.
