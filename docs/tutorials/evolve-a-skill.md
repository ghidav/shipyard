# Evolve a skill

The `gepa` recipe improves text, not weights. It rewrites a skill the harness reads, keeps a rewrite only when it scores better, and ends with the best text it found.

A module is a directory of text a trial carries into its container. A skill is the one kind of module this version delivers: a `SKILL.md` the harness reads before it starts work. A candidate is a set of named modules, identified by one digest. The seed is the candidate the search starts from.

!!! note "Illustrative"
    The skill, the blueprint and the `check` output below are real; the page shows no search output.

## The seed skill

A blueprint's modules live beside its `run.toml`, one directory per module:

```
blueprints/05-gepa-docker/
├── run.toml
└── modules/
    └── solving/
        └── SKILL.md
```

```markdown title="blueprints/05-gepa-docker/modules/solving/SKILL.md"
---
name: solving
description: How to approach a task in this environment before writing anything.
---

Read the instruction twice. State what "done" means in one line before acting.
Run the existing tests first when there are any, and read their output before editing.
Make the smallest change that makes the verifier pass, then re-run it.
```

The directory's name names the module. `SKILL.md` must open with a `---` frontmatter block holding a non-empty `name` and `description`, which the harness needs to surface the skill. Files beside `SKILL.md` travel with it. A skill without a description is refused:

```
blocked  [recipe] modules: 'solving' cannot be delivered as a skill: the frontmatter has no description
```

See [Modules](../concepts/modules.md).

## The blueprint

```toml title="blueprints/05-gepa-docker/run.toml"
[model]
name = "claude-sonnet-5-5"
provider = "anthropic"

[data]
dataset = "hello-world"
batch_size = 1
group_size = 2

[rollout]
harness = "claude-code"
sandbox = "docker"
concurrency = 2

[recipe]
kind = "gepa"
reflection_harness = "claude-code"
reflection_model = "claude-sonnet-5-5"
modules = "modules"
minibatch = 1
patience = 2
```

The policy is Claude Code on a provider-served model. The reflector is the program that rewrites the skill: `reflection_harness` running `reflection_model`, in its own container. Harbor's claude-code reads `ANTHROPIC_API_KEY` from the environment, for the policy and the reflector alike.

- `modules` is the seed's directory, relative to the blueprint.
- `minibatch` is how many tasks a child is first judged on.
- `patience` ends the search after that many rounds in a row with no new child.
- `budget`, left unset here, is how many trials of the policy, called rollouts, the search may spend. By default it is two passes over the tasks: 2 × tasks × `group_size`.
- `edits` is `"rewrite"` by default; `"incremental"` asks the reflector for the smallest change.

## Check

```console
$ shipyard check blueprints/05-gepa-docker --verbose
...
[recipe]
kind = "gepa"
reflection_harness = "claude-code"
reflection_model = "claude-sonnet-5-5"
reflection_image = "python:3.12-slim"
modules = "modules"
minibatch = 1
patience = 2
edits = "rewrite"

[checkpoints]
every = 1
ttl_hours = 168.0

ok  recipe gepa
ok  model claude-sonnet-5-5, provider anthropic
ok  dataset hello-world
ok  harness claude-code, sandbox docker
ok  dataset hello-world: 1 task under tasks/hello-world
ok  modules: 1 component(s) under blueprints/05-gepa-docker/modules (f33ca85e3cd4ecf1)
ok  reflector claude-code, model claude-sonnet-5-5, image python:3.12-slim
```

`f33ca85e3cd4ecf1` is the seed's digest. `reflection_image` is the reflector's container image, `python:3.12-slim` unless the blueprint names another. gepa writes no checkpoints; the `[checkpoints]` table is only a default here.

## What a round does

First the seed is measured on every task. Then each round:

1. **Parent.** One candidate is drawn at random from the frontier. The pool is every candidate the search has accepted, the seed first; the frontier is the pool's candidates that score best on at least one task.
2. **Component.** One module of the parent is chosen, round-robin.
3. **Minibatch.** `minibatch` consecutive tasks are chosen, a window that moves along the task list.
4. **Reflect.** One Harbor trial runs the reflector. It reads the module's files and one trace per minibatch task: the task's instruction, its score, the tail of the policy's transcript and what the grader printed. It writes a new `SKILL.md`. A rewrite that fails the skill frontmatter rule or comes back blank or unchanged, or a trial that writes nothing, is a decline.
5. **Score.** The child is rolled out on the same minibatch.
6. **Accept.** A child whose mean is strictly above the parent's on the minibatch joins the pool and is measured on every task.

The search stops when it has spent `budget` rollouts of the policy, or after `patience` rounds in a row with no new child. Reflection trials do not count against the budget.

With this blueprint's one task and `group_size = 2`, the default budget is 2 × 1 × 2 = 4 rollouts. The seed's measurement spends 2, which leaves room for one child's minibatch. A real search wants a dataset with more tasks than this one.

## The round rows

`metrics.jsonl` gets one row per round:

| column | what it is |
|---|---|
| `round` | the round, from 1 |
| `parent`, `component` | the parent's digest and the module rewritten |
| `child` | the child's digest; null when the reflector declined or wrote a text already seen |
| `parent_mean`, `child_mean` | both means over the minibatch |
| `accepted` | whether the child joined the pool |
| `pool`, `frontier` | how many candidates each holds after the round |
| `spent` | policy rollouts spent so far |

A last row, marked `evolution`, names the winner: `rounds`, `spent`, `best` (its digest), `best_mean`, `frontier`, `pool`, and `moved`. `moved: false` means the seed won.

In `jobs.jsonl`, a job that scored a candidate has `purpose` `rollout` and the candidate's digest under `modules`. A reflection job has `purpose` `reflection`, the `component` it rewrote and the `candidate` it started from. Its tokens are filed under the reflector's provider in `costs.json`.

## modules/best

The run keeps the seed under `runs/<id>/modules/seed/` and the winner under `runs/<id>/modules/best/`. The winner is the frontier candidate measured on the most tasks, then with the highest mean.

Both use the blueprint's own layout, `<module>/SKILL.md`, so the winner is a seed. Copy `modules/best/` into a blueprint's `modules/` and either search again from it, or carry it into another recipe:

```toml
[recipe]
kind = "evaluate"
modules = "modules"
```

An `evaluate` run, or a gradient run, carries those modules into every trial, and keeps a copy under `runs/<id>/modules/carried/`. See [gepa](../concepts/gepa.md).
