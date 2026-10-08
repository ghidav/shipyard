# Evolve a skill

The `gepa` recipe improves text. It rewrites a skill the harness reads, keeps a rewrite only when it scores better, and ends with the best text it found.

A module is a directory of text a trial carries into its container. This version delivers one kind of module, a skill: a `SKILL.md` the harness reads before it starts work. A candidate is a set of named modules, identified by a digest. The seed is the candidate the search starts from.

!!! note "A real run"
    The skill, rows and costs below come from one run of this blueprint, `gepa-docker__fK3MrF8`. That run took each round's minibatch as the next two tasks in the dataset's order, and took each parent's minibatch scores from the seed's measurement without running the parent. It asked for a rewrite even when the parent scored 1.0 on both minibatch tasks, and its `patience` counted rounds with no child to score. Its rows have no `skipped` column. The `check` output is what `check` prints for this blueprint. The search kept its seed because none of its four rewrites scored better.

## The seed skill

A blueprint's modules live beside its `run.toml`, one directory per module:

```
blueprints/gepa-docker/
├── run.toml
└── modules/
    └── solving/
        └── SKILL.md
```

```markdown title="blueprints/gepa-docker/modules/solving/SKILL.md"
---
name: solving
description: How to answer a reasoning puzzle so the checker accepts it.
---

Read the question twice and name what kind of puzzle it is before solving it.
Work step by step and check each step against the rules the question states.
Give the final answer exactly in the format the question asks for, and nothing after it.
```

The directory name is the module name. `SKILL.md` must open with a `---` frontmatter block holding a non-empty `name` and `description`, which the harness needs to surface the skill. Files beside `SKILL.md` travel with it. A skill without a description is refused:

```
blocked  [recipe] modules: 'solving' cannot be delivered as a skill: the frontmatter has no description
```

See [Modules](../concepts/modules.md).

## The blueprint

```toml title="blueprints/gepa-docker/run.toml"
[model]
name = "Qwen/Qwen3-8B"

[data]
dataset = "rg-small"
batch_size = 4
group_size = 2

[rollout]
harness = "pi@0.85.1"
sandbox = "docker"
concurrency = 4
max_tokens = 4096
max_context = 12288
timeout = 600

[recipe]
kind = "gepa"
reflection_harness = "claude-code"
reflection_model = "claude-sonnet-5-5"
pareto = 0
minibatch = 2
budget = 32
patience = 2
edits = "incremental"
```

The policy is the one from [Train with dapo](train-with-dapo.md): Qwen3-8B served by the run, with pi, on the same eight Reasoning Gym tasks and the same rollout limits. The reflector is the program that rewrites the skill: `reflection_harness` running `reflection_model`, in its own container. Harbor's claude-code reads `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`, and `check` warns when neither is set.

- `modules`, left at its default, is the seed's directory, `modules` beside `run.toml`.
- `pareto = 0` makes the eight tasks both the feedback tasks, which each round's minibatch is drawn from, and the Pareto tasks, which score every candidate the search keeps. gepa's default, as in three of GEPA's four benchmarks, holds two thirds of the tasks out as Pareto tasks: over eight tasks, 5 to select on and 3 to reflect on.
- `minibatch` is how many tasks the parent and a child run on each round. gepa's default is 3.
- `budget` is how many trials of the policy, called rollouts, the search may spend. Unset, it is two passes over the tasks: 2 × tasks × `group_size`, 32 here as well.
- `patience` ends the search after that many rounds in a row in which the best mean on the Pareto tasks did not rise. Unset, the budget alone ends the search, as in GEPA's released code (gepa 0.1.4, `utils/stop_condition.py`: `NoImprovementStopper` runs only when passed in `stop_callbacks`).
- `edits = "incremental"` asks the reflector for the smallest change. `"rewrite"` is the default.

## Check

```console
$ shipyard check blueprints/gepa-docker --verbose
...
[recipe]
kind = "gepa"
reflection_harness = "claude-code"
reflection_model = "claude-sonnet-5-5"
reflection_image = "python:3.12-slim"
modules = "modules"
pareto = 0
minibatch = 2
budget = 32
patience = 2
edits = "incremental"

[checkpoints]
every = 1
ttl_hours = 168.0

ok  recipe gepa
ok  model Qwen/Qwen3-8B, provider tinker (served by this run)
ok  dataset rg-small
ok  harness pi@0.85.1, sandbox docker
ok  dataset rg-small: 8 tasks under tasks/rg-small
ok  modules: 1 component(s) under blueprints/gepa-docker/modules (af0e102e3ce13a51)
ok  reflector claude-code, model claude-sonnet-5-5, image python:3.12-slim
ok  pareto: none held out; 8 tasks both reflected and selected on
ok  serving Qwen/Qwen3-8B on this machine for a docker sandbox
ok  tinker serves Qwen/Qwen3-8B
```

`af0e102e3ce13a51` is the seed's digest. `reflection_image` is the reflector's container image, `python:3.12-slim` unless the blueprint names another. gepa writes no checkpoints. The `[checkpoints]` table is only a default here.

## What a round does

First the seed is measured on the Pareto tasks, here all eight. Then each round:

1. **Parent.** One candidate is drawn from the frontier, with probability proportional to the number of Pareto tasks it is best on. The pool is every candidate the search has accepted, the seed first. The frontier is the pool's candidates that score best on at least one Pareto task.
2. **Minibatch.** `minibatch` tasks are drawn from the feedback tasks, here all eight, in passes shuffled with `[data] seed`.
3. **Parent's run.** The parent is rolled out on the minibatch.
4. **Perfect parent.** If every minibatch task scored 1.0, the round ends here: no child could score higher. A task's score is the mean over its measured rollouts, so every measured rollout of it must score 1.0, and a task with none measured is not perfect. The row says `"skipped": "perfect"`.
5. **Component.** One module of the parent is chosen, round-robin over the rounds that reach this step.
6. **Reflect.** One Harbor trial runs the reflector. It reads the module's files and one trace per minibatch task from the parent's run: the task's instruction, its score, the tail of the policy's transcript and what the grader printed. It writes a new `SKILL.md`. A rewrite that fails the skill frontmatter rule or comes back blank or unchanged, or a trial that writes nothing, is a decline.
7. **Score.** The child is rolled out on the same minibatch.
8. **Accept.** A child whose mean is strictly above the parent's on the minibatch joins the pool and is measured on every Pareto task.

The search stops when it has spent `budget` rollouts of the policy, or, with `patience` set, after that many rounds in a row in which the best mean on the Pareto tasks did not rise. A skipped round counts. Reflection trials do not count against the budget.

The seed's measurement spends 8 tasks × 2 = 16 rollouts. Each round spends 8, the parent and the child on 2 tasks each, so the budget of 32 allows two rounds when each scores a child. A round that scores no child, because its parent was perfect or its reflector declined, costs 4. An accepted child costs 16 more for its measurement on all eight. The run below compared each child with the seed's scores and spent 4 a round, the child's alone, so it had four.

## What the run did

The run took 35 minutes. The seed scored 15 of 16: every rollout passed but one on `graphs-shortest-path`, which spent its whole 4,096-token reply thinking and never wrote an answer. Then came four rounds, each on the next two tasks in the dataset's order. The parent column is the seed's scores on them:

| round | minibatch | parent | child | accepted |
|---|---|---|---|---|
| 1 | prime-factorization, time-intervals | 1.0 | 1.0 | no |
| 2 | number-sequence, countdown | 1.0 | 1.0 | no |
| 3 | puzzle24, shortest-path | 0.75 | 0.5 | no |
| 4 | knights-knaves, zebra-puzzles | 1.0 | 1.0 | no |

Three children tied a parent that already scored 1.0, and a tie is not accepted. gepa skips such a round: the parent's run is spent and no rewrite is asked for. Round 3's child failed both shortest-path rollouts the same way, out of tokens before an answer. The pool never grew past the seed.

The reflector read the traces well. Its first rewrite, under `edits = "incremental"`, kept the seed's three lines and added:

```markdown
When the task says to put the answer in a file (usually `/workspace/answer.txt`), write only the bare final answer there, in the requested format, with no explanation.
Write the file once, after you have finished and checked the reasoning. Do not rewrite it with the same content or re-verify it with extra tool calls; repeating writes has grown the context until the run crashed. After the write succeeds, reply with one short sentence and stop.
Keep your reasoning compact so there is room left to finish.
```

That names pi's habit with this model: calling tools long after the answer is written. Qwen3-8B did not follow it. Under the rewrites most trials still made dozens of calls, and gepa compares scores only. On three of the four minibatches the seed already scored 1.0, so no rewrite could score higher. A search can improve a skill only on tasks the policy fails often enough for a minibatch to show the difference.

## The round rows

`metrics.jsonl` gets one row per round. Round 3's:

```json
{"at": "2026-10-08T00:16:37.983678+00:00", "seq": 3, "round": 3, "parent": "af0e102e3ce13a51", "component": "solving", "child": "78999adda47baaec", "parent_mean": 0.75, "child_mean": 0.5, "accepted": false, "pool": 1, "frontier": 1, "spent": 28}
```

| column | what it is |
|---|---|
| `round` | the round, from 1 |
| `parent`, `component` | the parent's digest and the module rewritten; `component` is null on a skipped round |
| `skipped` | `"perfect"` when the parent scored 1.0 on every minibatch task, so no rewrite was asked for; null otherwise |
| `child` | the child's digest; null when the round was skipped, or the reflector declined or wrote a text already seen |
| `parent_mean`, `child_mean` | both means over the minibatch |
| `accepted` | whether the child joined the pool |
| `pool`, `frontier` | how many candidates each holds after the round |
| `spent` | policy rollouts spent so far |

A last row, marked `evolution`, names the winner:

```json
{"at": "2026-10-08T00:21:42.012475+00:00", "seq": 5, "evolution": true, "rounds": 4, "spent": 32, "best": "af0e102e3ce13a51", "best_mean": 0.9374999985532407, "frontier": 1, "pool": 1, "moved": false}
```

`moved: false` means the seed won. `best_mean` is its 15 of 16.

In `jobs.jsonl`, a job that scored a candidate has `purpose` `rollout` and the candidate's digest under `modules`. A reflection job has `purpose` `reflection`, the `component` it rewrote and the `candidate` it started from. Its tokens are filed under the reflector's provider.

## What the search costs

```json
{
  "parties": {
    "tinker": {
      "trials": 32,
      "input_tokens": 9552742,
      "cache_tokens": 9146880,
      "output_tokens": 115908
    },
    "anthropic": {
      "trials": 4,
      "input_tokens": 290200,
      "cache_tokens": 255080,
      "output_tokens": 3881
    }
  },
  "sandbox": {
    "docker": {
      "trials": 36,
      "seconds": 5570.081405999999
    }
  }
}
```

`tinker` is the policy's 32 rollouts. `anthropic` is the reflector's four trials, as Claude Code reported them to Harbor. The sandbox seconds count both: 32 rollouts and 4 reflections. See [Costs](../concepts/costs.md).

## modules/best

The run keeps the seed under `runs/<id>/modules/seed/` and the winner under `runs/<id>/modules/best/`. Here the two are the same text. The winner is the pool's candidate measured on the most Pareto tasks, then with the highest mean.

Both use the blueprint's own layout, `<module>/SKILL.md`, so the winner can serve as a seed. Copy `modules/best/` into a blueprint's `modules/` to search again from it or to carry it into another recipe:

```toml
[recipe]
kind = "evaluate"
modules = "modules"
```

An `evaluate` run, or a gradient run, carries those modules into every trial, and keeps a copy under `runs/<id>/modules/carried/`. See [gepa](../concepts/gepa.md).
