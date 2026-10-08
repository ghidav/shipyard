# gepa

`gepa` searches over the text the harness reads. It rewrites one module at a time, keeps the rewrites
that score better, and writes out the best set it found. The weights never move, so it writes no
checkpoint.

This page uses these terms:

- **Candidate.** A set of named [modules](modules.md) with one digest.
- **Seed.** The candidate in the blueprint's modules directory.
- **Feedback tasks.** The tasks each round's minibatch is drawn from, GEPA's D_feedback.
- **Pareto tasks.** The tasks every candidate the search keeps is scored on, GEPA's D_pareto. The frontier and the winner are read from these scores.
- **Pool.** Every candidate the search has accepted. The seed is the first.
- **Frontier.** Every pool candidate that is best on at least one Pareto task, ties included, less the dominated: a candidate leaves when every task it is best on also has another frontier candidate as best.
- **Round.** One proposal: a parent and a child run on one minibatch, one rewritten component, and a judgement on the child.
- **Reflector.** The agent that writes the rewrite.

## The blueprint

```toml
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
pareto = 0
minibatch = 1
patience = 2
```

`shipyard check --verbose` reports the seed, the reflector and how the tasks split:

```
ok  modules: 1 component(s) under blueprints/05-gepa-docker/modules (f33ca85e3cd4ecf1)
ok  reflector claude-code, model claude-sonnet-5-5, image python:3.12-slim
ok  pareto: none held out; 1 task both reflected and selected on
```

The policy can be served by a provider, as it is here, or by this run through the [proxy](proxy.md).
`gepa` runs `group_size` rollouts of every task it scores. Of `[data]` it reads `dataset`,
`group_size` and `seed`. It does not use `batch_size` or `epochs`, although `batch_size` must
still be set. hello-world has one task, so `pareto = 0` scores it both as a feedback task and as a
Pareto task.

## The two task sets

GEPA splits its data in two (Alg. 1 line 1): the feedback tasks, whose minibatches the reflector
learns from, and the Pareto tasks, which score the candidates the search keeps. Parents are drawn,
and the winner is picked, by Pareto scores alone. `pareto` says where the Pareto tasks come from:

| `pareto` | Feedback tasks | Pareto tasks |
|---|---|---|
| unset | the run's tasks less the Pareto tasks | two thirds of the run's tasks, rounded down, drawn with `[data] seed` |
| a count, such as `3` | the run's tasks less the Pareto tasks | that many of the run's tasks, drawn with `[data] seed` |
| `0` | the run's tasks | the same tasks |
| a dataset name, such as `"rg-val"` | the run's tasks | that dataset's tasks |

The default holds out two thirds, as in three of GEPA's four benchmarks: GEPA §4.1 (split sizes)
and §4.3 (the validation set is D_pareto). §4.1 states no single proportion: HotpotQA, IFBench and
HoVer hold 300 validation tasks beside 150 training ones, and PUPA 111 beside 111. IFBench's is the
one split the paper says it drew itself. A run of eight tasks then reflects on 3 and selects on 5:

```
ok  pareto: 5 of 8 tasks held out with seed 0 to select on, 3 to reflect on
```

`pareto = 0` searches and selects on the same tasks, as GEPA does when it is used to solve a fixed
set of tasks (§6). `check` blocks a count that leaves no feedback task, and a default split of a
single task.

## The loop

1. **Seed.** The modules directory is read, copied to `runs/<id>/modules/seed/`, and scored on every Pareto task.
2. **Rounds.** While the budget and patience last, each round does the following:
    1. **Parent.** One candidate is drawn from the frontier, with probability proportional to the number of Pareto tasks it is best on.
    2. **Component.** The next module is taken, round-robin over the seed's components.
    3. **Minibatch.** `minibatch` feedback tasks are drawn. The draws go through the feedback tasks in
       passes, each pass in a new order shuffled with `[data] seed`, so no task comes back before every
       other has had its turn. A pass's last minibatch, when short, is filled from the start of the same pass.
    4. **Parent's run.** The parent is run on the minibatch.
    5. **Reflect.** The reflector is asked for a rewrite of the component and is shown the parent's traces from that run.
    6. **Child's run.** If a new child comes back, it is run on the same minibatch.
    7. **Accept or decline.** If the child's mean beats the parent's strictly, over the tasks both were
       measured on, the child joins the pool and is scored on every Pareto task, all of them afresh.
       Otherwise the child is not accepted, and its text is never scored again.
3. **Winner.** The best candidate is written to `runs/<id>/modules/best/`.

Steps 3 to 7 are GEPA's Alg. 1 (lines 9 to 17): the minibatch is sampled from the feedback tasks,
and the parent is run on it each round, so the reflector reads this round's traces and both means
come from the same round. The paper does not say how the minibatch is sampled; passes in a fresh
shuffle are what the authors' released code does. `minibatch = 3` is the paper's (§4.3). An
accepted child's Pareto scores come from its scoring on the Pareto tasks, never from the minibatch
it won on, even where `pareto = 0` makes a minibatch task a Pareto task too.

A task's score under a candidate is the mean reward over its measured rollouts. A masked rollout is left
out, and a budget cut counts as 0, as admission scores it (see [Admission](admission.md)).

## The reflector

The reflector runs as one Harbor trial per round. Its task is built for that round:

| Part | Holds |
|---|---|
| `/app/current/<name>/` | The component's files. |
| `/app/traces/NN-<task>.txt` | One file per task in the minibatch. Each holds the task, its score (or `unmeasured`), the head of its instruction (1500 characters), the tail of the worst rollout's transcript (3000), and the tail of what the verifier printed (2000). |
| The instruction | The request: improve `SKILL.md` for tasks like these, and write the new file to `/logs/artifacts/SKILL.md`. |

The trial runs `reflection_harness` with `reflection_model` in the image `reflection_image`. It uses the
run's sandbox, is not graded, and has an 1800-second setup timeout. It does not go through
the run's proxy and does not carry the candidate. `jobs.jsonl` records it with `purpose = "reflection"`,
`component`, and `candidate` (the parent's digest). Its tokens are billed to the provider that Harbor reports for the trial.

The rewrite is read back from the trial's artifacts. A code fence or closing tag wrapped around it is
stripped, and the result must pass the [skill frontmatter rule](modules.md#skills). Files the reflector
does not write are dropped from the child. Each of the following is a decline:

- a failed trial
- no file
- a refused rewrite
- an unchanged or blank text

With `edits = "incremental"`, the instruction also asks for the smallest change that keeps the structure
of the text.

## The knobs

| Key | Default | Meaning |
|---|---|---|
| `reflection_harness` | required | The harness the reflector runs as. |
| `reflection_model` | the harness's own default | The model the reflector uses. |
| `reflection_image` | `"python:3.12-slim"` | The reflector's container image. |
| `modules` | `"modules"` | The seed directory, relative to the blueprint. |
| `pareto` | two thirds of the run's tasks, as in three of GEPA's four benchmarks: GEPA §4.1 (split sizes) and §4.3 (the validation set is D_pareto) | The Pareto tasks: a count of the run's tasks to hold out, `0` for none, or a dataset name. See [The two task sets](#the-two-task-sets). |
| `minibatch` | `3` (GEPA §4.3) | The number of feedback tasks in each round's minibatch. |
| `budget` | 2 × tasks × `group_size` (GEPA states no default: §4.3 matches MIPROv2's rollouts per benchmark) | The rollouts the search may spend. The tasks are every one it may score, feedback and Pareto, each counted once. |
| `patience` | `3` (GEPA has none; its loop runs until the budget is spent) | How many rounds in a row may pass with no child to score before the search stops, and never fewer than the number of components. |
| `edits` | `"rewrite"` | Set to `"incremental"` to ask for small edits. |

## When it stops

**Budget.** The budget counts rollouts asked for. Each task scored costs `group_size`: the seed's
scoring on the Pareto tasks, the parent's and the child's runs on each minibatch, and each accepted
child's scoring on the Pareto tasks. The search checks the budget before each round, so the round that
crosses it finishes, including the full scoring of an accepted child.

**Patience.** Patience counts rounds in a row that score no child. That happens when the reflector
declines or fails, or returns a text already in the pool or one not accepted before. Such a round still
spends the parent's run on its minibatch. Any round that scores a child resets the count, whether the
child is accepted or not. With more components than `patience`, the search waits one quiet round per
component instead, so every component is offered to the reflector before the search gives up.

## The winner

The winner comes from the whole pool, by its scores on the Pareto tasks, as GEPA returns the best
average there (Alg. 1 line 21). The candidate measured on the most Pareto tasks wins, and ties go to
the higher mean. Coverage comes first because a child whose full scoring mostly failed has a mean over very
few tasks. The winner is written in the layout of a modules directory, so it can be the next run's seed,
or a training run can carry it.

## The metrics rows

Each round appends one row to `metrics.jsonl`:

| Key | Meaning |
|---|---|
| `round` | The round number, from 1. |
| `parent`, `component` | The parent's digest and the module that was rewritten. |
| `child` | The child's digest, or `null` when no child was scored: the reflector declined or failed, or returned a text already in the pool or not accepted before. |
| `parent_mean`, `child_mean` | Their means on the minibatch, from this round's runs. `child_mean` is `null` when no child was scored. |
| `accepted` | Whether the child joined the pool. |
| `pool`, `frontier` | Their sizes after the round. |
| `spent` | Rollouts spent so far. |

A final row closes the search. It holds `evolution: true`, `rounds`, `spent`, `best` (the winner's
digest), `best_mean` (its mean over the Pareto tasks), `frontier`, `pool`, and `moved`, which says
whether the winner differs from the seed. Each scoring is a job whose row has `batch` set to the
round (0 for the seed) and `modules` set to the candidate's digest.

For a walkthrough, see [Evolve a skill](../tutorials/evolve-a-skill.md).
