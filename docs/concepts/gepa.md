# gepa

`gepa` searches over the text the harness reads. It rewrites one module at a time, keeps the rewrites
that score better, and writes out the best set it found. The weights never move, so it writes no
checkpoint.

This page uses these terms:

- **Candidate.** A set of named [modules](modules.md) with one digest.
- **Seed.** The candidate in the blueprint's modules directory.
- **Pool.** Every candidate the search has accepted. The seed is the first.
- **Frontier.** Every pool candidate that is best on at least one task, ties included, less the dominated: a candidate leaves when every task it is best on also has another frontier candidate as best.
- **Round.** One proposal: a parent, one rewritten component, and a judgement on the child.
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
minibatch = 1
patience = 2
```

`shipyard check --verbose` reports the seed and the reflector:

```
ok  modules: 1 component(s) under blueprints/05-gepa-docker/modules (f33ca85e3cd4ecf1)
ok  reflector claude-code, model claude-sonnet-5-5, image python:3.12-slim
```

The policy can be served by a provider, as it is here, or by this run through the [proxy](proxy.md).
`gepa` scores every task of the named datasets, `group_size` rollouts each. It does not use
`batch_size` or `epochs`, although `batch_size` must still be set.

## The loop

1. **Seed.** The modules directory is read, copied to `runs/<id>/modules/seed/`, and scored on every task.
2. **Rounds.** While the budget and patience last, each round does the following:
    1. **Parent.** One candidate is drawn from the frontier, with probability proportional to the number of tasks it is best on.
    2. **Component.** The next module is taken, round-robin over the seed's components.
    3. **Window.** `minibatch` consecutive tasks are taken, wrapping around the list. The window moves
       on once every component has had a round.
    4. **Parent's score.** The parent is scored on the window. A text is scored only once per task, so
       known outcomes are reused.
    5. **Reflect.** The reflector is asked for a rewrite of the component and is shown the parent's traces on the window.
    6. **Child's score.** If a new child comes back, it is scored on the same window.
    7. **Accept or decline.** If the child's mean beats the parent's strictly, over the tasks both were
       measured on, the child joins the pool and is scored on every task. Otherwise the child is not
       accepted, and its text is never scored again.
3. **Winner.** The best candidate is written to `runs/<id>/modules/best/`.

A task's score under a candidate is the mean reward over its measured rollouts. A masked rollout is left
out, and a budget cut counts as 0, just as in training (see [Admission](admission.md)).

## The reflector

The reflector runs as one Harbor trial per round. Its task is built for that round:

| Part | Holds |
|---|---|
| `/app/current/<name>/` | The component's files. |
| `/app/traces/NN-<task>.txt` | One file per task in the window. Each holds the task, its score (or `unmeasured`), the head of its instruction (1500 characters), the tail of the worst rollout's transcript (3000), and the tail of what the verifier printed (2000). |
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
| `minibatch` | `3` | The number of tasks in each round's window. |
| `budget` | 2 × tasks × `group_size` | The rollouts the search may spend. |
| `patience` | `3` | How many rounds in a row may pass with no child to score before the search stops, and never fewer than the number of components. |
| `edits` | `"rewrite"` | Set to `"incremental"` to ask for small edits. |

## When it stops

**Budget.** The budget counts rollouts asked for. Each task scored costs `group_size`, and the seed's
first scoring counts too. The search checks the budget before each round, so the round that crosses it
finishes, including the full scoring of an accepted child.

**Patience.** Patience counts rounds in a row that score no child. That happens when the reflector
declines or fails, or returns a text already in the pool or one not accepted before. Any round that scores a
child resets the count, whether the child is accepted or not. With more components than `patience`,
the search waits one quiet round per component instead, so every component is offered to the reflector
before the search gives up.

## The winner

The winner comes from the whole pool. The candidate measured on the most tasks wins, and ties go to
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
| `parent_mean`, `child_mean` | Their means on the window. `child_mean` is `null` when no child was scored. |
| `accepted` | Whether the child joined the pool. |
| `pool`, `frontier` | Their sizes after the round. |
| `spent` | Rollouts spent so far. |

A final row closes the search. It holds `evolution: true`, `rounds`, `spent`, `best` (the winner's
digest), `best_mean`, `frontier`, `pool`, and `moved`, which says whether the winner differs from the
seed. Each scoring is a job whose row has `batch` set to the round (0 for the seed) and `modules` set to
the candidate's digest.

For a walkthrough, see [Evolve a skill](../tutorials/evolve-a-skill.md).
