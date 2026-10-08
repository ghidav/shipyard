# Recipes

A recipe is what a run does with its rollouts. You choose it by name in `[recipe] kind`. Below, a
**group** is one task's rollouts in a batch, and its **spread** is the population standard deviation
of their rewards.

| `kind` | What it does | Weights move |
|---|---|---|
| `dapo` | Gradient. The advantage is divided by the group's spread, and ppo clips the ratio asymmetrically. | yes |
| `dr-grpo` | Gradient. The advantage is not divided, and ppo clips symmetrically. It has an optional length rule. | yes |
| `cispo` | Gradient. The advantage is formed as in `dapo`, and the loss is Tinker's `cispo`. | yes |
| `gepa` | Searches over the text the harness reads. See [gepa](gepa.md). | no |
| `fst` | Fast-slow training: gepa and a gradient recipe in cycles. See [fst](#fst). | yes |
| `evaluate` | Measures the policy. | no |

The first three are the gradient recipes, and `fst` steps like them. They train the weights the run serves, so they need
`[model] provider = "tinker"`, which is the default. With any other provider, the run stops before it trains:

```
ValueError: [recipe] kind = 'dapo' trains the weights this run serves, and [model] provider = 'anthropic' serves them elsewhere
```

## A gradient recipe is a name

Each gradient recipe is a preset. The name fixes how the advantage is formed, which loss runs, how
it clips, how its token losses add up and the optimizer that steps. You can tune its knobs, but you cannot combine the parts
another way. The defaults come from each recipe's paper.

`shipyard check` prints the resolved config with every default filled in. After it comes one comment
line that says what the name means. Take a blueprint with this recipe table and no `[checkpoints]` table:

```toml
[recipe]
kind = "dapo"
learning_rate = 2e-5
```

Its resolved config, as `check` prints it, ends like this:

```
[recipe]
kind = "dapo"
learning_rate = 2e-05
substeps = 16
reference = "trainer"
kl_coef = 0.0
clip_low = 0.2
clip_high = 0.28
refill = 9
overlong_penalty = 0.5
overlong_buffer = 0.2

[checkpoints]
every = 1
ttl_hours = 168.0
# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28, averaged per prompt; overlong penalty up to 0.5 over the last 20% of the token budget; 16 substeps by prompt; adamw betas 0.9 / 0.95, eps 1e-08, weight decay 0.1, gradient norm clipped at 1.0, no warm-up (the paper's is 20 steps); degenerate groups dropped and refilled from the plan, up to 9 more rounds
```

Each name resolves as follows, with the section of its paper each default comes from:

| | `dapo` | `dr-grpo` | `cispo` |
|---|---|---|---|
| Paper | DAPO, arXiv 2503.14476 | Dr. GRPO, arXiv 2503.20783 | MiniMax-M1, arXiv 2506.13585 |
| Advantage | (r − mean) / (spread + 1e-6), Eq. 9 | r − mean, §3.2 | (r − mean) / (spread + 1e-6), Eq. 2 |
| Tinker loss | `ppo` | `ppo` | `cispo` |
| Token losses | averaged per prompt, Eq. 8 | summed, §3.2 | averaged per prompt, Eq. 4 |
| Clipping | `clip_low = 0.2`, `clip_high = 0.28`, §4.1 | `clip = 0.2`, App. G | `clip_high = 3.0`, from ScaleRL (arXiv 2510.13786) App. A.17.2 and FST App. D |
| `substeps` | `16`, §4.1 | `1`, not stated (App. G) | `16`, §3.1 |
| `refill` | `9`, Alg. 1, capped as in DAPO's released recipe | `0` | `9`, §3.1, which takes DAPO's |
| Overlong term | `overlong_penalty = 0.5`, `overlong_buffer = 0.2`, Eq. 13 and §4.1 | none, as the paper has none | as `dapo`, §3.1, which takes DAPO's |
| Length rule | none | shipyard's own, off: `length_penalty = 0.0` | none |
| Optimizer | AdamW, weight decay 0.1, gradient norm clipped at 1.0, DAPO's released recipe | AdamW, betas 0.9 / 0.95, gradient norm clipped at 1.0, App. G | AdamW, betas 0.9 / 0.95, eps 1e-15, §3.2 |
| `loss_fn_config` sent | `clip_low_threshold = 0.8`, `clip_high_threshold = 1.28` | `0.8` and `1.2` | `0.0` and `4.0` |

At shipyard's batch sizes the split often reaches one prompt group per substep: a batch whose 4 groups
carry a gradient takes 4 substeps, not 16, each on one group, at the recipe's `learning_rate`.

The clipping knobs are epsilons, but Tinker takes bounds, so `clip_low = 0.2` reaches it as a lower bound
of 0.8. `cispo` truncates the importance weight above `1 + clip_high` and has no lower bound. MiniMax-M1
tunes only its upper epsilon and does not publish it; ScaleRL finds ceilings of 4, 5 and 8 alike, and
FST uses 4. Dr. GRPO does not state how many optimizer steps it takes per rollout batch (App. G lists
one inner update epoch and no mini-batch size), so `dr-grpo` takes one. `check` prints these lines:

```
# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28, averaged per prompt; overlong penalty up to 0.5 over the last 20% of the token budget; 16 substeps by prompt; adamw betas 0.9 / 0.95, eps 1e-08, weight decay 0.1, gradient norm clipped at 1.0, no warm-up (the paper's is 20 steps); degenerate groups dropped and refilled from the plan, up to 9 more rounds
# dr-grpo: advantage = group mean, not divided by spread; loss = ppo, clip 0.2 / 0.2, summed over tokens; 1 substep; adamw betas 0.9 / 0.95, eps 1e-08, gradient norm clipped at 1.0; degenerate groups dropped
# cispo: advantage = group mean, divided by spread; loss = cispo, weight truncated above 4.0, no lower bound, averaged per prompt; overlong penalty up to 0.5 over the last 20% of the token budget; 16 substeps by prompt; adamw betas 0.9 / 0.95, eps 1e-15; degenerate groups dropped and refilled from the plan, up to 9 more rounds
```

### The overlong term (dapo and cispo)

DAPO shapes the reward of a response that runs near or past its length limit with its soft overlong
punishment (§3.4, Eq. 13). Up to `L_max − L_cache` tokens nothing changes; from there the penalty
grows linearly to −1 at `L_max`, and it is added to the correctness reward. DAPO sets `L_max` to 20,480
tokens and `L_cache` to 4,096 (§4.1). §3.4 also describes overlong filtering, which masks
truncated samples. DAPO's Table 1 lists overlong filtering as one step of its ablation; the authors'
release says the best run does not use it (verl `recipe/dapo` README). The overlong term here is the
punishment alone. MiniMax-M1 uses DAPO's length penalty with CISPO (§3.1).

In shipyard the limit is the trial's token budget: a trial may sample `max_context` tokens in all,
across its calls ([The proxy](proxy.md#the-token-budget-and-the-context)). A trial's length is the
tokens the policy sampled over all its calls, the same count the budget is spent by. A rollout that
sampled L tokens of a budget B loses

```
overlong_penalty × min(max(L − (1 − overlong_buffer) × B, 0) / (overlong_buffer × B), 1)
```

from its reward before the advantage is formed. The defaults are DAPO's:

- `overlong_buffer = 0.2` is `L_cache / L_max`, 4,096 of 20,480 tokens.
- `overlong_penalty = 0.5` is DAPO's −1 on its reward of −1 or 1 (Eq. 7), carried to Harbor's reward
  of 0 to 1. Every reward difference halves, so the advantage, divided by the spread, is the same.

A trial cut by the budget has sampled all of it. Admission scores it 0 ([rule 4](admission.md#masked-or-zero)),
and the term takes it to −0.5. A group is judged degenerate before the term, on its rewards alone,
as DAPO keeps a prompt by its accuracy (Eq. 11): failures that differ only in how far they ran carry
no gradient. `evaluate` and `gepa` measure without the term, so a budget cut is 0 there.
`overlong_penalty = 0` turns the term off. A proxy that knows no context length enforces no budget,
and a remote proxy may not report the one it enforces. Either way the run has no budget to dock
against: the term is off for that run, and the run logs one warning saying so.

`dr-grpo` and `fst` have no overlong term. Dr. GRPO states none (App. G). FST follows ScaleRL, which
controls length by interrupting long generations rather than by a reward term (ScaleRL §2, App.
A.10). Under both, a budget cut scores 0.

### The length rule (dr-grpo only)

The length rule is shipyard's own. Dr. GRPO controls length through its two removals alone, the
division by the response length and by the spread (§3.2, App. C), and docks no answer for its
length. `length_penalty` is `0.0`, off, by default.

`length_penalty` docks solved answers for their length. A solved answer has reward 1.0, and the rule
applies only in a group with at least two of them. Each solved reward loses
`length_penalty × max(L − length_floor, 0) / mean solved L`, where L is the number of tokens the policy
wrote, and at most `length_cap`, `0.5` by default. So a solved answer never scores below
`1 − length_cap`, nor below any failure scored that or less. `length_cap = inf` lifts the cap.
`check` warns when `length_cap` is 1 or more and `length_penalty` is above 0, since a long solved
answer could then score as low as a failure, or lower.

```
# dr-grpo: advantage = group mean, not divided by spread; loss = ppo, clip 0.2 / 0.2, summed over tokens; length penalty 0.1 over 512 tokens among solved answers, capped at 0.5; 1 substep; adamw betas 0.9 / 0.95, eps 1e-08, gradient norm clipped at 1.0; degenerate groups dropped
```

### The optimizer

Each recipe steps with AdamW as its paper sets it. Where the paper states a value, it is the paper's;
where it states none, it is the authors' released recipe's when that sets one (DAPO's weight decay and
gradient clipping), else the cookbook's (`train_step` in `tinker_cookbook/rl/train.py`): betas
0.9 / 0.95, eps 1e-8, no weight decay and no gradient clipping. The warm-up is the `warmup` key's,
`0` by default: the papers' warm-ups are sized for runs of hundreds of steps, and a shipyard run of
twenty steps would spend all of them below `learning_rate`.

| | betas | eps | weight decay | gradient clipping | the paper's warm-up |
|---|---|---|---|---|---|
| `dapo` | 0.9 / 0.95, cookbook's | 1e-8, cookbook's | 0.1, DAPO's released recipe | global norm 1.0, DAPO's released recipe | 20 steps, §4.1 |
| `dr-grpo` | 0.9 / 0.95, App. G | 1e-8, cookbook's | 0, App. G | global norm 1.0, App. G, Table 6 | none: a constant rate, App. G |
| `cispo` | 0.9 / 0.95, MiniMax-M1 §3.2 | 1e-15, §3.2 | 0, cookbook's | none: the paper states none | none stated |
| `fst` | 0.9 / 0.999, App. D | 1e-8, cookbook's and PyTorch's | 0, App. D | none: the paper states none | 10 steps, App. D |

`check` names the paper's warm-up beside a run that takes none: "no warm-up (the paper's is 20 steps)".

Gradient clipping scales an optimizer step's gradient down to the given global norm when it is
larger; Tinker takes the norm as `grad_clip_norm`. MiniMax-M1 sets eps to 1e-15 because most of its
gradients are below 1e-14 (§3.2).

With `warmup = N`, the learning rate rises linearly over the run's first N updates: update k, counted
from 0, uses `learning_rate × (k + 1) / N`, and every update from the N-th on uses `learning_rate`.
All substeps of a step use the same rate, and the row's `learning_rate` is the rate applied. A step
that trains nothing takes no update, so it does not move the warm-up on.

A run from `from_checkpoint` with `[model] restore_optimizer = true` skips any warm-up: its
optimizer state continues the earlier run's, and every update uses `learning_rate`. A run from a
checkpoint without it, or from the base model, warms up when `warmup` is set.

The optimizer is part of the name, not a key; `check` prints it on the comment line.

### How token losses add up

Tinker's `ppo` and `cispo` losses sum the token losses of every datum. Summed, a prompt counts in
proportion to how many tokens its rollouts wrote. DAPO (Eq. 8), MiniMax-M1 (Eq. 4) and FST (Eq. 4)
instead divide each prompt's token losses by that prompt's tokens, so every prompt weighs the same.
`dapo`, `cispo` and `fst` do this in the advantage: credit multiplies each token's advantage by

```
(mean target tokens per group in the step) / (target tokens of its group)
```

Every prompt's tokens then carry the same total weight, and the step's total stays that of the plain
sum. The KL term, which is folded into the advantage, is scaled with it, so `kl_coef` weighs the KL
against the reward the same way in every prompt, as in a loss whose two terms are aggregated alike.
`dr-grpo` keeps Tinker's sum. Its §3.2 divides by a constant instead of a prompt's length, and a
constant only rescales the learning rate.

### Substeps

`substeps` is the number of optimizer steps a batch is split into. The papers' mini-batches are sets
of prompts, so the batch is split by prompt: each substep holds whole groups, cut in batch order, and
there are never more substeps than groups that carry a gradient. A batch of four such groups at the
default 16 takes four substeps.

Every substep forms its ratio against μ, the logprobs of the weights the batch was sampled at. The first
substep runs on those same weights, so with `reference = "trainer"` its ratio is 1. From the second on,
the weights have moved, the ratio moves with them, and the clip bounds act. With `substeps = 1` and
`reference = "trainer"`, the ratio is 1 on every token and the bounds never act.

## The common knobs

| Key | Default | Meaning |
|---|---|---|
| `learning_rate` | required, > 0 | AdamW's step size, after the warm-up if any (see [The optimizer](#the-optimizer)). |
| `warmup` | `0` | Updates over which the learning rate rises linearly to `learning_rate`. DAPO warms up over 20 (§4.1) and FST over 10 (App. D). |
| `substeps` | `16` for `dapo` and `cispo`, `1` for `dr-grpo` and `fst` | Optimizer steps per batch, split by prompt (see [Substeps](#substeps)). |
| `reference` | `"trainer"` | Where μ comes from. μ is the sampling logprobs that the ratio is formed against. `"trainer"` recomputes μ in one forward pass on the training engine. `"sampler"` reads the logprobs the proxy recorded and makes no extra pass. |
| `kl_coef` | `0.0`; `0.001` for `fst` | A per-token penalty for drifting from the run's starting weights, which are the base model or `from_checkpoint`. It is subtracted from the advantage: `kl_coef × (μ − anchor)`, token by token, not centred on the step's mean. DAPO (§2.3), Dr. GRPO (App. G) and MiniMax-M1 (§3.1) train without one; FST uses 0.001 (App. D) and does not say how the term enters the loss. |
| `modules` | unset | A directory of skills to carry into every rollout. See [Modules](modules.md). |

`"trainer"` is the default because the sampler and the trainer are different engines and their logprobs
differ slightly. `"sampler"` saves the pass, but it refuses a batch whose records carry no logprobs.
With `kl_coef = 0.01`, the comment line gains `; kl 0.01 to the starting weights`.

## How a step runs

Each step starts on the plan's next batch, and steps are numbered from 0 (see [Datasets](datasets.md)).

1. **Publish.** The current weights are saved for sampling as `sample-<step>` and kept for 12 hours.
2. **Point.** The [proxy](proxy.md) is told to serve those weights.
3. **Sample.** One Harbor job runs over the batch, each task `group_size` times. Every model call leaves
   a record. With `refill`, the step samples the plan's next batches while fewer than `batch_size`
   groups carry a gradient (see [Dynamic sampling](#dynamic-sampling)).
4. **Pack.** Each rollout's records become token sequences by the token-prefix rule. A call joins the
   sequence whose last prompt plus completion begins its prompt, the longest such. If none matches, the
   call starts a new sequence. Each completion is a training target exactly once. Failed
   calls and calls carrying images never pack.
5. **Credit.** Rollouts are grouped by task, and only the measured ones count (see [Admission](admission.md)).
   Degenerate groups are dropped, then the advantages, μ, the KL term and each prompt's weight are computed.
6. **Step.** `forward_backward` and `optim_step` run on Tinker once per substep.
7. **Log.** One row is appended to `metrics.jsonl`.
8. **Checkpoint.** When `step + 1` is divisible by `[checkpoints] every`, `step-<step + 1>` is saved. After the last step, `final` is saved with no expiry.

A checkpoint saves a state path, from which training can continue, and a sampler path, from which a
policy can be sampled. Both are kept for `ttl_hours`, which is at least 1 hour. Each checkpoint appends a row
to `checkpoints.jsonl` with `tag`, `state_path`, `sampler_path` and `ttl_hours`.

### Degenerate groups

A group is degenerate when only one of its rollouts was measured, or when its measured rewards are all
equal after `dr-grpo`'s length rule. The overlong term of `dapo` and `cispo` comes after this judgement.
A degenerate group carries no gradient, so credit drops it before any reference pass. As a result, `group_size = 1` never trains, and never refills. If no group of a
step is left, the step logs `trained: false` and takes no gradient and no checkpoint. Under `dapo` and
`cispo`, or `dr-grpo` without a length penalty, a policy that solves every task, or fails every one,
gives flat groups and so takes no step.

### Dynamic sampling

DAPO keeps sampling until its batch holds only prompts that carry a gradient (§3.2 and Alg. 1), and
MiniMax-M1 does the same for CISPO (§3.1). `refill` is the most extra sampling rounds a step may take
to get there:

| Recipe | `refill` | Why |
|---|---|---|
| `dapo`, `cispo` | `9` | DAPO's released recipe stops at ten generation batches a step (verl `recipe/dapo`, `max_num_gen_batches = 10`): the first and nine more. |
| `dr-grpo` | `0` | Dr. GRPO states no dynamic sampling (App. G). |
| `fst` | not a key | FST follows ScaleRL (§2), whose zero-variance filtering drops flat groups without resampling (ScaleRL §3.2); a slow step also stays inside its cycle's lookahead. |

When credit would leave fewer than `batch_size` groups with a gradient, the step samples the plan's next
batch at the same weights, as one more Harbor job, and adds its groups. The tasks it takes are consumed,
as DAPO consumes its dataloader: the next step starts after them, so a run that refills has fewer steps
than batches. The step stops sampling when `batch_size` groups carry a gradient, when `refill` rounds
are spent, or when the plan runs out, and then trains on what it has.

A round can bring more groups with a gradient than the step needs. The first `batch_size` of them, in
the order they were sampled, are trained on; the rest are **surplus**, left out before the reference pass, so every step
trains on the same number of prompts (DAPO §3.2). Surplus rollouts are graded and paid for like any
other: every round's job has its row in `jobs.jsonl`, with the step's number as its `batch`, and its
tokens in `costs.json`. `train_tokens` and `reference_tokens` count what was trained on.

`refill = 0` switches it off: degenerate groups are dropped and the step trains on what its own batch
left. The row counts the extra rounds as `refills` and their rollouts as `refill_rollouts`, which
`rollouts` includes.

### The pinned-weights refusal

Under `reference = "trainer"`, μ must come from the weights the rollouts were sampled at. Each job is
stamped with the trainer's update count, and credit refuses a batch if that count has moved since:

```
this run applied 1 update(s) between sampling task <task> and crediting it, so the trainer no longer holds the weights those rollouts came from; credit a batch before training on it
```

Each record also names the weights that served it. A batch served by weights other than the ones the run
pointed the proxy at stops the run.

## The metrics row

Each step appends one row, with its keys in this order. A measure with nothing behind it is left out,
never written as 0. Every row also carries `at` and `seq`.

| Key | Meaning |
|---|---|
| `step` | The step's number, from 0. |
| `trained` | Whether a gradient was taken. |
| `groups`, `rollouts` | Groups (tasks) and rollouts in the step, refills included. |
| `refills`, `refill_rollouts` | Extra sampling rounds, and the rollouts they added; present when the step could refill: `refill > 0` and `group_size > 1`. |
| `graded`, `masked` | Rollouts with a reward, and those left out with a mask. |
| `degenerate` | Groups dropped as degenerate. |
| `surplus` | Groups with a gradient left out once `batch_size` were kept; present when the step could refill. |
| `sequences`, `sequences_per_rollout` | Sequences trained on, and their mean per credited rollout. |
| `train_tokens` | Tokens sent to `forward_backward`. |
| `reward_mean`, `reward_spread` | The mean reward over graded rollouts, and the mean of the groups' spreads. |
| `length_mean` | Mean tokens the policy wrote per graded rollout. |
| `kl_v1`, `kl_v2` | mean(μ − π) and half its mean square over the trained tokens, where π is the training pass's logprobs. |
| `entropy` | mean(−μ) over the trained tokens. |
| `anchor_kl` | mean(μ − anchor), present only when `kl_coef > 0`. |
| `overlong` | Graded rollouts that sampled into the last `overlong_buffer` of their token budget; present under `dapo` and `cispo` when the proxy reports a budget. |
| `learning_rate`, `substeps`, `loss_fn` | As applied, `learning_rate` after any warm-up; `substeps` never exceeds the groups trained on. |
| `seconds` | The time taken to apply the gradient. |

## fst

`fst` is fast-slow training (Tiwari, Sareen, Agrawal et al., arXiv 2605.12484). The weights and a
population of `population` skill texts change together, in cycles:

1. **Fast phase.** The run takes the next `cycle` batches. [gepa](gepa.md) runs on their tasks
   against the current weights, seeded with the previous population, and the top `population` of its
   frontier become the new population. The first cycle starts from the blueprint's modules directory.
   Those tasks, the paper's anchor set, are the Pareto tasks. FST names one anchor set (§3, App. A)
   and no other source for the minibatches, so fst draws them from it too, in cycle `c` with
   `[data] seed + c`.
2. **Slow phase.** `cycle` gradient steps follow, one per batch. Each task's group of `group_size`
   rollouts is split evenly across the population: `group_size / population` rollouts per text, one
   job per text. The rollouts are normalised as one group, so the advantage compares what the text
   did and what the sampling did on the same problem. `slow` names the gradient recipe whose
   advantage, loss and aggregation the step uses.

```toml
[recipe]
kind = "fst"
learning_rate = 2e-5
reflection_harness = "claude-code"
reflection_model = "anthropic/claude-sonnet-5"
```

`check` prints what it resolves to:

```
# fst: cycles of 6 cispo steps, each after gepa evolves 4 texts on the next 6 batches; every group split group_size / 4 per text; advantage = group mean, divided by spread; loss = cispo, weight truncated above 4.0, no lower bound, averaged per prompt; 1 substep; adamw betas 0.9 / 0.999, eps 1e-08, no warm-up (the paper's is 10 steps); degenerate groups dropped; kl 0.001 to the starting weights
```

The defaults are the paper's: `slow = "cispo"` with the importance weight truncated above 4.0 (App. D),
token losses averaged per prompt (Eq. 4), `substeps = 1` (App. D: `ppo_mini_batch_size` equals
`train_batch_size`), no refill (§2, after ScaleRL §3.2), no overlong term (see
[The overlong term](#the-overlong-term-dapo-and-cispo)), AdamW at betas 0.9 / 0.999 (App. D, which also warms up over 10 steps; `warmup` sets it), `kl_coef = 0.001`, `cycle = 6`, `population = 4`, `edits = "incremental"`, and a gepa
budget of five passes over the fast phase's tasks. The fast phase scores each (task, text) pair with
one rollout, whatever share of a group the text takes in the slow steps: App. D spends 960 metric
calls over 192 examples, and a metric call is one rollout. Scoring the texts carried from the
previous cycle comes on top of that budget, so a full population does not spend the search's passes
before it starts. `population` must divide `group_size`; `check` blocks the run otherwise.

Each cycle's population is kept under `runs/<id>/modules/cycle-<n>/<rank>/`, and the last
population's first text under `modules/best/`. `metrics.jsonl` holds gepa's round rows and one
`evolution` row per cycle, then the step rows; every one of them carries its `cycle`.

## evaluate

`evaluate` measures a policy without a gradient or a checkpoint. It runs one job per batch, each task
`group_size` times. The policy can be served by this run through the proxy, or by a provider. The mean
covers graded rollouts only: a masked rollout is left out rather than counted as 0. When nothing was
graded, the mean is `null`. `evaluate` can carry [modules](modules.md).

It writes one row per batch and a final row over the whole run. Here is `metrics.jsonl` from a run of
Qwen/Qwen3-8B with `group_size = 2`:

```json
{"at": "2026-10-07T20:57:50.873382+00:00", "seq": 1, "batch": 0, "tasks": 1, "rollouts": 2, "graded": 2, "masked": 0, "mean": 1.0}
{"at": "2026-10-07T20:57:50.873722+00:00", "seq": 2, "evaluation": true, "batches": 1, "rollouts": 2, "graded": 2, "masked": 0, "mean": 1.0}
```

For walkthroughs, see [Measure a policy](../tutorials/measure-a-policy.md) and [Train with dapo](../tutorials/train-with-dapo.md).
