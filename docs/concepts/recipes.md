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
| `evaluate` | Measures the policy. | no |

The first three are the gradient recipes. They train the weights the run serves, so they need
`[model] provider = "tinker"`, which is the default. With any other provider, the run stops before it trains:

```
ValueError: [recipe] kind = 'dapo' trains the weights this run serves, and [model] provider = 'anthropic' serves them elsewhere
```

## A gradient recipe is a name

Each gradient recipe is a preset. The name fixes how the advantage is formed, which loss runs and how
it clips. You can tune its knobs, but you cannot combine the parts another way.

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
substeps = 1
reference = "trainer"
kl_coef = 0.0
clip_low = 0.2
clip_high = 0.28

[checkpoints]
every = 1
ttl_hours = 168.0
# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28; degenerate groups dropped
```

Each name resolves as follows:

| | `dapo` | `dr-grpo` | `cispo` |
|---|---|---|---|
| Advantage | (r − mean) / (spread + 1e-6) | r − mean | (r − mean) / (spread + 1e-6) |
| Tinker loss | `ppo` | `ppo` | `cispo` |
| Knobs and defaults | `clip_low = 0.2`, `clip_high = 0.28` | `clip = 0.2`, `length_penalty = 0.0`, `length_floor = 0` | `clip_high = 0.2` |
| `loss_fn_config` sent | `clip_low_threshold = 0.8`, `clip_high_threshold = 1.28` | `0.8` and `1.2` | `0.0` and `1.2` |

The knobs are epsilons, but Tinker takes bounds, so `clip_low = 0.2` reaches it as a lower bound of 0.8.
`cispo` truncates the importance weight above `1 + clip_high` and has no lower bound. `check` prints these lines:

```
# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28; degenerate groups dropped
# dr-grpo: advantage = group mean, not divided by spread; loss = ppo, clip 0.2 / 0.2; degenerate groups dropped
# cispo: advantage = group mean, divided by spread; loss = cispo, weight truncated above 1.2, no lower bound; degenerate groups dropped
```

### The length rule (dr-grpo only)

`length_penalty` docks solved answers for their length. A solved answer has reward 1.0, and the rule
applies only in a group with at least two of them. Each solved reward loses
`length_penalty × max(L − length_floor, 0) / mean solved L`, where L is the number of tokens the policy
wrote. The loss is capped at 0.5, so the longest solved answer still scores above every failure.

```
# dr-grpo: advantage = group mean, not divided by spread; loss = ppo, clip 0.2 / 0.2; length penalty 0.1 over 512 tokens among solved answers; degenerate groups dropped
```

## The common knobs

| Key | Default | Meaning |
|---|---|---|
| `learning_rate` | required, > 0 | Adam's step size (beta1 0.9, beta2 0.95, eps 1e-8). |
| `substeps` | `1` | Optimizer steps per batch. The batch's sequences are shuffled and split into this many parts, never more parts than sequences. |
| `reference` | `"trainer"` | Where μ comes from. μ is the sampling logprobs that the ratio is formed against. `"trainer"` recomputes μ in one forward pass on the training engine. `"sampler"` reads the logprobs the proxy recorded and makes no extra pass. |
| `kl_coef` | `0.0` | A per-token penalty for drifting from the run's starting weights, which are the base model or `from_checkpoint`. It is subtracted from the advantage: `kl_coef × (μ − anchor)`. |
| `modules` | unset | A directory of skills to carry into every rollout. See [Modules](modules.md). |

`"trainer"` is the default because the sampler and the trainer are different engines and their logprobs
differ slightly. `"sampler"` saves the pass, but it refuses a batch whose records carry no logprobs.
With `kl_coef = 0.01`, the comment line gains `; kl 0.01 to the starting weights`.

## How a step runs

Each batch in the plan is one step, numbered from 0 (see [Datasets](datasets.md)).

1. **Publish.** The current weights are saved for sampling as `sample-<step>` and kept for 12 hours.
2. **Point.** The [proxy](proxy.md) is told to serve those weights.
3. **Sample.** One Harbor job runs over the batch, each task `group_size` times. Every model call leaves a record.
4. **Pack.** Each rollout's records become token sequences by the token-prefix rule. A call joins the
   sequence whose last prompt plus completion begins its prompt, the longest such. If none matches, the
   call starts a new sequence. Each completion is a training target exactly once. Failed
   calls and calls carrying images never pack.
5. **Credit.** Rollouts are grouped by task, and only the measured ones count (see [Admission](admission.md)).
   Degenerate groups are dropped, then the advantages, μ and the KL term are computed.
6. **Step.** `forward_backward` and `optim_step` run on Tinker, `substeps` times.
7. **Log.** One row is appended to `metrics.jsonl`.
8. **Checkpoint.** When `step + 1` is divisible by `[checkpoints] every`, `step-<step + 1>` is saved. After the last batch, `final` is saved.

A checkpoint saves a state path, from which training can continue, and a sampler path, from which a
policy can be sampled. Both are kept for `ttl_hours`, which is at least 1 hour. Each checkpoint appends a row
to `checkpoints.jsonl` with `tag`, `state_path`, `sampler_path` and `ttl_hours`.

### Degenerate groups

A group is degenerate when only one of its rollouts was measured, or when its measured rewards are all
equal after `dr-grpo`'s length rule. A degenerate group carries no gradient, so credit drops it before
any reference pass. As a result, `group_size = 1` never trains. If no group of a batch is left, the step
logs `trained: false` and takes no gradient and no checkpoint. Under `dapo` and `cispo`, or `dr-grpo`
without a length penalty, a policy that solves every task, or fails every one, gives flat groups and so
takes no step.

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
| `step` | The batch index, from 0. |
| `trained` | Whether a gradient was taken. |
| `groups`, `rollouts` | Groups (tasks) and rollouts in the batch. |
| `graded`, `masked` | Rollouts with a reward, and those left out with a mask. |
| `degenerate` | Groups dropped as degenerate. |
| `sequences`, `sequences_per_rollout` | Sequences trained on, and their mean per credited rollout. |
| `train_tokens` | Tokens sent to `forward_backward`. |
| `reward_mean`, `reward_spread` | The mean reward over graded rollouts, and the mean of the groups' spreads. |
| `length_mean` | Mean tokens the policy wrote per graded rollout. |
| `kl_v1`, `kl_v2` | mean(μ − π) and half its mean square over the trained tokens, where π is the training pass's logprobs. |
| `entropy` | mean(−μ) over the trained tokens. |
| `anchor_kl` | mean(μ − anchor), present only when `kl_coef > 0`. |
| `learning_rate`, `substeps`, `loss_fn` | As applied. |
| `seconds` | The time taken to apply the gradient. |

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
