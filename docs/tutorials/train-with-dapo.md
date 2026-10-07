# Train with dapo

`dapo` is a gradient recipe: it changes the policy's weights from the rewards its rollouts earn. A rollout is one trial seen from training: the model calls the proxy recorded, and the reward the task gave. This tutorial trains Qwen/Qwen3-8B with pi on the hello-world task from [Measure a policy](measure-a-policy.md), four attempts at a time.

!!! note "Illustrative numbers"
    The blueprint and the `check` output below are real. The numbers in the prose about the metrics are illustrations of the rules, not output.

## The blueprint

```toml title="blueprints/dapo-docker/run.toml"
[model]
name = "Qwen/Qwen3-8B"
lora_rank = 32

[data]
dataset = "hello-world"
batch_size = 1
group_size = 4
epochs = 2

[rollout]
harness = "pi@0.85.1"
sandbox = "docker"
concurrency = 4
max_tokens = 4096

[recipe]
kind = "dapo"
learning_rate = 2e-5

[checkpoints]
every = 1
ttl_hours = 24
```

A group is the `group_size` rollouts of one task in one step. Their rewards are compared with each other, so a group needs more than one member. `epochs = 2` passes over the one task twice, so this run takes two steps. `learning_rate` has no default for a gradient recipe. `lora_rank` is the rank of the LoRA adapter trained on the base model; 32 is also the default.

A gradient recipe trains the weights its own proxy serves, so `[model] provider` must stay `tinker`.

## Check

```console
$ shipyard check blueprints/dapo-docker --verbose
...
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
ttl_hours = 24.0
# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28; degenerate groups dropped

ok  recipe dapo
ok  model Qwen/Qwen3-8B, provider tinker (served by this run)
ok  dataset hello-world
ok  harness pi@0.85.1, sandbox docker
ok  dataset hello-world: 1 task under tasks/hello-world
ok  serving Qwen/Qwen3-8B on this machine for a docker sandbox
ok  tinker serves Qwen/Qwen3-8B
```

The comment line says what the name `dapo` resolves to:

- **advantage**: each rollout's reward minus its group's mean, divided by the group's spread (standard deviation).
- **loss**: PPO, with the probability ratio clipped to [1 − 0.2, 1 + 0.28].
- **degenerate groups dropped**: a group whose rewards are all equal, or that has one measured member, carries no gradient and is left out.

See [Recipes](../concepts/recipes.md) for `dr-grpo` and `cispo`.

## What one step does

1. **Publish.** The current weights are saved for sampling as `sample-<step>`, kept for 12 hours, and the proxy is pointed at them.
2. **Sample.** One Harbor job runs `group_size` trials of each task in the batch. The proxy records every model call.
3. **Credit.** Masked rollouts are set aside (see [Admission](../concepts/admission.md)). Each group's advantages are formed and degenerate groups are dropped. With `reference = "trainer"`, the reference logprobs are recomputed on the training engine, and credit refuses a batch if the weights moved since it was sampled.
4. **Apply.** The credited sequences are split into `substeps` parts, each one forward-backward pass and one Adam step.
5. **Log** one row to `metrics.jsonl`.
6. **Checkpoint** when `step + 1` is a multiple of `[checkpoints] every`, and only after a step that trained.

After the last step the run always saves a checkpoint tagged `final`, which Tinker keeps with no expiry.

## The metrics row

One row per step, after the `at` and `seq` every metrics row starts with. A measure with nothing behind it is absent from the row, never written as zero.

| column | what it is |
|---|---|
| `step` | the step, from 0 |
| `trained` | whether a gradient was taken |
| `groups`, `rollouts`, `graded`, `masked` | tasks in the batch, trials, trials with a reward, trials set aside |
| `degenerate` | groups dropped as flat or lone |
| `sequences`, `sequences_per_rollout` | training sequences built from the credited rollouts, and their mean per rollout |
| `train_tokens` | tokens sent through the forward-backward pass |
| `reward_mean`, `reward_spread`, `length_mean` | the batch's rewards, their spread within groups, tokens written per rollout |
| `kl_v1`, `kl_v2` | the reference logprobs μ against the training pass's: mean(μ − π) and half its mean square |
| `entropy` | mean(−μ) over the trained tokens |
| `anchor_kl` | the KL to the starting weights, only when `kl_coef > 0` |
| `learning_rate`, `substeps`, `loss_fn`, `seconds` | what the step used and how long it took |

### How to read five of them

**`trained`**. `false` means no gradient was taken: every group was degenerate or fully masked. On hello-world, a policy that solves all four attempts scores 1, 1, 1, 1. That group is flat, so the step logs `trained: false` and moves on.

**`reward_mean`**. The mean reward over the graded rollouts of the batch, flat groups included. Masked rollouts are absent from it. A rising `reward_mean` across steps is the policy improving.

**`reward_spread`**. The mean, over groups, of the standard deviation of the rewards inside each group. A group scoring 1, 1, 0, 0 has a spread of 0.5; a group scoring 1, 1, 1, 1 has 0. A spread that stays at 0 says the tasks are too easy or too hard for this policy: no group carries a gradient.

**`kl_v1`**. The mean, over the trained tokens, of the reference logprob minus the training pass's logprob. With `reference = "trainer"` and one substep, both come from the same weights on the same engine, so `kl_v1` sits near 0; far from 0, the two disagree. With more substeps it grows with how far the weights moved inside the step. With `reference = "sampler"` it measures how far the sampler's logprobs are from the trainer's.

**`sequences_per_rollout`**. How many training sequences each credited rollout became. 1.0 means each rollout trained as one sequence: every model call extended the one before. Above 1, some call's prompt did not extend any earlier call's prompt plus completion, and the rollout split there. Each completion is still trained exactly once. See [Forks and the bridge](../concepts/proxy.md#forks-and-the-bridge).

## checkpoints.jsonl

Each checkpoint appends one row, and `shipyard show` prints them as its `checkpoints` section.

| key | what it is |
|---|---|
| `at` | when it was saved |
| `tag` | `step-<n>` after n steps, or `final` |
| `state_path` | a `tinker://` path to the weights and optimizer state: what a run continues from |
| `sampler_path` | a `tinker://` path to the weights for sampling: what a run measures |
| `ttl_hours` | how long Tinker keeps both; at least 1, 168 by default, and null on `final`, which is kept |

## Continue from a checkpoint

To continue a run, start a new one from its checkpoint:

```toml title="blueprints/dapo-continue/run.toml"
[model]
name = "Qwen/Qwen3-8B"
from_checkpoint = "<the state_path of a checkpoints.jsonl row>"
restore_optimizer = true
```

The rest of the blueprint is as before. `from_checkpoint` loads the training client from that state, so `lora_rank` is not read. The first step publishes those weights for sampling and points the proxy at them; with `kl_coef > 0` the KL anchor samples them too. `check` blocks a `sampler_path` here. `restore_optimizer = true` also loads the optimizer's state. Left `false`, the weights continue with a fresh optimizer. The continuation is a new run with its own id, and its steps count from 0 again.

To measure a checkpoint instead, an `evaluate` blueprint names its `sampler_path` in `from_checkpoint`, and the proxy serves those weights.

## What training costs

`costs.json` counts training under the `tinker` party beside the sampling counts: `train_tokens` for the forward-backward passes and `reference_tokens` for the reference passes. See [Costs](../concepts/costs.md).
