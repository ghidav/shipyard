# Train with dapo

`dapo` is a gradient recipe: it changes the policy's weights from the rewards its rollouts earn. A rollout is one trial seen from training: the model calls the proxy recorded, and the reward the task gave. This tutorial trains Qwen/Qwen3-8B with pi on eight Reasoning Gym tasks, four tasks per step and four attempts per task.

!!! note "A real run"
    The blueprint, the `check` output, the metrics, the checkpoints and the costs below are from one run of this blueprint, `dapo-docker__Su3tmfE`, and from its continuation, `dapo-continue__ZCdgppB`. Two steps are far too few to show a policy improving. The run shows what a step does and what it records.

## The dataset

`rg-small` holds eight tasks from Harbor's `reasoning-gym-easy`: the first seed of eight categories, from time intervals and prime factorisation to zebra puzzles and shortest paths. Each task asks for one answer in `/workspace/answer.txt`, and its verifier scores it 0 or 1.

```console
$ harbor datasets download reasoning-gym-easy -o tasks
$ mkdir tasks/rg-small
$ for c in arithmetic-time-intervals arithmetic-prime-factorization logic-knights-knaves \
    logic-zebra-puzzles games-countdown games-puzzle24 graphs-shortest-path \
    cognition-number-sequence; do
    cp -R tasks/reasoning-gym-easy/reasoning-gym-$c-easy*-seed45-0 tasks/rg-small/
  done
```

## The blueprint

```toml title="blueprints/dapo-docker/run.toml"
[model]
name = "Qwen/Qwen3-8B"
lora_rank = 32

[data]
dataset = "rg-small"
batch_size = 4
group_size = 4

[rollout]
harness = "pi@0.85.1"
sandbox = "docker"
concurrency = 4
max_tokens = 4096
max_context = 12288
timeout = 600

[recipe]
kind = "dapo"
learning_rate = 2e-5

[checkpoints]
every = 1
ttl_hours = 24
```

A group is the `group_size` rollouts of one task in one step. Their rewards are compared with each other, so a group needs more than one member. Eight tasks at four per step make two steps. `learning_rate` has no default for a gradient recipe. `lora_rank` is the rank of the LoRA adapter trained on the base model; 32 is also the default.

Two rollout keys matter for these tasks:

- **`timeout = 600`.** Each Reasoning Gym task gives the agent 120 seconds. Qwen3-8B thinks at length, and a first run at that limit had 26 of its 32 rollouts cut by the clock. A rollout the clock cut is masked, never scored 0, so neither step had a group left to train on. `timeout` replaces the task's limit.
- **`max_context = 12288`.** It is the context each call must fit, and the most tokens one trial may sample. With Qwen3-8B, pi often keeps calling tools long after it has written its answer, up to 127 calls in one trial. Once its history passes 8,192 tokens, a call's prompt plus its 4,096-token `max_tokens` no longer fits, the proxy refuses the call, and pi stops. In this run 20 of the 32 trials ended that way, and the verifier graded each on the answer it had written.

A gradient recipe trains the weights its own proxy serves, so `[model] provider` must stay `tinker`.

## Check

```console
$ shipyard check blueprints/dapo-docker --verbose
...
[rollout]
...
max_tokens = 4096
max_context = 12288
renderer = "qwen3"
cut_volatile = true
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
ok  dataset rg-small
ok  harness pi@0.85.1, sandbox docker
ok  dataset rg-small: 8 tasks under tasks/rg-small
ok  serving Qwen/Qwen3-8B on this machine for a docker sandbox
ok  tinker serves Qwen/Qwen3-8B
```

`renderer = "qwen3"` is the renderer the proxy will load for the model. The comment line says what the name `dapo` resolves to:

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

## What the run did

The run took 30 minutes, about 15 per step. Almost all of that is the rollouts: the training pass took 4 to 6 seconds a step. Each step's 16 rollouts scored:

| step | task | rewards |
|---|---|---|
| 0 | arithmetic-time-intervals | 1, 1, 1, 1 |
| 0 | cognition-number-sequence | 1, 1, 1, 1 |
| 0 | games-puzzle24 | 1, 0, 1, 1 |
| 0 | graphs-shortest-path | 0, 0, 0, 0 |
| 1 | arithmetic-prime-factorization | 1, 1, 1, 1 |
| 1 | games-countdown | 1, 1, 0, 1 |
| 1 | logic-knights-knaves | 1, 1, 1, 1 |
| 1 | logic-zebra-puzzles | 1, 1, 1, 0 |

A group whose four rewards are equal has nothing to compare, so it is degenerate. Step 0 trained on one group of four rollouts, and step 1 on two. Step 0's row in `metrics.jsonl`:

```json
{"at": "2026-10-07T23:17:32.940356+00:00", "seq": 1, "step": 0, "trained": true, "groups": 4, "rollouts": 16, "graded": 16, "masked": 0, "degenerate": 3, "sequences": 4, "sequences_per_rollout": 1.0, "train_tokens": 30273, "reward_mean": 0.6874999971064815, "reward_spread": 0.10825317547305482, "length_mean": 4651.3125, "kl_v1": 0.0, "kl_v2": 0.0, "entropy": 0.14287219941616058, "learning_rate": 2e-05, "substeps": 1, "loss_fn": "ppo", "seconds": 4.253}
```

| | step 0 | step 1 |
|---|---|---|
| `degenerate` | 3 | 2 |
| `sequences` | 4 | 8 |
| `train_tokens` | 30,273 | 49,832 |
| `reward_mean` | 0.69 | 0.88 |
| `length_mean` | 4,651 | 3,753 |
| `entropy` | 0.143 | 0.186 |

The two steps drew different tasks, so the rise in `reward_mean` compares two batches, not the policy before and after.

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

**`trained`**. `false` means no gradient was taken: every group was degenerate or fully masked. Both steps above trained, step 0 on its one group with a spread.

**`reward_mean`**. The mean reward over the graded rollouts of the batch, flat groups included. Masked rollouts are absent from it. Step 0's 0.69 is 11 of 16. Across many steps, a rising `reward_mean` is the policy improving.

**`reward_spread`**. The mean, over groups, of the standard deviation of the rewards inside each group. In step 0, puzzle24's 1, 0, 1, 1 has a deviation of 0.43 and the other three groups have 0, so the spread is 0.11. A spread that stays at 0 says the tasks are too easy or too hard for this policy: no group carries a gradient.

**`kl_v1`**. The mean, over the trained tokens, of the reference logprob minus the training pass's logprob. With `reference = "trainer"` and one substep, both come from the same weights on the same engine, so `kl_v1` is 0 here. With more substeps it grows with how far the weights moved inside the step. With `reference = "sampler"` it measures how far the sampler's logprobs are from the trainer's.

**`sequences_per_rollout`**. How many training sequences each credited rollout became. 1.0 means each rollout trained as one sequence: every model call extended the one before. Here every call after a trial's first was bridged, 1,749 of 1,781, so even a trial of 127 calls trained as one sequence. Above 1, some call's prompt did not extend any earlier call's prompt plus completion, and the rollout split there. Each completion is still trained exactly once. See [Forks and the bridge](../concepts/proxy.md#forks-and-the-bridge).

## checkpoints.jsonl

Each checkpoint appends one row, and `shipyard show` prints them as its `checkpoints` section. The run saved one per step:

```json
{"at": "2026-10-07T23:17:45.329114+00:00", "tag": "step-1", "state_path": "tinker://51e37fd8-eb13-58bb-b463-376215fad05e:train:0/weights/step-1", "sampler_path": "tinker://51e37fd8-eb13-58bb-b463-376215fad05e:train:0/sampler_weights/step-1", "ttl_hours": 24.0}
{"at": "2026-10-07T23:32:33.800274+00:00", "tag": "step-2", "state_path": "tinker://51e37fd8-eb13-58bb-b463-376215fad05e:train:0/weights/step-2", "sampler_path": "tinker://51e37fd8-eb13-58bb-b463-376215fad05e:train:0/sampler_weights/step-2", "ttl_hours": 24.0}
```

A third row follows with the tag `final`, the same weights as `step-2`, and `ttl_hours` null.

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
from_checkpoint = "tinker://51e37fd8-eb13-58bb-b463-376215fad05e:train:0/weights/final"
restore_optimizer = true

[data]
dataset = "rg-four"
```

The rest of the blueprint is as before. `from_checkpoint` loads the training client from that state, so `lora_rank` is not read. The first step publishes those weights for sampling and points the proxy at them; with `kl_coef > 0` the KL anchor samples them too. `check` blocks a `sampler_path` here. `restore_optimizer = true` also loads the optimizer's state. Left `false`, the weights continue with a fresh optimizer. The continuation is a new run with its own id, and its steps count from 0 again.

`rg-four` is the first four tasks of `rg-small` in name order, so the continuation runs one step. Every one of its 16 rollouts scored 1, so all four groups were degenerate and the step took no gradient:

```json
{"at": "2026-10-07T23:46:36.096695+00:00", "seq": 1, "step": 0, "trained": false, "groups": 4, "rollouts": 16, "graded": 16, "masked": 0, "degenerate": 4, "sequences": 0, "reward_mean": 0.9999999971064815, "reward_spread": 0.0, "length_mean": 2851.0}
```

The run still saved `final`, so a continuation that trains nothing still ends with a state to continue from. To make it train, give it tasks the policy sometimes fails.

To measure a checkpoint instead, an `evaluate` blueprint names its `sampler_path` in `from_checkpoint`, and the proxy serves those weights.

## What training costs

`costs.json` counts training under the `tinker` party beside the sampling counts: `train_tokens` for the forward-backward passes and `reference_tokens` for the reference passes. The run's:

```json
{
  "parties": {
    "tinker": {
      "trials": 32,
      "input_tokens": 9670174,
      "cache_tokens": 9247232,
      "output_tokens": 134466,
      "reference_tokens": 80105,
      "train_tokens": 80105
    }
  },
  "sandbox": {
    "docker": {
      "trials": 32,
      "seconds": 6073.979176
    }
  }
}
```

The prompts dominate: 1,781 calls, each re-sending the trial's history, up to 8,192 tokens. The backend had 96% of those prompt tokens cached, since each bridged prompt begins with the call before it. Training is small beside them: the 80,105 tokens of the twelve credited rollouts, once through the reference pass and once through the training pass. The continuation spent 4,129,096 input tokens, 3,951,872 of them cached, and 45,616 output tokens, with no training counts, since its step trained nothing. See [Costs](../concepts/costs.md).
