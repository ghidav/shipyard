# Configuration

Every key `run.toml` accepts. A key not listed here is an error, and a table not listed here is
an error too ([Blueprints](../concepts/blueprints.md)). `shipyard check` prints the config with
every default filled in.

The **For** column says when a key matters:

- **all**: every run.
- **served**: only when the model is served, that is `[model] provider = "tinker"`. Otherwise
  the key is accepted and has no effect.
- **gradient**: only for `dapo`, `dr-grpo` and `cispo`.

## `[model]`

| Key | Type | Default | For | Meaning |
|---|---|---|---|---|
| `name` | string | required | all | The model. For a served model, the base model as the backend lists it, such as `"Qwen/Qwen3-8B"`; otherwise the name the harness asks its provider for. |
| `provider` | string | `"tinker"` | all | Who serves the weights. `"tinker"`: this run serves them through its proxy, from a Tinker-API backend. Anything else: the harness calls that provider itself. |
| `from_checkpoint` | string | unset | served | A `tinker://` path to start from instead of the base model: a checkpoint's `state_path` to go on training, its `sampler_path` to measure it. `check` blocks the other one. |
| `lora_rank` | integer | unset (32) | gradient | The LoRA rank of a new training client. Not used with `from_checkpoint`. |
| `restore_optimizer` | boolean | `false` | gradient | With `from_checkpoint`, load the optimizer state as well as the weights. The state continues the earlier run's, so the run skips its recipe's warm-up. |

## `[data]`

| Key | Type | Default | For | Meaning |
|---|---|---|---|---|
| `dataset` | string or list of strings | required | all | The dataset, or datasets, under `tasks/` ([Datasets](../concepts/datasets.md)). |
| `batch_size` | integer ≥ 1 | required | all | Tasks per batch. One batch is one Harbor job. |
| `group_size` | integer ≥ 1 | `1` | all | How many times each task runs in its batch. |
| `epochs` | integer ≥ 1 | `1` | all | Passes over the tasks. |
| `seed` | integer | `0` | all | The shuffle's seed; epoch `e` is shuffled with `seed + e`. `gepa` draws its held-out tasks and its minibatches with it, and `fst`'s fast phase in cycle `c` its minibatches with `seed + c`. |

`gepa` reads only `dataset`, `group_size` and `seed`.

## `[rollout]`

| Key | Type | Default | For | Meaning |
|---|---|---|---|---|
| `harness` | string | required | all | The Harbor agent that runs each task, such as `"claude-code"`. `name@version` pins its version: `"pi@0.85.1"`. A blank name blocks the run. |
| `sandbox` | string | `"docker"` | all | A Harbor environment type, such as `"docker"`, `"podman"`, `"apple-container"` or `"modal"`. `check` lists them all on a typo. |
| `concurrency` | integer ≥ 1 | `4` | all | Trials of one job running at once. |
| `timeout` | float | unset | all | Seconds the agent may run, in place of the task's own timeout. |
| `setup_timeout` | float | unset | all | Seconds the harness's setup may take, in place of the agent's default. For a sandbox elsewhere, `check` warns when it is set under 600; unset, the agent's default stands unchecked. |
| `env` | table of strings | `{}` | all | Environment variables for the harness in every trial. `check` blocks a key starting with `TINKER_`, and `SHIPYARD_CONTROL_TOKEN`, and a value such as `"${TINKER_API_KEY}"` that Harbor would fill with one. |
| `kwargs` | table | `{}` | all | Extra arguments for the Harbor agent. |
| `jobs_dir` | string | `"jobs"` | all | Where job directories go, relative to the working directory. |
| `endpoint_url` | string | `""` | served | A proxy started elsewhere. The run starts none of its own and needs `SHIPYARD_PROXY_TOKEN` set. A gradient run, or one with `from_checkpoint`, also needs `SHIPYARD_CONTROL_TOKEN` to point it at weights; `check` does not look for that one. |
| `host` | string | `"host.docker.internal"` | served | The name a sandbox on this machine reaches the proxy by. |
| `bind` | string | `"0.0.0.0"` | served | The interface the proxy listens on. |
| `bind_port` | integer 0 to 65535 | `0` | served | The port the proxy listens on; `0` takes a free one. |
| `temperature` | float | `1.0` | served | Set on every call, whatever the harness asks. |
| `top_p` | float | `1.0` | served | Set on every call, whatever the harness asks. |
| `top_k` | integer | `-1` | served | Set on every call, whatever the harness asks; `-1` is off. |
| `max_tokens` | integer ≥ 1 | `8192` | served | The longest reply. A call asking for more, or naming no limit, gets this. |
| `max_context` | integer ≥ 0 | `0` | served | The context window each call must fit, and the most tokens one trial may sample in all. `0` takes the model's from the backend. |
| `renderer` | string | `""` | served | A tinker-cookbook renderer to use in place of the model's own. |
| `cut_volatile` | boolean | `true` | served | Cut the lines the harness changes on every call from its system message, such as Claude Code's `<total_tokens>` line. |
| `fill_context` | boolean | `false` | served | A call that would overflow the context gets a shorter `max_tokens` instead of a refusal. Also switches the harness's own compaction off, where its profile knows how. |
| `check_turns` | boolean | `false` | served | Count the calls the harness made in its own log, and mask a trial whose count is more than the proxy recorded. Claude Code and opencode only. |

How the served keys behave is in [The proxy](../concepts/proxy.md).

## `[recipe]`

`kind` is required and picks the recipe: `"dapo"`, `"dr-grpo"`, `"cispo"`, `"gepa"`, `"fst"` or
`"evaluate"`. Each kind accepts its own keys and no others ([Recipes](../concepts/recipes.md)).

### Common to `dapo`, `dr-grpo`, `cispo` and `fst`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `learning_rate` | float > 0 | required | AdamW's learning rate, reached after the recipe's warm-up. The betas, eps, weight decay, gradient clipping and warm-up are the recipe's, from its paper ([The optimizer](../concepts/recipes.md#the-optimizer)), and are not keys. |
| `substeps` | integer ≥ 1 | `16` for `dapo` (DAPO §4.1) and `cispo` (MiniMax-M1 §3.1); `1` for `dr-grpo` (Dr. GRPO states none) and `fst` (FST App. D) | Optimizer steps per batch. The batch is split by prompt into this many parts of whole groups, or one per group carrying a gradient when it has fewer, each one `forward_backward` and one `optim_step`. Every part keeps the reference logprobs of the weights the batch was sampled at. |
| `reference` | `"trainer"` or `"sampler"` | `"trainer"` | Where the loss's reference logprobs come from: a forward pass on the trainer, or the logprobs recorded when sampling, with no extra pass. |
| `kl_coef` | float ≥ 0 | `0.0`, as DAPO (§2.3), Dr. GRPO (App. G) and MiniMax-M1 (§3.1) train | Weight of a per-token KL term to the run's starting weights, folded into the advantage token by token, not centred. `0` is off. |
| `modules` | string | unset | A modules directory, relative to the blueprint, carried into every job ([Modules](../concepts/modules.md)). |

The gradient recipes need a served model.

### `dapo`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `clip_low` | float, 0 < x < 1 | `0.2` (DAPO §4.1) | The ratio's lower bound is `1 - clip_low`. |
| `clip_high` | float > 0 | `0.28` (DAPO §4.1) | The ratio's upper bound is `1 + clip_high`. |
| `refill` | integer ≥ 0 | `9` (DAPO Alg. 1; ten generation batches a step in DAPO's released recipe) | The most extra batches a step samples from the plan while fewer than `batch_size` groups carry a gradient ([Dynamic sampling](../concepts/recipes.md#dynamic-sampling)). `0` is off. |
| `overlong_penalty` | float ≥ 0 | `0.5` (DAPO Eq. 13: −1 on its reward of −1 or 1, Eq. 7, which is −0.5 on Harbor's 0 to 1) | The most a rollout loses from its reward for its length, reached at its token budget ([The overlong term](../concepts/recipes.md#the-overlong-term-dapo-and-cispo)). `0` is off. |
| `overlong_buffer` | float, 0 < x ≤ 1 | `0.2` (DAPO §4.1: L_cache / L_max, 4,096 of 20,480 tokens) | The last share of the token budget over which the loss grows linearly from 0 to `overlong_penalty`. |

Token losses are averaged per prompt (DAPO Eq. 8). The optimizer is AdamW warmed up over 20 steps
(DAPO §4.1), at the cookbook's betas 0.9 / 0.95, eps 1e-8, no weight decay and no gradient
clipping; the paper states none of these.

### `dr-grpo`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `clip` | float, 0 < x < 1 | `0.2` (Dr. GRPO App. G) | The ratio's bounds are `1 - clip` and `1 + clip`. |
| `length_penalty` | float ≥ 0 | `0.0` (Dr. GRPO docks no answer for its length) | shipyard's own length rule ([The length rule](../concepts/recipes.md#the-length-rule-dr-grpo-only)): when at least two answers in a group are solved, each solved one loses `length_penalty × (length - length_floor) / mean solved length`, at most `length_cap`. `0` is off. |
| `length_floor` | integer ≥ 0 | `0` (shipyard's own; Dr. GRPO has no length rule) | Tokens of an answer the length penalty does not count. |
| `length_cap` | float > 0, `inf` allowed | `0.5` (shipyard's own; Dr. GRPO has no length rule) | The most the length penalty takes from a solved answer, so a solved answer never falls below a failure scored `1 - length_cap` or less. `inf` lifts the cap; `check` warns at 1 or more while `length_penalty` is above 0. |
| `refill` | integer ≥ 0 | `0` (Dr. GRPO states no dynamic sampling, App. G) | As for `dapo`. |

Token losses are summed (Dr. GRPO §3.2). The optimizer is AdamW at betas 0.9 / 0.95, no weight
decay, the gradient norm clipped at 1.0 and a constant learning rate (Dr. GRPO App. G, Table 6),
with the cookbook's eps 1e-8.

### `cispo`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `clip_high` | float > 0 | `3.0` (ScaleRL App. A.17.2, FST App. D) | The importance weight is cut above `1 + clip_high`. It has no lower bound. |
| `refill` | integer ≥ 0 | `9` (MiniMax-M1 §3.1, which takes DAPO's dynamic sampling) | As for `dapo`. |
| `overlong_penalty` | float ≥ 0 | `0.5` (MiniMax-M1 §3.1, which takes DAPO's length penalty) | As for `dapo`. |
| `overlong_buffer` | float, 0 < x ≤ 1 | `0.2` (as for `dapo`) | As for `dapo`. |

Token losses are averaged per prompt (MiniMax-M1 Eq. 4). The optimizer is AdamW at betas 0.9 / 0.95
and eps 1e-15 (MiniMax-M1 §3.2), with no weight decay, no gradient clipping and no warm-up, the
cookbook's; the paper states none of these.

### `gepa`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `reflection_harness` | string | required | The harness the reflector runs as, in its own Harbor trial. A blank name blocks the run. |
| `reflection_model` | string | unset | The model the reflector's harness uses; unset is the harness's own default. |
| `reflection_image` | string | `"python:3.12-slim"` | The container image the reflector runs in. |
| `modules` | string | `"modules"` | The seed: a modules directory, relative to the blueprint. |
| `pareto` | integer ≥ 0 or string | unset | The Pareto tasks, which score every candidate the search keeps; minibatches come from the rest. A count holds that many of the run's tasks out, drawn with `[data] seed`; `0` holds none out and selects on the tasks the search reflects on; a dataset name selects on that dataset. Unset holds out two thirds of the run's tasks, rounded down, as in three of GEPA's four benchmarks: GEPA §4.1 (split sizes) and §4.3 (the validation set is D_pareto). §4.1 states no single proportion: 300 validation tasks beside 150 training ones for three benchmarks, 111 beside 111 for PUPA. |
| `minibatch` | integer ≥ 1 | `3` (GEPA §4.3) | How many feedback tasks the parent and a new candidate run on each round. |
| `budget` | integer ≥ 1 | unset (GEPA states no default: §4.3 matches MIPROv2's rollouts per benchmark) | Rollouts the search may spend, every scoring counted. Unset is two passes: 2 × tasks × `group_size`, over every task it may score. |
| `patience` | integer ≥ 1 | `3` (GEPA has none; its loop runs until the budget is spent) | Rounds in a row with no new candidate before the search stops. |
| `edits` | `"rewrite"` or `"incremental"` | `"rewrite"` | `"incremental"` asks the reflector to edit the text in place and keep what works, not rewrite it. |

See [gepa](../concepts/gepa.md).

### `fst`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `slow` | `"dapo"`, `"dr-grpo"` or `"cispo"` | `"cispo"` (FST App. D) | The gradient recipe each slow step uses, with its loss and clipping. Its own knobs (`clip_low`, `clip_high`, `clip`, `length_penalty`, `length_floor`, `length_cap`) are accepted when they belong to it, with its defaults. `refill`, `overlong_penalty` and `overlong_buffer` are not keys of `fst`: a slow step trains on the batch it sampled, with no overlong term. |
| `cycle` | integer ≥ 1 | `6` (FST App. D) | Slow steps per cycle, and the batches each fast phase looks ahead. |
| `population` | integer ≥ 1 | `4` (FST App. E, the light recipe) | Texts kept per cycle. It must divide `group_size`. |
| `anchor` | integer ≥ 1 | unset | The fast phase evolves on the first `anchor` tasks of the lookahead; unset is all of them. |
| `kl_coef` | float ≥ 0 | `0.001` (FST App. D) | As above. |
| `budget` | integer ≥ 1 | unset | Rollouts each fast phase may spend; the fast phase scores each (task, text) pair with one rollout. Unset is five passes: 5 × its tasks (FST App. D: 960 metric calls over 192 examples, one rollout each). One more pass per text carried from the previous cycle, beyond the first, is added on top to score them. |
| `edits` | `"rewrite"` or `"incremental"` | `"incremental"` (FST App. E) | As for `gepa`. |

`reflection_harness` (required), `reflection_model`, `reflection_image`, `modules` (the seed, default
`"modules"`), `minibatch` and `patience` are as for `gepa`. The fast phase selects on the paper's anchor set.
FST names one anchor set (§3, App. A) and no other source for the minibatches, so fst draws them
from it too. Token losses are averaged per prompt (FST Eq. 4), whatever `slow` is. The optimizer is
AdamW at betas 0.9 / 0.999 and no weight decay, warmed up over 10 steps (FST App. D), with eps 1e-8,
PyTorch's and the cookbook's, and no gradient clipping, as the paper states none. See
[fst](../concepts/recipes.md#fst).

### `evaluate`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `modules` | string | unset | A modules directory, relative to the blueprint, carried into every job. |

## `[checkpoints]`

Optional. Only the gradient recipes save checkpoints.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `every` | integer ≥ 1 | `1` | Save a checkpoint after every this many steps; a step that trained nothing saves none. A `final` one is saved once the last step is done, whatever `every` says. |
| `ttl_hours` | float ≥ 1 | `168.0` | How long the backend keeps each `step-<n>` checkpoint, in hours. The backend keeps one for an hour at least. `final` is kept with no expiry. |

## Not in the file

- `TINKER_API_KEY`: the Tinker-API backend's key, for a served model.
- `TINKER_BASE_URL`: another Tinker-API backend; the default is Tinker's own.
- `SHIPYARD_RUNS`: where runs live, when no `--root` is given.
- `SHIPYARD_QUIET=1`: no progress bars on a terminal.
- `SHIPYARD_PROXY_TOKEN`: the harness's key to the proxy; required with `endpoint_url`.
- `SHIPYARD_CONTROL_TOKEN`: the run's key to point the proxy at weights.

[Commands](cli.md#environment) says which command reads each.

`shipyard` reads `.env` in the working directory first; a variable set in the shell wins.
