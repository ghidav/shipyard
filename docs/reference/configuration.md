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
| `restore_optimizer` | boolean | `false` | gradient | With `from_checkpoint`, load the optimizer state as well as the weights. |

## `[data]`

| Key | Type | Default | For | Meaning |
|---|---|---|---|---|
| `dataset` | string or list of strings | required | all | The dataset, or datasets, under `tasks/` ([Datasets](../concepts/datasets.md)). |
| `batch_size` | integer ≥ 1 | required | all | Tasks per batch. One batch is one Harbor job. |
| `group_size` | integer ≥ 1 | `1` | all | How many times each task runs in its batch. |
| `epochs` | integer ≥ 1 | `1` | all | Passes over the tasks. |
| `seed` | integer | `0` | all | The shuffle's seed; epoch `e` is shuffled with `seed + e`. |

`gepa` reads only `dataset` and `group_size`.

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
| `cut_volatile` | boolean | `false` | served | Cut the lines the harness changes on every call from its system message, such as Claude Code's `<total_tokens>` line. |
| `fill_context` | boolean | `false` | served | A call that would overflow the context gets a shorter `max_tokens` instead of a refusal. Also switches the harness's own compaction off, where its profile knows how. |
| `check_turns` | boolean | `false` | served | Count the calls the harness made in its own log, and mask a trial whose count is more than the proxy recorded. Claude Code and opencode only. |

How the served keys behave is in [The proxy](../concepts/proxy.md).

## `[recipe]`

`kind` is required and picks the recipe: `"dapo"`, `"dr-grpo"`, `"cispo"`, `"gepa"`, `"fst"` or
`"evaluate"`. Each kind accepts its own keys and no others ([Recipes](../concepts/recipes.md)).

### Common to `dapo`, `dr-grpo`, `cispo` and `fst`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `learning_rate` | float > 0 | required | Adam's learning rate. |
| `substeps` | integer ≥ 1 | `1` | Optimizer steps per batch. The batch is split into this many parts, or one per sequence when it has fewer sequences, each one `forward_backward` and one `optim_step`. |
| `reference` | `"trainer"` or `"sampler"` | `"trainer"` | Where the loss's reference logprobs come from: a forward pass on the trainer, or the logprobs recorded when sampling, with no extra pass. |
| `kl_coef` | float ≥ 0 | `0.0` | Weight of a per-token KL term to the run's starting weights, folded into the advantage. `0` is off. |
| `modules` | string | unset | A modules directory, relative to the blueprint, carried into every job ([Modules](../concepts/modules.md)). |

The gradient recipes need a served model.

### `dapo`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `clip_low` | float, 0 < x < 1 | `0.2` | The ratio's lower bound is `1 - clip_low`. |
| `clip_high` | float > 0 | `0.28` | The ratio's upper bound is `1 + clip_high`. |

### `dr-grpo`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `clip` | float, 0 < x < 1 | `0.2` | The ratio's bounds are `1 - clip` and `1 + clip`. |
| `length_penalty` | float ≥ 0 | `0.0` | When at least two answers in a group are solved, each solved one loses `length_penalty × (length - length_floor) / mean solved length`, at most 0.5. `0` is off. |
| `length_floor` | integer ≥ 0 | `0` | Tokens of an answer the length penalty does not count. |

### `cispo`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `clip_high` | float > 0 | `0.2` | The importance weight is cut above `1 + clip_high`. It has no lower bound. |

### `gepa`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `reflection_harness` | string | required | The harness the reflector runs as, in its own Harbor trial. A blank name blocks the run. |
| `reflection_model` | string | unset | The model the reflector's harness uses; unset is the harness's own default. |
| `reflection_image` | string | `"python:3.12-slim"` | The container image the reflector runs in. |
| `modules` | string | `"modules"` | The seed: a modules directory, relative to the blueprint. |
| `minibatch` | integer ≥ 1 | `3` | How many tasks a new candidate is first judged on. |
| `budget` | integer ≥ 1 | unset | Rollouts the search may spend. Unset is two passes: 2 × tasks × `group_size`. |
| `patience` | integer ≥ 1 | `3` | Rounds in a row with no new candidate before the search stops. |
| `edits` | `"rewrite"` or `"incremental"` | `"rewrite"` | `"incremental"` asks the reflector to edit the text in place and keep what works, not rewrite it. |

See [gepa](../concepts/gepa.md).

### `fst`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `slow` | `"dapo"`, `"dr-grpo"` or `"cispo"` | `"cispo"` | The gradient recipe each slow step uses. Its own knobs (`clip_low`, `clip_high`, `clip`, `length_penalty`, `length_floor`) are accepted when they belong to it, with its defaults, except `clip_high` under `cispo`, which is `3.0` here. |
| `cycle` | integer ≥ 1 | `6` | Slow steps per cycle, and the batches each fast phase looks ahead. |
| `population` | integer ≥ 1 | `4` | Texts kept per cycle. It must divide `group_size`. |
| `anchor` | integer ≥ 1 | unset | The fast phase evolves on the first `anchor` tasks of the lookahead; unset is all of them. |
| `kl_coef` | float ≥ 0 | `0.001` | As above, with the paper's default. |
| `budget` | integer ≥ 1 | unset | Rollouts each fast phase may spend. Unset is five passes: 5 × its tasks × `group_size / population`. |
| `edits` | `"rewrite"` or `"incremental"` | `"incremental"` | As for `gepa`, with the paper's default. |

`reflection_harness` (required), `reflection_model`, `reflection_image`, `modules` (the seed, default
`"modules"`), `minibatch` and `patience` are as for `gepa`. See [fst](../concepts/recipes.md#fst).

### `evaluate`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `modules` | string | unset | A modules directory, relative to the blueprint, carried into every job. |

## `[checkpoints]`

Optional. Only the gradient recipes save checkpoints.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `every` | integer ≥ 1 | `1` | Save a checkpoint after every this many steps; a step that trained nothing saves none. A `final` one is saved once the last step is done, whatever `every` says. |
| `ttl_hours` | float ≥ 1 | `168.0` | How long the backend keeps each checkpoint, in hours. The backend keeps one for an hour at least. |

## Not in the file

- `TINKER_API_KEY`: the Tinker-API backend's key, for a served model.
- `TINKER_BASE_URL`: another Tinker-API backend; the default is Tinker's own.
- `SHIPYARD_RUNS`: where runs live, when no `--root` is given.
- `SHIPYARD_QUIET=1`: no progress bars on a terminal.
- `SHIPYARD_PROXY_TOKEN`: the harness's key to the proxy; required with `endpoint_url`.
- `SHIPYARD_CONTROL_TOKEN`: the run's key to point the proxy at weights.

[Commands](cli.md#environment) says which command reads each.

`shipyard` reads `.env` in the working directory first; a variable set in the shell wins.
