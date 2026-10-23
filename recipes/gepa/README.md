# GEPA (`gepa`)

GEPA ([Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)), through the `gepa` library: a search over the prompt and skills the agent carries (`modules/`), with no gradient step. To propose a candidate, the reflector is rolled out through its own harness at a Harbor task for each module it rewrites (`reflect.py`): the task holds the module and a few tasks' attempts as files, and the reflector rewrites the module in place. The best candidate is kept in the run as `modules/best/`, and each proposal's tasks under `gepa/reflections/`.

| | |
| --- | --- |
| **Models** | `[models.agent]`: `Qwen/Qwen3-8B` on Tinker, through `pi`, which it never trains (for a provider's agent, set `provider = "anthropic"`, a Claude model, and `[recipe] harness = "claude-code"`); `[models.reflector]`: the model that rewrites the modules |
| **Data** | `reasoning-gym-easy`: 2 trials a task; one step, the whole search |
| **Sandbox** | Together (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`, `ANTHROPIC_API_KEY` (the reflector's harness); `TOGETHER_API_KEY` for the sandboxes; the `gepa` and `together` extras, in one install: `uv tool install "shipyard[gepa,together]"` |
| **`[recipe]`** | `harness`, `group_size`, `reflector_harness` (a name, or a table with `name`, `env` and `kwargs`), `budget`, `minibatch`; `pareto` (optional: unset, two thirds of the tasks) |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/gepa
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
