# GEPA (`gepa`)

GEPA ([Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)), through the `gepa` library: a search over the prompt and skills the agent carries (`modules/`), with no gradient step. To propose a candidate, the reflector is rolled out through its own harness at a Harbor task for each module it rewrites (`reflect.py`): the task holds the module and a few tasks' attempts as files, and the reflector rewrites the module in place. The best candidate is kept in the run as `modules/best/`, and each proposal's tasks under `gepa/reflections/`.

| | |
| --- | --- |
| **Agent** | `Qwen/Qwen3-8B` on Tinker, through `pi`, which it never trains; for a provider's agent, set `provider = "anthropic"`, a Claude model and `harness = "claude-code"` |
| **Data** | `reasoning-gym-easy`: 2 trials a task; one step, the whole search |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`, `ANTHROPIC_API_KEY` (the reflector's harness); Modal credentials; the `gepa` extra and `harbor[modal]`, in one install: `uv tool install "shipyard[gepa]" --with "harbor[modal]"` |
| **`[recipe]`** | `budget`, `minibatch`; `pareto` (optional: unset, two thirds of the tasks); and `[recipe.reflector]`: `model`, `provider`, `harness`, with `env` and `kwargs` for its harness (optional) |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/gepa
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
