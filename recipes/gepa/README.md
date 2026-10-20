# GEPA (`gepa`)

GEPA ([Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)), through the `gepa` library: a search over the prompt and skills the agent carries (`modules/`), with no gradient step. A reflector reads a few trials and rewrites one module at a time; the best candidate is kept in the run as `modules/best/`.

| | |
| --- | --- |
| **Agent** | `Qwen/Qwen3-8B` on Tinker, through `pi`, which it never trains; for a provider's agent, set `provider = "anthropic"`, a Claude model and `harness = "claude-code"` |
| **Data** | `reasoning-gym-easy`: 2 rollouts a task; one step, the whole search |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`, `ANTHROPIC_API_KEY` (the reflector); Modal credentials; the `gepa` extra and `harbor[modal]`, in one install: `uv tool install "shipyard[gepa]" --with "harbor[modal]"` |
| **`[recipe]`** | `reflector`, `reflector_provider`, `budget`, `minibatch`; `pareto` (optional: unset, two thirds of the tasks) |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/gepa
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
