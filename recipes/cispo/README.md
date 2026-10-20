# CISPO (`cispo`)

CISPO ([MiniMax-M1, 2025](https://arxiv.org/abs/2506.13585)): DAPO's rules with Tinker's `cispo` loss, which truncates each token's importance weight, and stops its gradient, rather than clipping the token's update, so every token keeps contributing.

| | |
| --- | --- |
| **Agent** | `Qwen/Qwen3-8B` on Tinker, through `pi` |
| **Data** | `reasoning-gym-easy`: 16 tasks a step, 8 rollouts each, in 16 substeps |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; Modal credentials and `harbor[modal]` (`uv tool install shipyard --with "harbor[modal]"`) |
| **`[recipe]`** | `clip_high` (the weight is truncated at 1 + `clip_high`), `overlong_penalty`, `overlong_buffer`; `overlong_budget` (optional) |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/cispo
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
