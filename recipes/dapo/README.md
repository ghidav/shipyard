# DAPO (`dapo`)

DAPO ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)): each trial's reward, less a soft penalty near the end of its token budget, is normalized within its group; the loss is `ppo`, clipped asymmetrically and averaged per prompt; and a step keeps drawing batches until a batch's worth of groups have something to teach.

| | |
| --- | --- |
| **Agent** | `Qwen/Qwen3-8B` on Tinker, through `pi` |
| **Data** | `reasoning-gym-easy`: 16 tasks a step, 8 trials each, in 16 substeps |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; Modal credentials and `harbor[modal]` (`uv tool install shipyard --with "harbor[modal]"`) |
| **`[recipe]`** | `clip_low`, `clip_high`, `overlong_penalty`, `overlong_buffer`; `overlong_budget` (optional: unset, the model's context) |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/dapo
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
