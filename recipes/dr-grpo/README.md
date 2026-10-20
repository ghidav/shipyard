# Dr. GRPO (`dr-grpo`)

GRPO without its two biases ([Liu et al., 2025](https://arxiv.org/abs/2503.20783)): each rollout's advantage is its reward minus its group's mean, not divided by the group's spread, and the loss is `ppo`, clipped the same on both sides and summed over the tokens.

| | |
| --- | --- |
| **Agent** | `Qwen/Qwen3-8B` on Tinker, through `pi` |
| **Data** | `reasoning-gym-easy`: 16 tasks a step, 8 rollouts each |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; Modal credentials and `harbor[modal]` (`uv tool install shipyard --with "harbor[modal]"`) |
| **`[recipe]`** | `clip`; `length_penalty`, `length_floor` and `length_cap`, shipyard's length penalty on solved rollouts (off: `length_penalty = 0.0`) |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/dr-grpo
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
