# GRPO (`grpo`)

GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) on the tasks' verifiable rewards: each trial's advantage is its reward minus its group's mean, a group of equal rewards teaches nothing, and the loss is `importance_sampling` against the sampler's logprobs.

| | |
| --- | --- |
| **Agent** | `[models.agent]`: `Qwen/Qwen3-8B` on Tinker, trained; through `pi`, `[recipe] harness` |
| **Data** | `reasoning-gym-easy`: 16 tasks a step, 8 trials each |
| **Sandbox** | Together (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; `TOGETHER_API_KEY` for the sandboxes, and the `together` extra (`uv tool install "shipyard[together]"`) |
| **`[recipe]`** | `harness`, `group_size`; `test_group_size` (optional: unset, `group_size`) |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/grpo
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
