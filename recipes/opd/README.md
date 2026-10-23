# On-policy distillation (`opd`)

On-policy distillation ([Agarwal et al., 2023](https://arxiv.org/abs/2306.13649)): each token's advantage is how much more likely the teacher finds it than the agent does. The rewards are logged, not trained on.

| | |
| --- | --- |
| **Models** | `[models.agent]`: `Qwen/Qwen3-8B` on Tinker, trained, through `pi`; `[models.teacher]`: `Qwen/Qwen3-32B`, which opens a Tinker session of its own, is never trained, and must share the agent's tokenizer: its logprobs score the agent's token ids |
| **Data** | `reasoning-gym-easy`: 16 tasks a step, 4 trials each |
| **Sandbox** | Together (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; `TOGETHER_API_KEY` for the sandboxes, and the `together` extra (`uv tool install "shipyard[together]"`) |
| **`[recipe]`** | `harness`, `group_size` |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/opd
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
