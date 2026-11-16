# On-policy distillation (`opd`)

On-policy distillation ([Agarwal et al., 2023](https://arxiv.org/abs/2306.13649)): each token's advantage is how much more likely the teacher finds it than the agent does. The rewards are logged, not trained on.

| | |
| --- | --- |
| **Models** | `[models.agent]`: `Qwen/Qwen3.5-9B` on Tinker, trained, through `pi`; `[models.teacher]`: `Qwen/Qwen3.6-35B-A3B`, which opens a Tinker session of its own, is never trained, and must share the agent's tokenizer, as it does: its logprobs score the agent's token ids |
| **Data** | `reasoning-gym-easy`: 10 steps of 8 tasks, 4 trials each |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; a Modal login for the sandboxes (`uvx modal setup`), and `cloudflared` or a running Docker for the tunnel that carries their calls to this machine |
| **`[recipe]`** | `harness`, `group_size` |

```sh
uvx harbor@0.24.0 datasets download reasoning-gym-easy -o tasks
shipyard run recipes/opd
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
