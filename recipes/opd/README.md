# On-policy distillation (`opd`)

On-policy distillation ([Agarwal et al., 2023](https://arxiv.org/abs/2306.13649)): each token's advantage is how much more likely the teacher finds it than the policy does, times `teacher_coef`. The rewards are logged, not trained on.

| | |
| --- | --- |
| **Agent** | `Qwen/Qwen3-8B` on Tinker, through `pi`; the teacher, `Qwen/Qwen3-32B`, is frozen in the same Tinker session and must share the student's tokenizer |
| **Data** | `reasoning-gym-easy`: 16 tasks a step, 4 rollouts each |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; Modal credentials and `harbor[modal]` (`uv tool install shipyard --with "harbor[modal]"`) |
| **`[recipe]`** | `teacher`, `teacher_coef` |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/opd
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
