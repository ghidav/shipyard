# On-policy distillation (`opd`)

On-policy distillation ([Agarwal et al., 2023](https://arxiv.org/abs/2306.13649)): each token's advantage is how much more likely the teacher finds it than the agent does. The rewards are logged, not trained on.

| | |
| --- | --- |
| **Agent** | `Qwen/Qwen3-8B` on Tinker, through `pi`; the teacher, `Qwen/Qwen3-32B`, opens a Tinker session of its own, is never trained, and must share the agent's tokenizer, which the run checks when it starts |
| **Data** | `reasoning-gym-easy`: 16 tasks a step, 4 trials each |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; Modal credentials and `harbor[modal]` (`uv tool install shipyard --with "harbor[modal]"`) |
| **`[recipe]`** | `teacher` |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/opd
```

`run.toml` holds every setting; `trainer.py` holds the algorithm, step by step.
