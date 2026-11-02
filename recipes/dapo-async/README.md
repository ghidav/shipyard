# DAPO, asynchronously (`dapo-async`)

[`dapo`](../dapo/)'s algorithm with the rollouts overlapping the updates. Workers keep groups rolling out, each step takes the groups in the order they finished until a batch's worth have rewards that differ, and it updates on them while the next ones sample. Each call takes the weights published last when it starts, so a group can straddle an update; `max_lag` leaves out the groups too far behind and gives their tasks out again. It is the reference for an asynchronous loop: copy it and change the loop.

| | |
| --- | --- |
| **Agent** | `[models.agent]`: `Qwen/Qwen3-8B` on Tinker, trained; through `pi`, `[recipe] harness` |
| **Data** | `reasoning-gym-easy`: 16 groups a step, 8 trials each, in 16 substeps; 9 groups rolling out |
| **Sandbox** | Together (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; `TOGETHER_API_KEY` for the sandboxes, and the `together` extra (`uv tool install "shipyard[together]"`) |
| **`[recipe]`** | `harness`, `group_size`, `substeps`, `clip_low`, `clip_high`, `overlong_penalty`, `overlong_buffer`; `overlong_budget` (optional: unset, the model's context), `in_flight` (optional: unset, half again the groups `[rollout] sandboxes` holds), `max_lag` (optional: unset, any lag) |

```sh
uvx harbor datasets download reasoning-gym-easy -o tasks
shipyard run recipes/dapo-async
```

`run.toml` holds every setting; `trainer.py` holds the algorithm and its loop, step by step. Each step's row adds `lag_mean` and `lag_max`, how many updates the groups it trained on lag the weights served, and `stale`, the groups it left out for `max_lag`.
