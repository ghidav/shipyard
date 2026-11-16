# RFT, asynchronously (`rft-async`)

Rejection-sampling fine-tuning ([Yuan et al., 2023](https://arxiv.org/abs/2308.01825)): the agent is trained by cross-entropy on correct rollouts, and on nothing else. Workers keep groups rolling out while each step trains on the correct trials of the groups that finished, every kept token weighing the same. With `[models.teacher]` the teacher samples and the agent learns its successes; without it, the agent learns from its own.

| | |
| --- | --- |
| **Agent** | `[models.agent]`: `Qwen/Qwen3.5-9B` on Tinker, trained |
| **Teacher** | `[models.teacher]` (optional): `Qwen/Qwen3.6-35B-A3B` on Tinker, sampling. It must be served through the proxy, on Tinker or Together, and share the agent's tokenizer and renderer, since the agent trains on its token ids; these two do, with the `qwen3_5` renderer |
| **Data** | `reasoning-gym-easy`: 10 steps of 8 groups that hold a correct trial, 8 trials each, through `pi`, `[recipe] harness`; 9 groups rolling out |
| **Sandbox** | Modal (set `[rollout] sandbox = "docker"` to run trials on this machine) |
| **Needs** | `TINKER_API_KEY`; a Modal login for the sandboxes (`uvx modal setup`), and `cloudflared` or a running Docker for the tunnel that carries their calls to this machine |
| **`[recipe]`** | `harness`, `group_size`; `threshold` (optional: unset, 1, solved), `keep_per_group` (optional: unset, every correct trial), `in_flight` (optional: unset, half again the groups `[rollout] sandboxes` holds) |

```sh
uvx harbor@0.24.0 datasets download reasoning-gym-easy -o tasks
shipyard run recipes/rft-async
```

`run.toml` holds every setting; `trainer.py` holds the algorithm and its loop, step by step. Each step's row adds `kept`, the trials trained on, and `kept_tokens`, their tokens; its `reward_mean` is the sampler's solve rate on the groups the step took.
