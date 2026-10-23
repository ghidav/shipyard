# run

A run's directory and everything it writes: the config, the logs, the costs, the checkpoints and their states. Recipes reach it through `self.metrics`, `keep` and `self.logs.spend`.

```mermaid
flowchart LR
    data["Generator: the tasks, in batches"] --> draw["draw()"]
    draw -->|tasks| rollout["rollout(tasks, model, harness, ...)"]
    rollout -->|groups| score["the recipe's scoring"]
    score -->|advantages| update["update(model, groups)"]
    update -.->|new weights| rollout
    rollout -->|"jobs, requests, spend"| dir["the run's directory"]
    update -->|"the step's row, checkpoints"| dir
    dir -.->|resume| draw
```

| File | Holds |
| --- | --- |
| `directory.py` | `open_run`, `continued`, `reopened`, `running`; `recipe_of`, the trainer file a config runs with, and `trainer_in`, which imports a recipe's folder as its own package |
| `dry.py` | `checked`, for `shipyard run --dry`: the run's checks on a trainer made in a scratch directory, and the value the run takes for each key it decides elsewhere (a LoRA rank, a renderer, a context, the tasks' and harnesses' timeouts) |
| `logs.py` | `Logs`: the job, step and request rows, the usage ledger and the costs, and their `Wandb` mirror |
| `costs.py` | `Costs`: tokens and sandbox time, totalled from the usage ledger |
| `files.py` | the names of the run directory's files, its atomic JSON writes and JSONL appends |
| `data.py` | `tasks`, `Generator`: Harbor tasks, in resumable batches |