# Async loops live in recipes; shipyard gives the blocks

A recipe that overlaps its rollouts with its updates, or streams trials, writes its own loop: how many groups it keeps in flight, the queue between its workers and its steps, the order it takes groups in, when it updates, and what it does with stale groups. Shipyard gives it the blocks:

- the lender, which draws, puts back and retires tasks;
- `rollout`, with each trial's life inside it under two limits, sandboxes alive and trials sampling;
- the weights that answered each call, and a group's lag;
- `update`;
- `spawn`, for tasks the run owns.

The `dapo-async` recipe carries the reference loop, for others to copy. Decided 23 Oct 2026, while designing asynchronous training; the reference was to be a FlashREINFORCE recipe, which stayed an experiment, and became `dapo-async` on 2 Nov 2026, the loop the long runs had used.

## Considered Options

- **A stream in the core, with a take.** The engine would keep its slots busy, and hand a step exactly n complete groups in the order they completed, releasing their tasks and filing their rows at the take. Rejected: loops differ from recipe to recipe, and a take in the core fixes one loop's shape. FlashREINFORCE updates on single trials. DAPO filters groups, and a curriculum chooses tasks.
- **Queues handed to recipes as they are.** Each recipe would have to remember to release tasks, file rows and track weights. Rejected for blocks that keep that bookkeeping out of the loop.

## Consequences

- No loop can skip the bookkeeping, because it lives in the blocks:
  - a task is out until its group completes, with no call to release it;
  - each job writes its row when it completes, under the weights that answered its calls;
  - a step's row counts the groups it passes to `update`;
  - a spawned task fails the run when it raises, and when the run ends it is cancelled, with its trials closed on the proxy.
- A resume skips the tasks whose groups had completed before the checkpoint but no step had taken yet: at most the recipe's groups in flight, and only for that pass over the tasks.
- Warm-ups are a stage of a trial's life inside `rollout`, not something a recipe does. The sandbox limit sets how many trials can get ready ahead of the sampling limit.
