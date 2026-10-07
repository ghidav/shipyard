# Decisions

Each rule below is a deliberate choice in the code. The sentence after it gives the reason: the
silent failure it prevents. The concept pages hold the mechanics.

## Recipes

**A recipe is a name: `dapo`, `dr-grpo` and `cispo` each fix the advantage, the loss and the
clipping, and `check` prints what the name resolves to.**
A combination nobody has validated would train without complaint and look like a result.

**Knobs are epsilons, and Tinker gets bounds: `clip_low = 0.2` reaches `loss_fn_config` as
`clip_low_threshold = 0.8`.**
An epsilon passed as a bound would clamp every ratio near 0.2, and nothing would report an error.

**The length rule lives only in `dr-grpo`.**
Under `dapo`'s division by the group's spread, a small length difference among solved answers would
become a full-size advantage, and the step would mostly train on length.

## Training

**The step is shipyard's own, over the Tinker SDK; the cookbook is imported in one adapter module,
for the renderers and the proxy app only.**
A cookbook release cannot change the loss under a run, and shipyard sends the same datums and loss
config to any backend that `TINKER_BASE_URL` names.

**Every step publishes the weights and points the proxy at them, and a batch served by any other
weights stops the run.**
Otherwise an off-policy batch would be credited as on-policy.

**Under `reference = "trainer"`, credit refuses a batch if the trainer's weights moved since it was
sampled.**
Otherwise the ratio would be formed against weights that produced none of the tokens.

**μ comes from the trainer by default.**
The sampler and the trainer are different engines whose logprobs differ slightly, and recomputing μ
keeps the ratio within one engine.

**Under `reference = "sampler"`, a batch whose records carry no logprobs is refused.**
A gap would misalign every ratio after it.

**A group with one measured rollout, or with equal rewards after any length rule, is dropped before
the reference pass.**
It carries no gradient, and a step left with nothing logs `trained: false`, so it never reads like one
that trained.

**Substeps draw on the whole batch: it is shuffled, the same way in every process, before it is
split.**
Unshuffled, each substep would follow the gradient of only a few tasks.

**A checkpoint saves the state and the sampler weights, and a `final` one is always saved.**
A record that named only the sampler path could be measured again but never continued.

**A continuation is a new run, started from `[model] from_checkpoint`.**
Each run id then holds the metrics of one process.

**Weights published for sampling expire on their own, after 12 hours.**
A crashed run's leftovers expire without anyone cleaning them up.

## The proxy and the record

**One proxy serves the run's weights and records every call, and a gradient trains on its records.**
The harness's own account of what it sent could differ from what the model read, and nothing would
say so.

**Training reads exactly the tokens the proxy rendered and sampled: a call whose prompt does not
extend an earlier call's prompt plus completion starts a new sequence.**
Training on tokens the model never read would be a silent off-policy error.

**A narrow bridge keeps a tool loop in one sequence: exact message match, exact token equality, else
a fork.**
Without it, a reply re-rendered as different tokens would split every tool loop into one sequence per
call.

**A retry gets the stored answer, with no second sample.**
Otherwise the record would hold a sample the harness never saw.

**Tokens live in the environment, never in the blueprint, and only the control token opens
`/control/weights`.**
A secret in `run.toml` would be copied into every run directory, and a harness holding the control
token could re-point its own weights.

**The record is append-only: rows are appended and flushed one at a time, and documents are replaced
whole.**
A crash leaves at most a cut last line, which readers skip, and never a half-written document.

**Costs are counts per party: `costs.json` refuses any other key.**
Counts stay true when prices change, so nothing in the record goes stale.

## Rollouts and admission

**Datasets are directories, and a named dataset with no valid task is an error.**
Otherwise a run over zero batches would finish and report success.

**Harnesses use Harbor's own model connection plus a small profile, and an unknown harness gets the
generic OpenAI wiring with a warning.**
A harness wired by a guess that `check` kept quiet about would fail inside the sandbox, far from the
blueprint that chose it.

**A masked rollout enters no mean, but a zero does.**
Counted as zeros, failed containers would teach the policy that a task is hard.

**A budget cut or a filled context scores 0.**
Both are the policy's own doing, and masking them would make rambling look free.

**The proxy records every call it serves, including calls refused for the budget or the context and
calls that failed at the sampler, and a trial's records travel as one object.**
A partial trial is masked rather than trained on as if it were whole.

**Images are served but masked from training.**
An image's stand-in tokens cannot go into a datum.

## Modules and gepa

**A module's kind is its marker file, and a `server.py` or `agent.py` is refused by name.**
A directory that silently delivered nothing would look like a module that did not help.

**A skill must open with frontmatter that names and describes it.**
A harness surfaces a skill only through its frontmatter, so without the check the file would be
uploaded and never read.

**A run reads its modules once, when it opens, and writes their digest on every job row.**
Editing the blueprint's modules mid-run cannot change what later jobs carry.

**The reflector is a Harbor trial, and a failed one counts as a decline.**
One broken container cannot end the search, and its cost lands in the record like any other job's.

**The reflector sees its parent's own traces.**
Shown another text's traces, it would be asked to fix mistakes its own text did not make.

**Fitness is a vector over tasks: the frontier keeps every candidate that is best on at least one
task.**
A text that alone solves a task stays on the frontier, whatever its mean.

**A child must beat its parent strictly, on the tasks both were measured on.**
A tie buys nothing, and finding that out would cost a full scoring.

**The winner is chosen by coverage first, then by mean.**
A child whose full scoring mostly failed has a mean over very few tasks, so one lucky task could
otherwise win.

**The budget is in rollouts, and patience is in quiet rounds.**
Rounds that score no child spend no rollouts, so the budget alone would never stop them.
