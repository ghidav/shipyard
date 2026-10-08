# Decisions

Each rule below is a deliberate choice in the code. The sentence after it gives the reason: the
silent failure it prevents. The concept pages hold the mechanics.

## Recipes

**A recipe is a name: `dapo`, `dr-grpo` and `cispo` each fix the advantage, the loss, the
clipping, how token losses add up and the optimizer, and `check` prints what the name resolves to.**
An unvalidated combination trains without complaint and looks like a result.

**A gradient recipe's defaults are its paper's, and the docs name the section each comes from, or
say that the paper states none.**
A default from elsewhere makes a run that looks like the paper's train differently.

**Knobs are epsilons, and Tinker gets bounds: `clip_low = 0.2` reaches `loss_fn_config` as
`clip_low_threshold = 0.8`.**
An epsilon passed as a bound clamps every ratio near 0.2 without an error.

**The length rule applies only to `dr-grpo`.**
Under `dapo`'s division by the group's spread, a small length difference among solved answers becomes
a full-size advantage, and the step trains mostly on length.

**`dapo` and `cispo` dock a rollout that sampled near or up to its token budget, after the group is
judged degenerate on its rewards alone.**
DAPO keeps a prompt by its accuracy (Eq. 11), and a group of failures that differ only in how far
they ran would train on length alone if it were judged after the docking.

**A recipe's optimizer is its paper's, and the cookbook's wherever the paper states no value.**
A learning rate tuned with one set of betas, eps and warm-up trains differently with another.

## Training

**The step is shipyard's own, over the Tinker SDK. The cookbook is imported in one adapter module,
for the renderers and the proxy app only.**
A cookbook release cannot change the loss under a run, and shipyard sends the same datums and loss
config to any backend that `TINKER_BASE_URL` names.

**Every step publishes the weights and points the proxy at them, and a batch served by any other
weights stops the run.**
Otherwise an off-policy batch is credited as on-policy.

**Under `reference = "trainer"`, credit refuses a batch if the trainer's weights moved since it was
sampled.**
Otherwise the ratio is formed against weights that produced none of the tokens.

**μ comes from the trainer by default.**
The sampler and the trainer are different engines whose logprobs differ slightly, and recomputing μ
keeps the ratio within one engine.

**Under `reference = "sampler"`, a batch whose records carry no logprobs is refused.**
A gap misaligns every ratio after it.

**A group with one measured rollout, or with equal rewards after `dr-grpo`'s length rule, is dropped
before the reference pass.**
It carries no gradient, and a step left with nothing logs `trained: false`.

**Substeps split the batch by prompt, into at most as many parts as there are groups, and every
substep keeps μ from the weights the batch was sampled at.**
The papers' mini-batches are sets of prompts. With μ read again before each substep, the ratio stays
1 and the clipping never acts.

**`dapo`, `cispo` and `fst` average each prompt's token losses; `dr-grpo` keeps Tinker's sum.**
Summed, a prompt whose rollouts wrote long answers outweighs the others.

**A `dapo` or `cispo` step short of groups with a gradient samples the plan's next batch, and trains
on `batch_size` groups at most.**
Otherwise a step trains on whatever few groups are left, and its batch size drifts with how easy its
tasks are.

**A checkpoint saves the state and the sampler weights, and a `final` one is always saved.**
A record that names only the sampler path can be measured again but never continued.

**A continuation is a new run, started from `[model] from_checkpoint`.**
Each run id then holds the metrics of one process.

**Weights published for sampling expire on their own, after 12 hours.**
A crashed run's leftovers expire without anyone cleaning them up.

## The proxy and the record

**One proxy serves the run's weights and records every call, and a gradient trains on its records.**
The harness's account of what it sent can differ from what the model read.

**Training reads exactly the tokens the proxy rendered and sampled. A call whose prompt does not
extend an earlier call's prompt plus completion starts a new sequence.**
Training on tokens the model never read is a silent off-policy error.

**A narrow bridge keeps a tool loop in one sequence: the same reply, exact token equality, else a
fork.**
Without it, a reply re-rendered as different tokens splits every tool loop into one sequence per
call.

**A reply is identified by its tool calls' ids, not their arguments.**
A harness runs a normalized form of what the model wrote (Claude Code fills in defaults and coerces
types) and echoes that form back. The id the proxy gave the call survives the rewrite. The model then
reads the call it wrote, and the result of the one the harness ran.

**A retry gets the stored answer, with no second sample.**
Otherwise the record holds a sample the harness never saw.

**Tokens live in the environment, never in the blueprint, and only the control token opens
`/control/weights`.**
A secret in `run.toml` is copied into every run directory, and a harness holding the control token
can re-point its own weights.

**The record is append-only: rows are appended and flushed one at a time, and documents are replaced
whole.**
A crash leaves at most a cut last line, which readers skip, and never a half-written document.

**Costs are counts per party: `costs.json` refuses any other key.**
Counts stay true when prices change.

## Rollouts and admission

**Datasets are directories, and a named dataset with no valid task is an error.**
Otherwise a run over zero batches finishes and reports success.

**Harnesses use Harbor's own model connection plus a small profile, and an unknown harness gets the
generic OpenAI wiring with a warning.**
A harness wired by a guess that `check` does not warn about fails inside the sandbox, far from the
blueprint that chose it.

**A masked rollout enters no mean, but a zero does.**
Counted as zeros, failed containers teach the policy that a task is hard.

**A budget cut or a filled context scores 0.**
Both are the policy's doing, and masking them makes rambling look free.

**The proxy records every call it serves, including calls refused for the budget or the context and
calls that failed at the sampler, and a trial's records travel as one object.**
A partial trial is masked rather than trained on as if it were whole.

**Images are served but masked from training.**
An image's stand-in tokens cannot go into a datum.

## Modules and gepa

**A module's kind is its marker file, and a `server.py` or `agent.py` is refused by name.**
A directory that silently delivers nothing looks like a module that did not help.

**A skill must open with frontmatter that names and describes it.**
A harness surfaces a skill only through its frontmatter, so without the check the file is uploaded
and never read.

**A run reads its modules once, when it opens, and writes their digest on every job row.**
Editing the blueprint's modules mid-run cannot change what later jobs carry.

**The reflector is a Harbor trial, and a failed one counts as a decline.**
One broken container cannot end the search, and its cost lands in the record like any other job's.

**gepa holds tasks out to select on: by default two thirds of the run's, as in three of GEPA's four
benchmarks, drawn with `[data] seed`.**
A winner picked on the tasks it was rewritten from reports a score it was fitted to. The sources are
GEPA §4.1 (split sizes) and §4.3 (the validation set is D_pareto).

**Each round's minibatch is drawn in shuffled passes over the feedback tasks.**
In the dataset's listed order, a sorted dataset puts related tasks in every minibatch, and the
listing decides which tasks are reflected on first.

**Each round runs the parent on its minibatch, and the reflector sees those traces.**
Shown another text's traces, the reflector is asked to fix mistakes its own text did not make, and
the child is judged against a score its parent got in another round.

**Fitness is a vector over tasks: the frontier keeps every candidate that is best on at least one
task.**
A text that alone solves a task stays on the frontier, regardless of its mean.

**A child must beat its parent strictly, on the tasks both were measured on.**
A tie gains nothing, and finding that out costs a full scoring.

**An accepted child is scored afresh on every Pareto task.**
Otherwise its row keeps the minibatch scores it won with, and a lucky draw carries it onto the
frontier.

**The winner is chosen by coverage first, then by mean.**
A child whose full scoring mostly failed has a mean over very few tasks, so one lucky task could
otherwise win.

**A round whose parent scores 1.0 on every minibatch task asks for no rewrite.**
A child must beat its parent strictly, and none can there, so the reflection and the child's run are
wasted. GEPA's released code skips such a round by default (gepa 0.1.4, `api.py`:
`skip_perfect_score = True`, `perfect_score = 1.0`).

**The budget counts every rollout and always ends the search; patience is off unless set.**
Every round spends its parent's run, so a search ends regardless of what its reflector answers.
Patience is off by default as in GEPA's released code (gepa 0.1.4, `utils/stop_condition.py`:
`NoImprovementStopper` runs only when passed in `stop_callbacks`). When set, patience is GEPA's
no-improvement stopper: it counts rounds in which the best Pareto mean did not rise.
