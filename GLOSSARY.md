# Shipyard

Shipyard post-trains agents on Harbor tasks: a recipe runs a model through the harnesses it will be used in, grades each trial with the task's tests, and trains on exactly the tokens the model sampled.

## Language

### Runs

**Recipe**:
An algorithm written as a trainer that writes the step, and the evaluation when the config has a test set, together with the config that runs it.
_Avoid_: method, algorithm (what a recipe holds)

**Config**:
Every setting of a run, in its `run.toml`: the models it reaches, its data, how its trials run, and the recipe's own keys, such as its harnesses; checked against the recipe before anything opens.
_Avoid_: blueprint, settings

**Run**:
One execution of a recipe on a config, and the directory that records it.
_Avoid_: experiment, job (a job is part of a rollout)

**Step**:
One turn of a run, as the recipe writes it: most draw a batch, roll it out, score its sequences, and update the models in training.
_Avoid_: iteration, epoch

**Draw**:
Taking the next batch of tasks from the training dataset.

**Rollout**:
Running a model on a batch of tasks at one harness and renderer, as one job, and returning a group per task. A recipe that wants several harnesses or renderers runs several rollouts.
_Avoid_: rollout for a single trial (say trial)

**Score**:
Giving each sequence its advantages, which is the part of the step most recipes write.
_Avoid_: score for a model's logprobs

**Update**:
One training call on a model in training: its groups split into substeps at one learning rate, then its new weights served.

**Substep**:
One optimizer step of an update, on a share of its groups, whole groups each; the recipe sets how many.

**Evaluation**:
The recipe's rollouts of the test dataset, every few steps and after the last, that nothing trains on.
_Avoid_: test (the dataset), validation

**State**:
What a run has done so far: its steps, its updates by role, the batches it drew, when it last evaluated, and the modules it carries.
_Avoid_: training state (a model's)

**Checkpoint**:
A named point a run can go on from: the run's state, and each trained model's training state and sampler weights.
_Avoid_: snapshot, save

**Resume**:
Starting a new run from another run's checkpoint; the run it came from is its origin.
_Avoid_: continue, restart, reopen

### Tasks and trials

**Task**:
A Harbor task: an instruction, an environment, and the tests that grade an attempt at it.
_Avoid_: problem, prompt (the prompt is a module)

**Dataset**:
A named set of tasks; a config names one or more to train on and to test on.

**Batch**:
The tasks one step draws.
_Avoid_: mini-batch (a share of an update)

**Group size**:
How many trials each task gets in a rollout; a key of the recipe.
_Avoid_: attempts, rollouts per task

**Harness**:
The agent program a trial runs, such as Claude Code or pi, at a pinned version, with the environment and keyword arguments it starts with; it calls the model as it would a provider's API. A recipe declares the harnesses it runs as keys of its own.
_Avoid_: agent (the agent is the model), client

**Renderer**:
The way a model's conversation becomes tokens and back; a model's table lists the ones its lane holds, and a rollout uses one of them.

**Job**:
A rollout's trials, run together in Harbor, and the row the run writes for them.

**Trial**:
One harness running one task once, in a sandbox, graded by the task's tests.
_Avoid_: rollout, attempt, episode

**Sandbox**:
Where a trial runs: a container or VM started from its task's image, on this machine (Docker) or a provider's (Together, Modal); `[rollout] sandbox` names it.
_Avoid_: environment (Harbor's class that drives one), container

**Hooks**:
Code a task carries to run outside its sandboxes at events of each of its trials, such as preparing an outside system the task uses before the agent starts and tearing it down once the trial is done.
_Avoid_: callbacks, setup scripts

**Group**:
One task's trials in a rollout, whose rewards are compared with each other.

**Sequence**:
One chain of the agent's calls within a trial, as the tokens it read and sampled, with an advantage on each sampled token; what an update trains on.
_Avoid_: chain, datum, sample

**Verdict**:
A trial's reward, or the reason it is masked.

**Reward**:
The number a task's tests give a trial.
_Avoid_: score (what a recipe does with rewards)

**Masked**:
Said of a trial whose outcome says nothing about the agent, because the sandbox, the grader or the model's server failed; it has no reward and is not trained on.
_Avoid_: dropped, filtered, discarded

**Graded**:
Said of a trial with a verdict that is not masked: its reward counts, and its sequences can be trained on.
_Avoid_: measured, counted

### Models

**Model**:
A model the run reaches, named in the config under a role, which the recipe rolls out through harnesses, asks for logprobs, or trains.
_Avoid_: policy

**Role**:
The name the config gives a model, as `[models.<role>]`, such as agent, teacher or critic: the recipe reads the model by it, and its lane, spend and saves are filed under it.

**Agent**:
The role recipes give, by convention, to the model they roll out on the tasks; no role means more to the run than another.
_Avoid_: policy, student, and agent for a harness (Harbor's usage)

**Trainable**:
Said of a model a run could train: a base model or training state on Tinker or Together, not sampler weights.

**In training**:
Said of a model the recipe trains: it marks it so in its `__init__`, before the run starts, and the run opens a training session for it. Any trainable model may be.

**Provider**:
Who serves a model and bills its tokens: Tinker or Together, which shipyard can train on, or an API such as Anthropic's, which it only calls.

**Backend**:
What a model runs on once the run binds it: its session on Tinker or Together, or a provider's API.
_Avoid_: client (the training client is one part of a session), agent (every model has a backend, not only the agent)

**Session**:
A model's connection to its provider's service on Tinker or Together, which trains it or serves it as it is, and saves its training state.
_Avoid_: client

**Training state**:
A model's weights and optimizer state, saved so its training can go on.
_Avoid_: weights (alone), checkpoint (a run's)

**Sampler weights**:
A model's weights saved for sampling only: they can be served and scored, not trained further.
_Avoid_: sampler (the service that samples), inference checkpoint

**Sampler replica**:
One of the copies of a model's sampler, the service that answers its calls; a model with more sampler replicas answers a crowded rollout faster.
_Avoid_: generator (draws the run's tasks)

### The proxy

**Proxy**:
The one model server a run puts between the harnesses and the models it trains or scores; it stands in for a provider's API, answers only the trials the run has opened on it, and records their every call.
_Avoid_: endpoint, gateway

**Lane**:
One model the proxy serves, named after its role, with its own renderers and sampling; the weights it samples are each open trial's own.

**Open**:
Said of a trial the run has opened on the proxy, on the lane, renderer and weights its calls are answered with until the run closes it.
_Avoid_: registered, active, started

**Endpoint**:
What a trial's harness dials to reach a model: the run's proxy, which knows the trial by the token it was opened with, or a provider's API.

**Call**:
One request from a harness to a model, answered with one reply.
_Avoid_: request, turn (one message of a conversation), tool call

**Records**:
What the proxy recorded of one trial's calls, each answered, turned away or failed at the sampler, handed to the run when it closes the trial; or that it recorded none because a provider served the trial.
_Avoid_: calls (reads as tool or function calls), served (already names the weights that answered a call)

**Turned away**:
Said of a call the proxy declined without sampling, because the trial's budget or the model's context could not fit it.
_Avoid_: rejected, refused (also covers malformed calls)

**Budget**:
The most tokens one trial may sample across all its calls: as many as the model's context holds.
_Avoid_: limit

**Context**:
The most tokens one call's prompt and reply together may take: the model's context length.
_Avoid_: window

**Bridge**:
Building a call's prompt from the tokens of an earlier reply the conversation quotes, so the sequence holds exactly what the model sampled.
_Avoid_: splice, extend

### Modules and costs

**Modules**:
The prompt and skills a trial carries into its harness, which a recipe may search over instead of the weights.
_Avoid_: components

**Module**:
One of a trial's modules: its prompt, or one skill.

**Costs**:
A run's tokens, by model and kind (prefill, cache, sample, train), and its sandbox time.
_Avoid_: bill, usage (the per-call ledger costs are totalled from)
