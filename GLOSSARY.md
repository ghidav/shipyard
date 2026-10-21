# Shipyard

Shipyard post-trains agents on Harbor tasks: a recipe runs a model through the harnesses it will be used in, grades each trial with the task's tests, and trains on exactly the tokens the model sampled.

## Language

### Runs

**Recipe**:
An algorithm written as a trainer that overrides parts of the step, together with the blueprint that runs it.
_Avoid_: method, algorithm (what a recipe holds)

**Blueprint**:
Every setting of a run, in one file, checked against the recipe before anything opens.
_Avoid_: config, settings

**Run**:
One execution of a recipe on a blueprint, and the directory that records it.
_Avoid_: experiment, job (a job is part of a rollout)

**Step**:
One turn of a run: draw a batch, roll it out, score its sequences, update the models in training.
_Avoid_: iteration, epoch

**Draw**:
Taking the next batch of tasks from the training dataset.

**Rollout**:
Running the agent on a batch of tasks, through each harness and renderer, and returning a group per task.
_Avoid_: rollout for a single trial (say trial)

**Score**:
Giving each sequence its advantages, which is the part of the step most recipes write.
_Avoid_: score for a model's logprobs

**Update**:
One training call for each model in training, on the step's scored sequences.

**Evaluation**:
A rollout of the test dataset, every few steps, that nothing trains on.
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
A named set of tasks; a blueprint names one or more to train on and to test on.

**Batch**:
The tasks one step draws.
_Avoid_: mini-batch (a share of an update)

**Group size**:
How many trials each task gets in a rollout, at each harness and renderer.
_Avoid_: attempts, rollouts per task

**Harness**:
The agent program a trial runs, such as Claude Code or pi, at a pinned version; it calls the model as it would a provider's API.
_Avoid_: agent (the agent is the model), client

**Renderer**:
The way a model's conversation becomes tokens and back; one model may be served with several.

**Job**:
A rollout's trials at one harness and renderer, run together in Harbor.

**Trial**:
One harness running one task once, in a sandbox, graded by the task's tests.
_Avoid_: rollout, attempt, episode

**Group**:
One task's trials in a rollout, at one renderer, whose rewards are compared with each other.

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
A model the run reaches, held by the trainer under a role, which the run rolls out through harnesses or asks for logprobs.
_Avoid_: policy

**Role**:
The name a trainer holds a model under, such as agent, teacher or critic; each model's spend and saves are filed under it.

**Agent**:
The run's own model: the one it rolls out and, unless the recipe says otherwise, trains.
_Avoid_: policy, student, and agent for a harness (Harbor's usage)

**Trainable**:
Said of a model a run could train: a base model or training state on Tinker or Together, not sampler weights.

**In training**:
Said of a model this run trains; set by the recipe before the run starts.

**Provider**:
Who serves a model and bills its tokens: Tinker or Together, which shipyard can train on, or an API such as Anthropic's, which it only calls.

**Backend**:
What a model runs on once the run binds it: a session on the agent's training service, a sampling session for another model on Tinker, or a provider's API.
_Avoid_: client (the training client is one part of a session), agent (every model has a backend, not only the agent)

**Session**:
The agent's connection to its provider's training service, which trains the agent or serves it as it is, and saves its training state.
_Avoid_: client

**Training state**:
A model's weights and optimizer state, saved so its training can go on.
_Avoid_: weights (alone), checkpoint (a run's)

**Sampler weights**:
A model's weights saved for sampling only: they can be served and scored, not trained further.
_Avoid_: sampler (the service that samples), inference checkpoint

### The proxy

**Proxy**:
The model server the run puts between the harnesses and the models it trains or scores; it stands in for a provider's API and records every call.
_Avoid_: endpoint, gateway

**Lane**:
One model the proxy serves, named after its role, with its own weights, renderers and records.

**Endpoint**:
What a trial's harness dials to reach a model: a lane of the run's proxy, or a provider's API.

**Call**:
One request from a harness to a model, answered with one reply.
_Avoid_: request, turn (one message of a conversation), tool call

**Records**:
What the proxy recorded of one trial's calls, each answered, turned away or failed at the sampler, or that it recorded none because a provider served the trial.
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
