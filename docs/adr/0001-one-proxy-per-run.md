# One proxy per run, with each trial opened on it

A run starts one proxy for all its trials. The run opens each trial on that proxy with its own lane, renderer and weights, and closes it to take its records. The costly parts are shared: the renderer and its tokenizer, the sampling sessions, the port and the tunnel. What belongs to one trial stays with that trial. Decided 21 Oct 2026, after measuring the alternatives.

## Considered Options

- **A proxy per trial, as a process on the run's machine**: each one costs 0.49 GB and at least 4 s before its Tinker session opens (1.2 s to import, 2.8 s to build the Qwen3-8B renderer). Off Docker, each needs a quick tunnel of its own. A crash would quietly mask one trial instead of failing the run. verifiers ran a server and a tunnel per rollout, measured it, and moved to pools of 32 rollouts (PrimeIntellect-ai/verifiers PR #1605).
- **A proxy inside each sandbox**: Harbor runs the agent as the image's user, root by default. The model under training could read the Tinker key, stop the proxy, or rewrite the records it is trained on. Every task image would also need Python, torch and transformers.
- **A proxy per job**: pins weights per rollout, but pays start-up, memory and a tunnel for every rollout.

## Consequences

The one proxy's limit is CPU on a single core. Re-tokenizing every conversation on every call saturated it at 32 trials with 55k-token prompts. Tokenizing each message once per trial cut its CPU 4.7× and left the records byte-identical. A run that outgrows one core should spread its trials over a few processes of the one proxy, each trial pinned to one of them, and never give each trial a proxy of its own.
