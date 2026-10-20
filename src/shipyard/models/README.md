# models

The policies' backends: the agent's training client, frozen samplers, providers' models, and the proxy that serves an agent on a training client to its harnesses. 

```mermaid
flowchart LR
    draw["draw()"] -->|tasks| rollout["rollout(tasks)"]
    rollout -->|groups| score["score(groups)"]
    score -->|advantages| update["update(groups)"]
    update -->|trains| client["the agent's training client"]
    client -.->|new weights| proxy["harness-proxy"]
    proxy -->|records| rollout
    provider["a provider's API"] -.->|an agent on a provider| rollout
    client -->|logprobs| score
    frozen["frozen policies"] -->|logprobs| score
```

| Module | Holds |
| --- | --- |
| `agent.py` | `Session`, `Frozen`, `Provider`: the backends a `Policy` binds to |
| `clients/` | the agent's training client by `[agent] provider` (`client_for`): `TinkerClient`, `TogetherClient` |
| `endpoints.py` | `ProxyEndpoint`, `ProviderEndpoint`: what a trial's harness dials |
| `context.py` | `insert`: text spliced into a sequence just before the task's instruction |