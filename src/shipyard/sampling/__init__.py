"""The parts that produce rollouts: the endpoint a harness dials, the engine that runs it,
the secrets each trial holds, and the admission that judges what came back."""

from shipyard.sampling.secrets import Pool, Secrets

__all__ = ["Pool", "Secrets"]
