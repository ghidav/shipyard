"""DAPO, with Claude's feedback on each failed rollout read by the policy as its own teacher.

Claude reads a failed attempt (its score, the task, the end of the transcript and what the
tests printed) and says what the agent should have done. The policy, given that feedback
before the task, scores the tokens it sampled without it, and each token's advantage gains
`teacher_coef × (teacher − reference)`. A solved rollout gets no feedback and trains by DAPO
alone.

Needs the `anthropic` package and `ANTHROPIC_API_KEY`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from anthropic import AsyncAnthropic

from shipyard.config import Blueprint
from shipyard.training import DAPOTrainer
from shipyard.types import Group, Member

#: The model that writes the feedback.
CRITIC = "claude-sonnet-5-5"
#: Most feedback requests in flight at once. A job's failed rollouts are critiqued together.
CONCURRENCY = 8
ASK = """Here is an agent's failed attempt at a task.

{attempt}
In a few sentences, tell the agent what it should have done differently."""


class FeedbackDAPO(DAPOTrainer):
    def __init__(self, config: Blueprint, directory: Path) -> None:
        super().__init__(config, directory)
        self.claude = AsyncAnthropic()
        self.gate = asyncio.Semaphore(CONCURRENCY)

    async def privileged(self, member: Member, group: Group) -> str | None:
        """Return Claude's feedback on a failed rollout, or None for a solved one."""
        if member.verdict.reward >= 1.0:
            return None
        async with self.gate:
            reply = await self.claude.messages.create(
                model=CRITIC,
                max_tokens=1024,
                messages=[{"role": "user", "content": ASK.format(attempt=self.attempt(member))}],
            )
        usage = reply.usage
        self.spent("anthropic", input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        return f"Feedback on an earlier attempt at this task:\n{reply.content[0].text}\n\n"
