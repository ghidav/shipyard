from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shipyard.jobs.modules import Modules
from shipyard.run.logs import AGENT


class State(BaseModel):
    """What a run has done so far, which a checkpoint saves and a resume goes on from."""

    #: A field the recipe no longer declares is dropped; one whose type changed refuses to load.
    model_config = ConfigDict(extra="ignore")

    #: Steps taken.
    steps: int = 0
    #: Updates made, by the role of the model they trained: `agent` for the run's own.
    updates: dict[str, int] = Field(default_factory=dict)
    #: Batches drawn.
    drawn: int = 0
    #: Step after which the test set was last evaluated.
    evaluated: int | None = None
    #: The modules every job carries.
    modules: Modules | None = None

    @field_validator("updates", mode="before")
    @classmethod
    def _by_role(cls, value: Any) -> Any:
        """Read a count saved before updates were counted by role as the agent's."""
        if isinstance(value, int) and not isinstance(value, bool):
            return {AGENT: value} if value else {}
        return value
