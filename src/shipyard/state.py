from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from shipyard.jobs.modules import Modules


class State(BaseModel):
    """What a run has done so far, which a checkpoint saves and a resume goes on from."""

    #: A field the recipe no longer declares is dropped; one whose type changed refuses to load.
    model_config = ConfigDict(extra="ignore")

    #: Steps taken.
    steps: int = 0
    #: Updates made, by the role of the model they trained.
    updates: dict[str, int] = Field(default_factory=dict)
    #: Batches drawn.
    drawn: int = 0
    #: Step after which the test set was last evaluated.
    evaluated: int | None = None
    #: The modules the run's rollouts carry when a recipe passes them.
    modules: Modules | None = None
