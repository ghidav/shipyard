from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from shipyard.jobs.modules import Modules


class Lending(BaseModel):
    """What the lender has given out, which a checkpoint saves: its place in its seeded order,
    and the tasks out, put back and retired, by path."""

    model_config = ConfigDict(extra="ignore")

    #: Tasks taken from the seeded order so far, retired ones skipped included.
    place: int = 0
    #: Tasks given out whose groups have not completed, once for each time given out.
    out: list[str] = Field(default_factory=list)
    #: Tasks put back, given out again before new ones, in the order put back.
    back: list[str] = Field(default_factory=list)
    #: Tasks given out no more.
    retired: list[str] = Field(default_factory=list)


class State(BaseModel):
    """What a run has done so far, which a checkpoint saves and a resume goes on from."""

    #: A field the recipe no longer declares is dropped; one whose type changed refuses to load.
    model_config = ConfigDict(extra="ignore")

    #: Steps taken.
    steps: int = 0
    #: Updates made, by the role of the model they trained.
    updates: dict[str, int] = Field(default_factory=dict)
    #: What the lender has given out.
    lender: Lending = Field(default_factory=Lending)
    #: Step after which the test set was last evaluated.
    evaluated: int | None = None
    #: The modules the run's rollouts carry when a recipe passes them.
    modules: Modules | None = None
