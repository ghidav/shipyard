from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from shipyard.jobs.modules import Modules


class State(BaseModel):
    """What a run has done so far: what a checkpoint saves, and a resume goes on from. A recipe
    with state of its own, such as a memory bank, declares it in a subclass,
    `class State(Trainer.State)`, each field with a default. A checkpoint saves it as JSON:
    numbers, text, paths, lists and dicts of them, pydantic models, and `Modules`, whole."""

    #: A field the recipe no longer declares is dropped; one whose type changed refuses to load.
    model_config = ConfigDict(extra="ignore")

    #: Steps taken, updates made, and batches drawn.
    steps: int = 0
    updates: int = 0
    drawn: int = 0
    #: Step after which the test set was last evaluated.
    evaluated: int | None = None
    #: The modules every job carries.
    modules: Modules | None = None
