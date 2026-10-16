"""Training: trainers and recipes, credit assignment, datum packing and the Tinker client.

A user's trainer file subclasses one of the trainers exported here."""

from shipyard.training.recipes.cispo import CISPOTrainer
from shipyard.training.recipes.dapo import DAPOTrainer
from shipyard.training.recipes.dr_grpo import DrGRPOTrainer
from shipyard.training.recipes.fst import fst_trainer
from shipyard.training.recipes.gepa import GEPATrainer
from shipyard.training.trainer import Trainer

__all__ = [
    "CISPOTrainer",
    "DAPOTrainer",
    "DrGRPOTrainer",
    "GEPATrainer",
    "Trainer",
    "fst_trainer",
]
