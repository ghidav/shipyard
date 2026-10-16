"""The parts that move weights: the trainer and its recipes, the credit they assign, the
datums they pack, and the training client. A trainer file subclasses one of these."""

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
