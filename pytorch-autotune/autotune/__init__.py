"""
pytorch-autotune: LLM-powered systems-knob autotuning for PyTorch training.

Public API:
    tune(train_fn, sweep, ...)  -> TuneResult
    grid(**knobs)               -> SweepGrid
    Suggestion, Recommender     -> recommender protocol (for custom recommenders)
    TuneResult                  -> what tune() returns
"""
from autotune.recommenders.base import Recommender, Suggestion
from autotune.sweep import SweepGrid, grid
from autotune.tune import TuneResult, tune

__version__ = "0.0.1"

__all__ = [
    "tune",
    "grid",
    "SweepGrid",
    "Suggestion",
    "Recommender",
    "TuneResult",
    "__version__",
]
