"""
HuggingFace Trainer integration. Planned for v0.2.

HF Trainer already supports Optuna via `trainer.hyperparameter_search(backend="optuna")`.
This module will expose a one-line helper that wires LLMSampler into that flow:

    from autotune.integrations.hf_trainer import use_llm_sampler

    best = trainer.hyperparameter_search(
        hp_space=lambda trial: {...},
        backend="optuna",
        sampler=use_llm_sampler(provider="claude"),
    )

Stubbed now so the import surface is documented.
"""
from __future__ import annotations


def use_llm_sampler(*args, **kwargs):
    raise NotImplementedError(
        "HF Trainer integration ships in v0.2. "
        "Use autotune.integrations.optuna.LLMSampler directly for now."
    )
