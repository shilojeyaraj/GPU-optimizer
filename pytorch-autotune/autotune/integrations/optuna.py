"""
Optuna integration: LLMSampler.

Plugs the LLMRecommender (or any Recommender) in as an optuna.samplers.BaseSampler
so existing Optuna studies and HuggingFace Trainer.hyperparameter_search calls
can use LLM-driven suggestion with zero other changes.
"""
from __future__ import annotations

from typing import Any

from autotune.recommenders.base import Recommender


def _import_optuna():
    try:
        import optuna

        return optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna integration requires `pip install optuna`."
        ) from exc


class LLMSampler:
    """
    Optuna BaseSampler that asks an LLMRecommender (or any Recommender) for
    the next config given the history of completed trials.

    Usage:
        import optuna
        from autotune.integrations.optuna import LLMSampler

        study = optuna.create_study(
            direction="maximize",
            sampler=LLMSampler(provider="claude"),
        )
        study.optimize(objective, n_trials=50)
    """

    def __init__(
        self,
        provider: str = "claude",
        include_docs: bool = True,
        recommender: Recommender | None = None,
        seed: int | None = None,
    ):
        optuna = _import_optuna()
        self._optuna = optuna
        self._BaseSampler = optuna.samplers.BaseSampler
        self._fallback_sampler = optuna.samplers.RandomSampler(seed=seed)

        if recommender is not None:
            self.recommender = recommender
        else:
            from autotune.recommenders.llm import LLMRecommender

            self.recommender = LLMRecommender(
                provider=provider, include_docs=include_docs
            )

    # Optuna API surface — duck-typed against BaseSampler since we can't
    # subclass at module import time (Optuna may not be installed).

    def infer_relative_search_space(self, study, trial):
        return self._optuna.search_space.intersection_search_space(study)

    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}

        completed_trials = [
            t for t in study.trials
            if t.state == self._optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        history = [(dict(t.params), t.value) for t in completed_trials]

        space_for_recommender = _optuna_space_to_lists(search_space)
        constraints: dict[str, Any] = {}

        suggestion = self.recommender.suggest(
            search_space=space_for_recommender,
            history=history,
            constraints=constraints,
        )

        validated: dict[str, Any] = {}
        for param, dist in search_space.items():
            if param not in suggestion.config:
                continue
            value = suggestion.config[param]
            if isinstance(dist, self._optuna.distributions.CategoricalDistribution):
                if value in dist.choices:
                    validated[param] = value
        return validated

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._fallback_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._fallback_sampler.reseed_rng()


def _optuna_space_to_lists(search_space: dict) -> dict[str, list]:
    """Convert Optuna distributions to the {key: [options]} shape Recommender expects."""
    out: dict[str, list] = {}
    for param, dist in search_space.items():
        choices = getattr(dist, "choices", None)
        if choices is not None:
            out[param] = list(choices)
    return out
