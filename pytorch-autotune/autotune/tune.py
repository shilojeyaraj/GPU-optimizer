"""
tune() entrypoint — coordinates sweep, recommender, and backend.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from autotune.backends.local import TrialResult, run_trial
from autotune.recommenders.base import Recommender, Suggestion
from autotune.recommenders.heuristic import HeuristicRecommender
from autotune.recommenders.random import RandomRecommender
from autotune.sweep import SweepGrid


@dataclass
class TuneResult:
    best_config: dict | None
    best_throughput: float | None
    all_results: list[dict] = field(default_factory=list)
    recommendation: Suggestion | None = None


def tune(
    train_fn: Callable[[dict], float],
    sweep: SweepGrid,
    recommender: str | Recommender | None = "heuristic",
    backend: str = "local",
    n_trials: int = 20,
    constraints: dict | None = None,
    seed: int = 42,
    timeout_seconds: int = 3600,
) -> TuneResult:
    """
    Coordinate a sweep of training configs and return the best.

    Args:
        train_fn: callable taking a config dict, returns samples/sec. Must be
            an importable module-level function (the local backend imports it
            in a fresh subprocess).
        sweep: SweepGrid defining the search space
        recommender: "claude" | "gpt-4o" | "heuristic" | "random" | None,
            or a Recommender instance. None = pure grid order.
        backend: "local" (only option for now). Ray/Slurm in future versions.
        n_trials: maximum number of configs to evaluate
        constraints: hard constraints e.g. {"max_vram_gb": 24}
        seed: random seed
        timeout_seconds: kill a trial subprocess after this many seconds

    Returns:
        TuneResult.
    """
    if backend != "local":
        raise NotImplementedError(
            f"backend={backend!r} not yet supported. v0.1.0 ships local only; "
            "Ray and Slurm backends planned for v0.2."
        )

    constraints = constraints or {}
    recommender_obj = _resolve_recommender(recommender, seed=seed)
    train_fn_module, train_fn_name = _resolve_train_fn_path(train_fn)

    search_space = sweep.params
    history: list[tuple[dict, float]] = []
    all_results: list[dict] = []
    best_config: dict | None = None
    best_throughput: float | None = None

    for trial_idx in range(n_trials):
        if recommender_obj is None:
            candidates = sweep.all_configs()
            if trial_idx >= len(candidates):
                break
            config = candidates[trial_idx]
            reasoning = "grid enumeration"
        else:
            suggestion = recommender_obj.suggest(
                search_space=search_space,
                history=history,
                constraints=constraints,
            )
            config = suggestion.config
            reasoning = suggestion.reasoning

        result: TrialResult = run_trial(
            train_fn_module_path=train_fn_module,
            train_fn_name=train_fn_name,
            config=config,
            timeout_seconds=timeout_seconds,
        )

        record = result.to_dict()
        record["trial_index"] = trial_idx
        record["recommender_reasoning"] = reasoning
        all_results.append(record)

        if result.throughput is not None:
            history.append((config, result.throughput))
            if best_throughput is None or result.throughput > best_throughput:
                best_throughput = result.throughput
                best_config = config

    final_recommendation: Suggestion | None = None
    if recommender_obj is not None and history:
        try:
            final_recommendation = recommender_obj.suggest(
                search_space=search_space,
                history=history,
                constraints=constraints,
            )
        except Exception:
            final_recommendation = None

    return TuneResult(
        best_config=best_config,
        best_throughput=best_throughput,
        all_results=all_results,
        recommendation=final_recommendation,
    )


def _resolve_recommender(spec, seed: int) -> Recommender | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec in ("claude", "gpt-4o"):
            from autotune.recommenders.llm import LLMRecommender

            return LLMRecommender(provider=spec)
        if spec == "heuristic":
            return HeuristicRecommender()
        if spec == "random":
            return RandomRecommender(seed=seed)
        raise ValueError(
            f"unknown recommender {spec!r}; "
            "use 'claude' | 'gpt-4o' | 'heuristic' | 'random' | None or a Recommender instance"
        )
    return spec  # already a Recommender


def _resolve_train_fn_path(train_fn: Callable) -> tuple[str, str]:
    """
    The local backend imports train_fn in a fresh subprocess, so train_fn
    must be a module-level callable. Lambdas, partials, and nested defs
    won't work — fail fast with a clear message.
    """
    module = inspect.getmodule(train_fn)
    if module is None or module.__name__ == "__main__":
        raise ValueError(
            "train_fn must live in an importable module (not __main__). "
            "Move it into a file and import it: `from my_pkg.workloads import train`."
        )
    name = getattr(train_fn, "__name__", None)
    if name is None or name == "<lambda>":
        raise ValueError("train_fn must be a named module-level function, not a lambda.")
    return module.__name__, name
