"""
Optuna TPE baseline for the paper experiments.

TPE owns its own suggest/evaluate loop, so it can't ride through autotune.tune()
(which drives a Recommender). This module runs a TPESampler study over the same
6-dim categorical search space, executes each trial through the same isolated
local backend (autotune.backends.local.run_trial), and returns a TuneResult with
the identical schema the other baselines produce — so downstream analysis treats
every baseline's output the same way.

Failed trials (OOM / compile failure / timeout → throughput is None) are recorded
in all_results but raised as optuna.TrialPruned, so they consume a trial-budget
slot without informing the TPE surrogate. This mirrors tune(), which excludes
failed trials from the history it shows a Recommender — keeping the comparison
fair (TPESampler ignores pruned trials when modelling by default).
"""
from __future__ import annotations

from typing import Callable

from autotune.backends.local import TrialResult, run_trial
from autotune.tune import TuneResult, _resolve_train_fn_path


def run_tpe(
    train_fn: Callable[[dict], float],
    search_space: dict[str, list],
    n_trials: int = 30,
    seed: int = 42,
    timeout_seconds: int = 3600,
    run_trial_fn: Callable[..., TrialResult] | None = None,
) -> TuneResult:
    """Run a TPE study and return a TuneResult matching the other baselines.

    run_trial_fn is injectable for testing; defaults to the real subprocess
    backend.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    execute = run_trial_fn or run_trial

    module_path, fn_name = _resolve_train_fn_path(train_fn)
    all_results: list[dict] = []
    best_config: dict | None = None
    best_throughput: float | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_config, best_throughput
        config = {
            name: trial.suggest_categorical(name, list(choices))
            for name, choices in search_space.items()
        }
        result = execute(
            train_fn_module_path=module_path,
            train_fn_name=fn_name,
            config=config,
            timeout_seconds=timeout_seconds,
        )
        record = result.to_dict()
        record["trial_index"] = trial.number
        record["recommender_reasoning"] = "optuna_tpe"
        all_results.append(record)

        if result.throughput is None:
            raise optuna.TrialPruned()
        if best_throughput is None or result.throughput > best_throughput:
            best_throughput = result.throughput
            best_config = config
        return result.throughput

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    return TuneResult(
        best_config=best_config,
        best_throughput=best_throughput,
        all_results=all_results,
        recommendation=None,
    )
