"""CPU tests for the Optuna TPE baseline runner. run_trial is injected, so no
subprocess and no GPU are involved — we exercise the study loop, best-tracking,
result schema, and failed-trial pruning."""
from autotune.backends.local import TrialResult
from experiments.tpe_baseline import run_tpe

SMALL_SPACE = {
    "batch_size": [8, 16],
    "precision": ["fp32", "bf16"],
}


def _dummy_train(config: dict) -> float:
    # Never actually called — run_trial is mocked. Exists only so
    # _resolve_train_fn_path can resolve an importable module-level function.
    return 1.0


def _fake_run_trial(throughputs):
    """Return canned TrialResults in call order; None throughput = OOM failure."""
    seq = iter(throughputs)

    def run(*, train_fn_module_path, train_fn_name, config, timeout_seconds):
        tp = next(seq)
        return TrialResult(config=config, throughput=tp, oom=tp is None)

    return run


def test_tracks_best_across_trials():
    result = run_tpe(
        _dummy_train,
        search_space=SMALL_SPACE,
        n_trials=4,
        seed=0,
        run_trial_fn=_fake_run_trial([10.0, 50.0, None, 30.0]),
    )
    assert result.best_throughput == 50.0
    assert result.best_config is not None
    assert len(result.all_results) == 4
    assert result.recommendation is None


def test_failed_trial_recorded_with_schema():
    result = run_tpe(
        _dummy_train,
        search_space=SMALL_SPACE,
        n_trials=2,
        seed=0,
        run_trial_fn=_fake_run_trial([None, 20.0]),
    )
    failed = [r for r in result.all_results if r["throughput_samples_per_sec"] is None]
    assert len(failed) == 1
    assert failed[0]["oom"] is True
    assert failed[0]["recommender_reasoning"] == "optuna_tpe"
    assert "trial_index" in failed[0]
    assert result.best_throughput == 20.0  # best ignores the failed trial


def test_all_failed_yields_no_best():
    result = run_tpe(
        _dummy_train,
        search_space=SMALL_SPACE,
        n_trials=3,
        seed=0,
        run_trial_fn=_fake_run_trial([None, None, None]),
    )
    assert result.best_config is None
    assert result.best_throughput is None
    assert len(result.all_results) == 3
