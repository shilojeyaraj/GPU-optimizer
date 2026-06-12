"""
End-to-end smoke test with a toy training function that doesn't need a GPU.
Runs the full tune() loop including subprocess isolation.
"""
import time

from autotune import grid, tune


def toy_train(config: dict) -> float:
    """Fake throughput: bigger batch = higher throughput, bf16 = 2x."""
    time.sleep(0.05)
    base = config["batch_size"]
    multiplier = 2.0 if config.get("precision") == "bf16" else 1.0
    return base * multiplier


def test_tune_smoke_with_random_recommender():
    result = tune(
        toy_train,
        sweep=grid(batch_size=[16, 32], precision=["fp32", "bf16"]),
        recommender="random",
        n_trials=4,
        seed=0,
    )
    assert result.best_config is not None
    assert result.best_throughput is not None
    assert result.best_throughput > 0
    assert len(result.all_results) == 4


def test_tune_smoke_with_heuristic_recommender():
    result = tune(
        toy_train,
        sweep=grid(batch_size=[16, 32], precision=["fp32", "bf16"]),
        recommender="heuristic",
        n_trials=3,
        seed=0,
    )
    assert result.best_config is not None
    assert len(result.all_results) == 3


def test_tune_smoke_with_no_recommender_walks_grid():
    result = tune(
        toy_train,
        sweep=grid(batch_size=[16, 32], precision=["fp32"]),
        recommender=None,
        n_trials=10,
    )
    # n_trials > grid size: tune() should stop after the grid is exhausted.
    assert len(result.all_results) == 2
