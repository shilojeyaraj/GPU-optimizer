"""
Unit tests for experiments/aggregate_results.py. Synthesizes fake result JSON
blobs (matching run_experiment.py's payload schema) and asserts the aggregator
produces the expected summary, trajectory, and H2 pairing.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.aggregate_results import (
    _aggregate,
    _best_so_far,
    _h2_ablation,
    _load_runs,
    _paired_diff_stats,
    _scan_llm_costs,
    _trials_to_threshold,
)


def _trial(idx: int, tput: float | None, oom: bool = False) -> dict:
    return {
        "config": {"batch_size": 32},
        "throughput_samples_per_sec": tput,
        "error": None,
        "oom": oom,
        "compile_failure": False,
        "timeout": False,
        "trial_duration_seconds": 1.0,
        "memory_mb": 0.0,
        "extra": {},
        "trial_index": idx,
        "recommender_reasoning": "test",
    }


def _write_run(
    dir: Path,
    workload: str,
    baseline: str,
    seed: int,
    repeat: int,
    trials: list[dict],
) -> Path:
    payload = {
        "run_id": f"{workload}_{baseline}_seed{seed}_rep{repeat}",
        "args": {
            "workload": workload,
            "baseline": baseline,
            "seed": seed,
            "repeat": repeat,
            "n_trials": len(trials),
        },
        "gpu_info": {},
        "best_config": {"batch_size": 32},
        "best_throughput": max(
            (
                t["throughput_samples_per_sec"]
                for t in trials
                if t["throughput_samples_per_sec"] is not None
            ),
            default=None,
        ),
        "all_results": trials,
        "final_recommendation": None,
    }
    path = dir / f"{payload['run_id']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_best_so_far_is_monotonic_and_carries_failures():
    trials = [
        _trial(1, 100.0),
        _trial(2, None, oom=True),
        _trial(3, 80.0),
        _trial(4, 150.0),
        _trial(5, 120.0),
    ]
    assert _best_so_far(trials) == [100.0, 100.0, 100.0, 150.0, 150.0]


def test_best_so_far_handles_unordered_trial_indices():
    trials = [_trial(3, 50.0), _trial(1, 30.0), _trial(2, 100.0)]
    assert _best_so_far(trials) == [30.0, 100.0, 100.0]


def test_trials_to_threshold_returns_first_hit():
    curve = [10.0, 20.0, 50.0, 90.0]
    assert _trials_to_threshold(curve, target=45.0) == 3
    assert _trials_to_threshold(curve, target=200.0) is None


def test_trials_to_threshold_skips_leading_failures():
    curve = [None, None, 50.0, 100.0]
    assert _trials_to_threshold(curve, target=80.0) == 4


def test_aggregate_oracle_is_max_across_baselines(tmp_path):
    _write_run(tmp_path, "resnet50", "random", 42, 1, [_trial(1, 100.0)])
    _write_run(tmp_path, "resnet50", "llm_with_docs", 42, 1, [_trial(1, 200.0)])
    _write_run(tmp_path, "resnet50", "llm_no_docs", 42, 1, [_trial(1, 150.0)])

    runs = _load_runs(tmp_path)
    summary, _, oracle = _aggregate(runs)
    assert oracle["resnet50"] == 200.0

    random_row = next(r for r in summary if r["baseline"] == "random")
    assert random_row["pct_of_oracle"] == 50.0
    llm_row = next(r for r in summary if r["baseline"] == "llm_with_docs")
    assert llm_row["pct_of_oracle"] == 100.0


def test_aggregate_mean_std_across_repeats(tmp_path):
    for repeat, tput in enumerate([100.0, 120.0, 140.0], start=1):
        _write_run(
            tmp_path, "resnet50", "random", 42, repeat, [_trial(1, tput)]
        )

    runs = _load_runs(tmp_path)
    summary, _, _ = _aggregate(runs)
    row = summary[0]
    assert row["n_runs"] == 3
    assert row["mean_best"] == 120.0
    # population std of [100, 120, 140] using sample stdev (n-1) is 20.0
    assert row["std_best"] == 20.0


def test_aggregate_oom_rate(tmp_path):
    trials = [
        _trial(1, 50.0),
        _trial(2, None, oom=True),
        _trial(3, None, oom=True),
        _trial(4, 100.0),
    ]
    _write_run(tmp_path, "resnet50", "random", 42, 1, trials)
    runs = _load_runs(tmp_path)
    summary, _, _ = _aggregate(runs)
    assert summary[0]["mean_oom_rate"] == 0.5


def test_h2_ablation_pairs_by_workload_seed_repeat(tmp_path):
    # Two paired observations: with_docs > no_docs on both.
    _write_run(tmp_path, "resnet50", "llm_with_docs", 42, 1, [_trial(1, 150.0)])
    _write_run(tmp_path, "resnet50", "llm_no_docs", 42, 1, [_trial(1, 100.0)])
    _write_run(tmp_path, "bert_finetune", "llm_with_docs", 42, 1, [_trial(1, 80.0)])
    _write_run(tmp_path, "bert_finetune", "llm_no_docs", 42, 1, [_trial(1, 60.0)])
    # An unmatched run (no no_docs partner) -- must not be included.
    _write_run(tmp_path, "llama_lora", "llm_with_docs", 42, 1, [_trial(1, 10.0)])

    runs = _load_runs(tmp_path)
    h2 = _h2_ablation(runs)
    g = h2["global"]
    assert g["n"] == 2
    assert g["mean_a"] == 115.0  # mean of with_docs
    assert g["mean_b"] == 80.0   # mean of no_docs
    assert g["mean_diff"] == 35.0


def test_h2_ablation_empty_when_no_llm_runs(tmp_path):
    _write_run(tmp_path, "resnet50", "random", 42, 1, [_trial(1, 100.0)])
    runs = _load_runs(tmp_path)
    h2 = _h2_ablation(runs)
    assert h2["global"] == {"n": 0}


def test_paired_diff_stats_handles_zero_variance():
    stats = _paired_diff_stats([(10.0, 10.0)])
    assert stats["n"] == 1
    assert stats["mean_diff"] == 0.0


def test_scan_llm_costs_sums_response_events(tmp_path):
    log_dir = tmp_path / "autotune_logs"
    log_dir.mkdir()
    session = log_dir / "20260601_120000.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps({"event": "prompt"}),
                json.dumps(
                    {
                        "event": "response",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 500,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "response",
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 100,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = _scan_llm_costs(log_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_responses"] == 2
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 600
    # 1000 * $3/M + 600 * $15/M = $0.003 + $0.009 = $0.012
    assert abs(row["cost_usd"] - 0.012) < 1e-9


def test_scan_llm_costs_returns_empty_when_no_logs(tmp_path):
    assert _scan_llm_costs(tmp_path / "nonexistent") == []
