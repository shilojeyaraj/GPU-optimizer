"""
Aggregate experiments/results/*.json into paper-ready tables.

Pre-registered metrics (experiments/preregistration.md):
  - Primary:   best throughput at trial budget N
  - Secondary: trials-to-90%-of-best, OOM rate, $-cost-to-best

Outputs land in experiments/results/analysis/:
  - summary.csv          one row per (workload, baseline): mean +/- std best,
                         mean trials-to-90, OOM rate
  - trajectories.csv     long-form best-so-far curves; one row per trial
  - h2_ablation.json     paired llm_with_docs vs llm_no_docs comparison
  - llm_cost.csv         recovered token cost from LLM JSONL session logs

The headline summary is also printed to stdout.

Run after experiments/run_experiment.py has produced its result JSONs:

    python experiments/aggregate_results.py
    python experiments/aggregate_results.py --results-dir path/to/results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results"
DEFAULT_LLM_LOG_DIR = REPO_ROOT / "autotune_logs"

# Per-million-token pricing for claude-sonnet-4-6 as of 2026-05. Kept in sync
# with experiments/smoke_llm.py; the JSONL log keeps raw counts so cost can be
# recomputed at any later date.
PRICE_INPUT_PER_M = 3.00
PRICE_OUTPUT_PER_M = 15.00
PRICE_CACHE_WRITE_PER_M = 3.75
PRICE_CACHE_READ_PER_M = 0.30

# Threshold for the secondary "trials-to-X%" metric. Prereg uses 90%.
TRIALS_TO_THRESHOLD = 0.90


def _load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"warn: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        args = payload.get("args", {})
        runs.append(
            {
                "path": path,
                "run_id": payload.get("run_id"),
                "workload": args.get("workload"),
                "baseline": args.get("baseline"),
                "seed": args.get("seed"),
                "repeat": args.get("repeat"),
                "n_trials": args.get("n_trials"),
                "best_throughput": payload.get("best_throughput"),
                "all_results": payload.get("all_results", []),
            }
        )
    return runs


def _best_so_far(trials: list[dict]) -> list[float | None]:
    """Monotonic best-throughput-seen-so-far across trial_index order.

    Failed trials (None throughput) carry the prior best forward.
    """
    sorted_trials = sorted(
        trials, key=lambda t: t.get("trial_index", 0)
    )
    curve: list[float | None] = []
    best: float | None = None
    for t in sorted_trials:
        tput = t.get("throughput_samples_per_sec")
        if tput is not None and (best is None or tput > best):
            best = tput
        curve.append(best)
    return curve


def _trials_to_threshold(curve: list[float | None], target: float) -> int | None:
    """First trial_index (1-based) at which best-so-far >= target. None if never."""
    for i, value in enumerate(curve, start=1):
        if value is not None and value >= target:
            return i
    return None


def _oom_count(trials: list[dict]) -> int:
    return sum(1 for t in trials if t.get("oom"))


def _safe_mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.fmean(vals)


def _safe_std(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    return statistics.stdev(vals)


def _paired_diff_stats(pairs: list[tuple[float, float]]) -> dict:
    """Paired (a, b) -> mean diff, std diff, Cohen's d_z, n.

    Effect size only; no p-value (avoids a scipy dep). Run the proper paired
    t / Wilcoxon test in the paper notebook once these are wired up.
    """
    diffs = [a - b for a, b in pairs]
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean_d = statistics.fmean(diffs)
    std_d = statistics.stdev(diffs) if n >= 2 else 0.0
    cohens_dz = mean_d / std_d if std_d > 0 else math.inf if mean_d != 0 else 0.0
    return {
        "n": n,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "cohens_dz": cohens_dz,
        "mean_a": statistics.fmean(a for a, _ in pairs),
        "mean_b": statistics.fmean(b for _, b in pairs),
    }


def _aggregate(runs: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """
    Returns (summary_rows, trajectory_rows, oracle_by_workload).

    oracle_by_workload[workload] = max best_throughput seen across ALL baselines
    and repeats. Used so trials-to-X% is comparable across baselines on the
    same workload.
    """
    oracle_by_workload: dict[str, float] = {}
    for run in runs:
        wl = run["workload"]
        if run["best_throughput"] is None:
            continue
        oracle_by_workload[wl] = max(
            oracle_by_workload.get(wl, 0.0), run["best_throughput"]
        )

    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        by_cell[(run["workload"], run["baseline"])].append(run)

    trajectory_rows: list[dict] = []
    summary_rows: list[dict] = []

    for (workload, baseline), cell_runs in sorted(by_cell.items()):
        oracle = oracle_by_workload.get(workload)
        threshold = (
            oracle * TRIALS_TO_THRESHOLD if oracle is not None else None
        )

        bests: list[float] = []
        trials_to_threshold: list[int] = []
        oom_rates: list[float] = []

        for run in cell_runs:
            curve = _best_so_far(run["all_results"])
            for i, value in enumerate(curve, start=1):
                trajectory_rows.append(
                    {
                        "workload": workload,
                        "baseline": baseline,
                        "seed": run["seed"],
                        "repeat": run["repeat"],
                        "trial_index": i,
                        "best_so_far": value,
                    }
                )

            if run["best_throughput"] is not None:
                bests.append(run["best_throughput"])
            if threshold is not None:
                hit = _trials_to_threshold(curve, threshold)
                if hit is not None:
                    trials_to_threshold.append(hit)
            n_trials = run["n_trials"] or len(run["all_results"]) or 0
            if n_trials > 0:
                oom_rates.append(_oom_count(run["all_results"]) / n_trials)

        summary_rows.append(
            {
                "workload": workload,
                "baseline": baseline,
                "n_runs": len(cell_runs),
                "oracle_throughput": oracle,
                "mean_best": _safe_mean(bests),
                "std_best": _safe_std(bests),
                "pct_of_oracle": (
                    (_safe_mean(bests) / oracle * 100)
                    if oracle and _safe_mean(bests) is not None
                    else None
                ),
                "mean_trials_to_90": _safe_mean(trials_to_threshold),
                "n_reached_90": len(trials_to_threshold),
                "mean_oom_rate": _safe_mean(oom_rates),
            }
        )

    return summary_rows, trajectory_rows, oracle_by_workload


def _h2_ablation(runs: list[dict]) -> dict:
    """Paired llm_with_docs vs llm_no_docs on matched (workload, seed, repeat)."""
    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    for run in runs:
        if run["baseline"] not in {"llm_with_docs", "llm_no_docs"}:
            continue
        if run["best_throughput"] is None:
            continue
        key = (run["workload"], run["seed"], run["repeat"])
        by_key[key][run["baseline"]] = run["best_throughput"]

    pairs: list[tuple[float, float]] = []
    per_workload_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (workload, _, _), arms in by_key.items():
        if "llm_with_docs" in arms and "llm_no_docs" in arms:
            p = (arms["llm_with_docs"], arms["llm_no_docs"])
            pairs.append(p)
            per_workload_pairs[workload].append(p)

    return {
        "threshold_for_test": TRIALS_TO_THRESHOLD,
        "global": _paired_diff_stats(pairs),
        "per_workload": {
            wl: _paired_diff_stats(ps) for wl, ps in per_workload_pairs.items()
        },
    }


def _scan_llm_costs(log_dir: Path) -> list[dict]:
    """Walk autotune_logs/ (recursive) for JSONL session logs.

    Returns one row per session file with summed token usage + cost. Sessions
    can't be tied back to specific (workload, baseline) runs without a
    log_dir-per-run change; this is a global cost recovery for the paper's
    "$-cost-to-best" metric. See PROGRESS.md open item.
    """
    rows: list[dict] = []
    if not log_dir.exists():
        return rows
    for path in sorted(log_dir.rglob("*.jsonl")):
        input_t = output_t = cache_w = cache_r = 0
        n_responses = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "response":
                continue
            usage = event.get("usage") or {}
            input_t += usage.get("input_tokens") or 0
            output_t += usage.get("output_tokens") or 0
            cache_w += usage.get("cache_creation_input_tokens") or 0
            cache_r += usage.get("cache_read_input_tokens") or 0
            n_responses += 1
        cost = (
            input_t * PRICE_INPUT_PER_M
            + output_t * PRICE_OUTPUT_PER_M
            + cache_w * PRICE_CACHE_WRITE_PER_M
            + cache_r * PRICE_CACHE_READ_PER_M
        ) / 1_000_000
        rows.append(
            {
                "session_log": str(path.relative_to(log_dir.parent)),
                "n_responses": n_responses,
                "input_tokens": input_t,
                "output_tokens": output_t,
                "cache_creation_tokens": cache_w,
                "cache_read_tokens": cache_r,
                "cost_usd": cost,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(summary_rows: list[dict], h2: dict) -> None:
    if not summary_rows:
        print("no result JSONs found.")
        return

    print()
    print("=== Summary by (workload, baseline) ===")
    header = (
        f"{'workload':<16}{'baseline':<18}{'n':<4}"
        f"{'mean_best':>12}{'std':>10}{'%oracle':>9}"
        f"{'trials/90':>11}{'reached':>9}{'oom%':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        mean_b = f"{row['mean_best']:.2f}" if row["mean_best"] is not None else "n/a"
        std_b = f"{row['std_best']:.2f}" if row["std_best"] is not None else "n/a"
        pct = (
            f"{row['pct_of_oracle']:.1f}"
            if row["pct_of_oracle"] is not None
            else "n/a"
        )
        ttt = (
            f"{row['mean_trials_to_90']:.1f}"
            if row["mean_trials_to_90"] is not None
            else "n/a"
        )
        oom = (
            f"{row['mean_oom_rate'] * 100:.1f}"
            if row["mean_oom_rate"] is not None
            else "n/a"
        )
        print(
            f"{row['workload']:<16}{row['baseline']:<18}{row['n_runs']:<4}"
            f"{mean_b:>12}{std_b:>10}{pct:>9}"
            f"{ttt:>11}{row['n_reached_90']:>9}{oom:>7}"
        )

    g = h2.get("global", {})
    if g.get("n", 0) > 0:
        print()
        print("=== H2: llm_with_docs vs llm_no_docs (paired) ===")
        print(f"n pairs:        {g['n']}")
        print(f"mean with_docs: {g['mean_a']:.2f}")
        print(f"mean no_docs:   {g['mean_b']:.2f}")
        print(f"mean diff:      {g['mean_diff']:+.2f}")
        print(f"std diff:       {g['std_diff']:.2f}")
        print(f"Cohen's d_z:    {g['cohens_dz']:+.2f}")
        print("(no p-value here; run the proper paired test in the paper notebook)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory of *.json result blobs. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--llm-log-dir",
        default=str(DEFAULT_LLM_LOG_DIR),
        help=(
            "Directory of LLM JSONL session logs to recover token cost from. "
            f"Default: {DEFAULT_LLM_LOG_DIR}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write summary.csv etc. Default: <results-dir>/analysis",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "analysis"

    runs = _load_runs(results_dir)
    if not runs:
        print(f"no result JSONs in {results_dir}", file=sys.stderr)
        return 1

    summary_rows, trajectory_rows, oracle = _aggregate(runs)
    h2 = _h2_ablation(runs)
    llm_costs = _scan_llm_costs(Path(args.llm_log_dir))

    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "trajectories.csv", trajectory_rows)
    if llm_costs:
        _write_csv(out_dir / "llm_cost.csv", llm_costs)

    h2_path = out_dir / "h2_ablation.json"
    h2_path.parent.mkdir(parents=True, exist_ok=True)
    h2_path.write_text(json.dumps(h2, indent=2), encoding="utf-8")

    _print_summary(summary_rows, h2)
    print()
    print(f"wrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
