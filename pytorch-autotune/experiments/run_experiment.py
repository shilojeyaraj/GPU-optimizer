"""
Single entrypoint for paper experiments.

Usage:
    python experiments/run_experiment.py \
        --workload resnet50 \
        --baseline llm_with_docs \
        --n-trials 30 \
        --repeat 1 \
        --seed 42

Writes a JSON result blob to experiments/results/<run_id>.json.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import sys
from pathlib import Path

import yaml

from autotune import grid, tune
from autotune.profile import get_gpu_info

REPO_ROOT = Path(__file__).resolve().parent.parent

BASELINES = {
    "random": {"recommender": "random"},
    "heuristic": {"recommender": "heuristic"},
    "llm_no_docs": {"recommender": "claude", "include_docs": False},
    "llm_with_docs": {"recommender": "claude", "include_docs": True},
    "optuna_tpe": {"recommender": "_optuna_tpe"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        required=True,
        choices=["resnet50", "bert_finetune", "llama_lora"],
    )
    parser.add_argument("--baseline", required=True, choices=list(BASELINES))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    search_space = cfg["search_space"]
    constraints = cfg["constraints"]

    train_fn = importlib.import_module(
        f"experiments.workloads.{args.workload}"
    ).train
    effective_seed = args.seed + args.repeat

    baseline_kwargs = BASELINES[args.baseline]
    if baseline_kwargs["recommender"] == "_optuna_tpe":
        from experiments.tpe_baseline import run_tpe

        result = run_tpe(
            train_fn,
            search_space=search_space,
            n_trials=args.n_trials,
            seed=effective_seed,
        )
    else:
        recommender_arg: str | object = baseline_kwargs["recommender"]
        if "include_docs" in baseline_kwargs:
            from autotune.recommenders.llm import LLMRecommender

            recommender_arg = LLMRecommender(
                provider=baseline_kwargs["recommender"],
                include_docs=baseline_kwargs["include_docs"],
            )

        result = tune(
            train_fn,
            sweep=grid(**search_space),
            recommender=recommender_arg,
            n_trials=args.n_trials,
            constraints=constraints,
            seed=effective_seed,
        )

    run_id = (
        f"{args.workload}_{args.baseline}_seed{args.seed}_rep{args.repeat}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_path = REPO_ROOT / "experiments" / "results" / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "args": vars(args),
        "gpu_info": get_gpu_info(),
        "best_config": result.best_config,
        "best_throughput": result.best_throughput,
        "all_results": result.all_results,
        "final_recommendation": (
            {"config": result.recommendation.config, "reasoning": result.recommendation.reasoning}
            if result.recommendation
            else None
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"results saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
