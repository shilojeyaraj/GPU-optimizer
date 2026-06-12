# pytorch-autotune

LLM-grounded autotuning of PyTorch **systems** knobs (batch size, precision,
`torch.compile` mode, attention backend, gradient accumulation, dataloader
workers), packaged as an OSS library and benchmarked against random / heuristic
/ Optuna TPE baselines for a NeurIPS workshop paper.

This repo is a monorepo: the library lives at [`pytorch-autotune/`](pytorch-autotune/);
the paper experiments live at [`pytorch-autotune/experiments/`](pytorch-autotune/experiments/).
Status is tracked in [`PROGRESS.md`](PROGRESS.md); the full execution plan is
[`pytorch-autotune-plan.md`](pytorch-autotune-plan.md).

## What this is

A library you point at your `train_fn` that returns samples/sec. Instead of
random search or TPE, it asks an LLM (Claude or GPT-4o) for the next config to
try, given the search space, the trial history, and a hardware constraint:

```python
from autotune import tune, grid

def train(config):
    # your training step; return samples/sec
    ...

result = tune(
    train,
    sweep=grid(
        batch_size=[16, 32, 64, 128],
        precision=["fp32", "fp16", "bf16"],
        compile_mode=["default", "reduce-overhead", "max-autotune"],
        use_flash_attention=[True, False],
        gradient_accumulation_steps=[1, 2, 4],
        dataloader_workers=[0, 2, 4, 8],
    ),
    recommender="claude",
    n_trials=30,
    constraints={"max_vram_gb": 24},
)
```

Existing HPO tools (Optuna, Ray Tune, W&B Sweeps) focus on **quality** knobs
(learning rate, dropout). They treat systems knobs as out-of-scope; practitioners
default to folklore or stop at the first config that doesn't OOM. The hypothesis
is that an LLM with PyTorch systems knowledge can converge to a near-optimal
config in substantially fewer trials than uninformed search.

## What this is *not*

- Not a profiler. Not a GPU monitoring dashboard. Not a training framework.
- Not a quality-knob HPO replacement — orthogonal to Optuna's usual job.
- Not yet measured on real GPUs. See [Current state](#current-state).

(An earlier version of this repo was a C++/Postgres GPU optimizer with a
dashboard. That was carved out at commit `5705616`. None of it is in this codebase.)

## The research question

The paper's central question is a single ablation in the LLM recommender's
system prompt:

- **`include_docs=True`** -- the prompt contains a block of PyTorch tuning
  knowledge (bf16 vs fp16 stability, when `torch.compile` modes pay off, flash
  attention dtype constraints, dataloader worker heuristics).
- **`include_docs=False`** -- the same prompt with that block removed. The LLM
  reasons from first principles.

If `with_docs` meaningfully beats `no_docs`, encoded systems knowledge in the
prompt transfers to better suggestions. If they tie, general LLM reasoning
suffices and the docs are decorative. Both outcomes are publishable; the
pre-registered framings are in [`experiments/preregistration.md`](pytorch-autotune/experiments/preregistration.md).

**Matrix:** 3 workloads (ResNet-50, BERT-base fine-tune, LLaMA-7B + LoRA) x
5 baselines (random, heuristic, Optuna TPE, LLM no-docs, LLM with-docs) x
3 seeds = 45 runs at 30 trials each. Primary metric: best throughput at trial
budget N=30. Secondary: trials-to-90%-of-oracle, OOM rate, $-cost-to-best.

## Current state

| Layer | State | Verified on |
|---|---|---|
| Core library (`autotune/`) | implemented | CPU: 24 unit tests |
| All 5 recommenders | implemented | CPU: unit tests + lint |
| All 3 workloads (synthetic-data, on-device) | implemented | CPU: import + smoke; **GPU path unrun** |
| All 5 baselines runnable through `run_experiment.py` | wired | CPU: compile check |
| Live-LLM smoke (`experiments/smoke_llm.py`) | wired | `--help` + missing-key path; **live call deferred** |
| Aggregation pipeline (`experiments/aggregate_results.py`) | implemented | CPU: 12 unit tests |
| **Local total** | | **36 unit tests pass, ruff clean** |

What's **not** verified yet (gated on GPU access):
- The `train()` bodies actually executing on real CUDA hardware
- The live Claude API call returning LLM-grounded configs in both ablation arms
- Any throughput number, any regret curve, any claim about which method wins

The dev box is CPU-only torch 2.8.0+cpu with no transformers/peft installed.
Everything CPU-checkable is checked; the rest is correct-by-construction and
will be validated when GPU time lands.

## What's pending

In order, from cheapest to most expensive:

1. **Live-LLM smoke** ($0.01-$0.05 on your laptop). Validates the LLM path
   end-to-end before spending any GPU budget. See
   [`experiments/README.md`](pytorch-autotune/experiments/README.md#live-llm-smoke-run-before-spending-gpu-budget).
2. **Compute access.** Targeting Lambda academic credits + Vast.ai fallback.
   Hard ceiling **$100** for the entire project.
3. **GPU smoke** -- one known-good config per workload to confirm `train()` runs.
4. **The 45-run matrix.** ~10 GPU-hours on A100; ~$5-10 on Vast.ai.
5. **Analysis notebook + paper draft.** Aggregator outputs are paper-ready;
   what's missing is the notebook with plots + the proper paired Wilcoxon test
   (the aggregator reports effect size only, no p-value).

## Layout

| Path | Contents |
|---|---|
| [`pytorch-autotune/`](pytorch-autotune/) | The library: `autotune/`, `tests/`, `examples/`, `pyproject.toml`, user-facing README |
| [`pytorch-autotune/experiments/`](pytorch-autotune/experiments/) | Paper scripts: `run_experiment.py`, `tpe_baseline.py`, `smoke_llm.py`, `aggregate_results.py`, `preregistration.md`, `config.yaml`, workloads, results dir |
| [`pytorch-autotune-plan.md`](pytorch-autotune-plan.md) | Full execution plan: budget, scope, phases, decisions log |
| [`PROGRESS.md`](PROGRESS.md) | Snapshot + build log (canonical "where are we") |

## Quickstart

```powershell
cd pytorch-autotune
pip install -e ".[llm,paper]"
pytest tests/unit -q          # 36 tests, ~8 seconds
ruff check .
```

For the user-facing library docs (install, API, examples) see
[`pytorch-autotune/README.md`](pytorch-autotune/README.md).

## License

Apache 2.0 -- see [LICENSE](LICENSE).
