# pytorch-autotune — build progress

Running status of the build. The **what/why** lives in
[`pytorch-autotune-plan.md`](pytorch-autotune-plan.md) (canonical plan: scope,
budget, phases, decisions). The **user-facing library docs** live in
[`pytorch-autotune/README.md`](pytorch-autotune/README.md). This file is the
**where-are-we**: what's implemented, what's verified, what's next.

---

## Snapshot — 2026-06-11

**Phase: library skeleton + experiment harness + analysis pipeline complete; live LLM path verified end-to-end script-wise; nothing run on GPU yet.**

The library, all three workloads, and all five baselines are implemented and
import-clean. No experiment has been *executed* — that needs a GPU (see
"Verification caveat" below). The frontier is acquiring compute and running the
matrix.

### Done & verified (CPU)

| Area | State | Verification |
|---|---|---|
| Core lib (`sweep`, `tune`, `profile`, `_trial_runner`, `backends/local`) | committed | 24 unit tests pass |
| Recommenders (`random`, `heuristic`, `llm`, `base` Protocol) | committed | unit tests + lint |
| Optuna integration (`LLMSampler`) | committed | unit tests |
| Workload: `resnet50.py` | real | import-clean; CUDA path unrun |
| Workload: `bert_finetune.py` | **new** — real | import-clean; CUDA path unrun |
| Workload: `llama_lora.py` | **new** — real | import-clean; CUDA path unrun |
| Baseline: `optuna_tpe` (`tpe_baseline.py`) | **new** — real | 3 CPU tests (best-tracking, schema, pruning) |
| `run_experiment.py` `--baseline optuna_tpe` | **new** — wired | compile-check; full run needs GPU |
| `pyproject` `[paper]` extra (transformers, peft, torchvision) | **new** | — |
| `LLMRecommender` usage-token capture (input/output/cache) | **new** | 8 unit tests pass; cost-per-trial now recoverable |
| `experiments/smoke_llm.py` -- live Claude smoke CLI | **new** | ruff clean; `--help` + missing-key error path verified; live call deferred to user |
| `experiments/aggregate_results.py` -- analysis aggregator | **new** | 12 CPU tests (best-so-far, trials-to-90, oracle pct, H2 pairing, OOM rate, cost recovery) |

Full local check at snapshot: **36 unit tests pass, ruff clean.**

### Workload design notes (so the paper write-up is consistent)

- All workloads are **synthetic-data** (random tensors on-device) — we measure
  compute throughput, not dataset I/O. Pattern: warmup → measure window with
  `torch.cuda.synchronize()` bookends; return samples/sec.
- `use_flash_attention` is a **real toggle** on BERT/LLaMA: it pins the SDPA
  backend (FLASH+efficient vs MATH-only) via `torch.nn.attention.sdpa_kernel`
  (legacy `sdp_kernel` fallback). It's a no-op for ResNet (no attention). FLASH
  is fp16/bf16-only, so fp32 falls back to math automatically.
- `gradient_accumulation_steps` is honored (micro-batch loop, scaled loss, one
  optimizer step). Throughput counts `batch_size × accum_steps` per step.
- `dataloader_workers` is **inert** across all workloads — synthetic on-device
  batches mean there's no host data pipeline for worker count to affect. If the
  paper needs worker-count sensitivity, a synthetic `DataLoader` would have to be
  added. **Open decision.**
- LLaMA-7B base is built **directly in the trial dtype** (default-dtype context)
  to avoid a ~28 GB fp32 host copy; bf16 lands ~14 GB, matching
  `autotune.profile` VRAM estimates.

### Pending — the actual frontier

1. **Compute.** Apply: Lambda academic grant, TPU Research Cloud, Kaggle,
   Vast.ai (plan §4.4). Hard ceiling **$100**.
2. **Live-LLM smoke** — script is wired (`experiments/smoke_llm.py`); user runs
   it once with their `ANTHROPIC_API_KEY` (~$0.01-$0.05) to confirm both ablation
   arms return LLM-grounded configs and that token usage logs land in
   `experiments/results/smoke_logs/`.
3. **GPU smoke** — run each workload once at a known-good config to confirm the
   `train()` bodies execute and produce sane throughput (esp. LLaMA memory).
4. **Run the matrix** — 3 workloads × 5 baselines × 3 repeats = 45 runs.
5. **Analysis + paper** — aggregator is wired (`experiments/aggregate_results.py`:
   produces `summary.csv`, `trajectories.csv`, `h2_ablation.json`, `llm_cost.csv`
   under `experiments/results/analysis/`). Remaining: paper notebook with plots
   + proper statistical tests (paired Wilcoxon for H2 — aggregator reports
   effect size only) using `experiments/preregistration.md` as the hypothesis
   ledger.

### Verification caveat

This dev box is **CPU-only torch (2.8.0+cpu), no transformers/peft installed,
no GPU**. So:
- Everything CPU-checkable is checked (imports, lint, unit tests, the TPE study
  loop via injected `run_trial`).
- The `train()` bodies in all three workloads and the subprocess-execution path
  for real workloads are **correct-by-construction but unexecuted**. First GPU
  run is where they get truly validated — expect to iterate on dtype/attn-impl
  edge cases against the installed transformers version.

---

## Build log

- **2026-06-11** — Added `experiments/aggregate_results.py`: reads result JSONs
  + LLM session logs and emits the paper's headline tables (summary.csv,
  trajectories.csv, h2_ablation.json, llm_cost.csv). Implements the
  pre-registered secondary metrics (trials-to-90% against per-workload oracle,
  OOM rate, $-cost-to-best) and a paired with_docs/no_docs effect-size
  computation. 36 unit tests pass, ruff clean. Also drafted the Lambda
  academic-credits application (kept in chat, not committed).
- **2026-05-29** — Added live LLM smoke (`experiments/smoke_llm.py`): exercises
  both ablation arms against the real Claude API, computes cost from token
  usage, exits non-zero on fallback. To support it, added input/output/cache
  token capture to `LLMRecommender._call_provider` (logged into the existing
  JSONL session log under the `response` event). 24 unit tests + 8 LLM unit
  tests pass, ruff clean.
- **2026-05-28** — Implemented `bert_finetune` + `llama_lora` workloads; added
  `[paper]` extra. Wrote `tpe_baseline.run_tpe` + 3 CPU tests; wired
  `--baseline optuna_tpe` into `run_experiment.py`. Updated experiments README.
  24 unit tests pass, ruff clean.
- **2026-05-25** — Carve-out committed (`5705616`): `pytorch-autotune/` library
  built inside the repo; dashboard / C++ / infra deleted; root rewired as a
  monorepo (README, Makefile, CI, release→PyPI, Apache 2.0). Budget cut to a
  $100 ceiling in the plan.
