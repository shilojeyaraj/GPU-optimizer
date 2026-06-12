# Experiments

Paper experiment scripts. Reproducibility artifact.

## Reproducing paper results

```bash
cd pytorch-autotune
pip install -e ".[llm,paper]"

# One (workload × baseline × repeat) at a time.
python experiments/run_experiment.py \
    --workload resnet50 \
    --baseline llm_with_docs \
    --n-trials 30 \
    --repeat 1

# Full matrix is 3 workloads × 5 baselines × 3 repeats = 45 runs.
```

## Workloads

All three are implemented and synthetic-data (measure compute, not I/O):

- `resnet50.py` — torchvision ResNet-50, images/sec
- `bert_finetune.py` — BERT-base sequence classification, sequences/sec
- `llama_lora.py` — LLaMA-7B + PEFT LoRA (frozen base), sequences/sec

`use_flash_attention` toggles the SDPA backend (FLASH vs MATH) on the two
transformer workloads; it is a no-op for ResNet (no attention).
`dataloader_workers` is inert across all three (batches are synthetic and
on-device — there is no host data pipeline).

## Baselines

All 5 are runnable through `run_experiment.py --baseline <name>`:

- `random` — uniform random over the search space
- `heuristic` — static GPU/model rules (NVIDIA/PyTorch tuning guides)
- `optuna_tpe` — Optuna TPESampler study (`experiments/tpe_baseline.py`); drives its
  own suggest/evaluate loop but reuses the same isolated backend and result schema
- `llm_no_docs` — LLM recommender, PyTorch knowledge section stripped from the prompt
- `llm_with_docs` — LLM recommender with the docs section (the headline ablation)

Failed trials (OOM / compile failure / timeout) count as failed trials for every
baseline — no retries (`max_oom_retries: 0`). TPE records them but treats them as
pruned so they don't inform its surrogate, matching how `tune()` withholds failed
trials from the recommender history.

## Live LLM smoke (run BEFORE spending GPU budget)

One Claude API call per ablation arm (~$0.01-$0.05 on `claude-sonnet-4-6`).
Validates the end-to-end LLM path -- API key, SDK, prompt construction, JSON
parse, search-space validation, token-usage logging -- and exits non-zero if
either arm fell back to random.

```bash
cd pytorch-autotune
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...           # PowerShell: $env:ANTHROPIC_API_KEY = "..."
python experiments/smoke_llm.py
```

JSONL session logs land in `experiments/results/smoke_logs/{with_docs,no_docs}/`.
Token counts + cost are recoverable from the `response` events for the paper
cost-per-trial metric.

## Aggregating results

After `run_experiment.py` has produced result JSONs:

```bash
python experiments/aggregate_results.py
```

Writes to `experiments/results/analysis/`:

- `summary.csv` -- one row per (workload, baseline): mean +/- std best
  throughput, percent of per-workload oracle, mean trials-to-90%, OOM rate
- `trajectories.csv` -- long-form best-so-far curves (one row per trial) for
  plotting regret curves
- `h2_ablation.json` -- paired `llm_with_docs` vs `llm_no_docs` comparison
  (mean diff, std diff, Cohen's d_z). No p-value at this layer; run the proper
  paired test in the paper notebook.
- `llm_cost.csv` -- token cost recovered from JSONL session logs per the
  cost-per-trial paper metric

## Hardware setup checklist

Before any benchmarking session, do this on the target machine:

- [ ] Dedicated GPU (no other processes — check `nvidia-smi`)
- [ ] `sudo nvidia-smi -pm 1` (persistence mode)
- [ ] `sudo nvidia-smi --lock-gpu-clocks=<min>,<max>` if locking
- [ ] Save `nvidia-smi -q` output next to results
- [ ] 3 warmup trials before measurement
