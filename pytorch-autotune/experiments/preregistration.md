# Pre-Registration: LLM Recommenders for Systems-Level Training Knobs

Dated 2026-05-25. Committed before any experiment runs.

## Hypothesis

An LLM (Claude Sonnet 4.6) with PyTorch documentation included in the system prompt will recommend higher-throughput training configurations than the same LLM without documentation, expert-rules heuristics, and random search — but worse than exhaustive grid search — across ResNet-50, BERT-base fine-tune, and LLaMA-7B LoRA workloads.

## Predictions, ranked by confidence

1. **High**: LLM-with-docs beats random search at trial budget N=30. Expected throughput uplift: 10–30% over random.
2. **Medium**: LLM-with-docs beats LLM-without-docs. This is the central ablation; if it doesn't hold, the LLM's contribution is general reasoning rather than encoded systems knowledge.
3. **Medium**: Expert-rules heuristic beats random search but loses to LLM-with-docs once the LLM sees ≥ 5 trial measurements.
4. **Low**: Optuna TPE matches or slightly exceeds LLM-with-docs on workloads it has seen at least 10 trials of.

## Baselines

- Random search (`autotune.recommenders.random.RandomRecommender`)
- Expert heuristic (`autotune.recommenders.heuristic.HeuristicRecommender`)
- Optuna TPE (`optuna.samplers.TPESampler`, default config)
- LLM without docs (`LLMRecommender(provider="claude", include_docs=False)`)
- LLM with docs (`LLMRecommender(provider="claude", include_docs=True)`)

## Metrics

- **Primary**: best throughput (samples/sec) reached at trial budget N=30
- **Secondary**: time-to-90%-of-best, OOM rate, $-cost-to-best, quality preservation (final loss ratio vs fp32 baseline)

## Workloads

- ResNet-50, batch 8–128, on RTX 4090 (24 GB)
- BERT-base fine-tune (HuggingFace), batch 8–64, on RTX 4090
- LLaMA-7B LoRA (HuggingFace + PEFT), batch 1–8, on RTX 4090

## Search space (6 dimensions, locked)

batch_size, precision, compile_mode, use_flash_attention, gradient_accumulation_steps, dataloader_workers. See `config.yaml` for value lists.

## Hardware

- Primary: RTX 4090 (Vast.ai)
- Secondary (validation only, via Habitat prediction): A100-80GB

## Statistical reporting

- 3 repeat runs per (workload, baseline) cell
- Report mean ± std
- Investigate any cell with > 10% inter-repeat variance before publishing

## Falsification conditions

- If LLM-with-docs does not beat random at N=30 on ≥ 2 of 3 workloads: paper reframes as "current LLMs are not yet reliable systems advisors; here is the failure taxonomy."
- If LLM-with-docs ties LLM-without-docs (within 1 std): paper reframes as "general LLM reasoning suffices; encoded PyTorch knowledge contributes little."

These are the publishable framings. The thesis cannot fail to produce a result.
