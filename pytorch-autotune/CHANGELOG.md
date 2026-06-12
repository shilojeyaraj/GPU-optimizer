# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package skeleton: `tune()` entrypoint, `SweepGrid`, `Recommender` protocol
- Local subprocess backend with per-trial isolation and OOM detection
- LLM recommender (Claude + GPT-4o) with `include_docs` ablation flag
- Heuristic and random recommenders as baselines
- Optuna `LLMSampler` integration
- Three reference workloads: ResNet-50, BERT fine-tune, LLaMA-7B LoRA

## [0.0.1] - 2026-05-25

- Project initialized.
