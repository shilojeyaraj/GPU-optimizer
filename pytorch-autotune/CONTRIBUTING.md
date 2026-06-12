# Contributing to pytorch-autotune

Thanks for your interest. This is a small project with a clear scope, so contributions are most useful when they fit one of these shapes:

## Easy wins

- **Add a recommender** — implement the `Recommender` protocol in `autotune/recommenders/base.py` and drop a new file in `autotune/recommenders/`. One file, ~50 lines. Examples: Ollama (local LLMs), Gemini, a domain-specific heuristic.
- **Add a workload** — drop a file in `experiments/workloads/` that defines `train(config: dict) -> float` returning samples/sec. Examples: ViT, Whisper, Stable Diffusion LoRA.
- **Add a backend** — implement a new execution backend in `autotune/backends/`. Examples: Modal, AWS Batch.

## Dev setup

```bash
git clone https://github.com/shilojeyaraj/pytorch-autotune
cd pytorch-autotune
pip install -e ".[dev]"
pre-commit install
pytest tests/unit -v
```

Unit tests must run **without a GPU**. CI runs on CPU.

## PR process

1. Open an issue first for non-trivial changes — saves both of us time.
2. Keep PRs focused: one feature or fix per PR.
3. Tests required for new code paths.
4. `ruff check` and `mypy autotune/` must pass.

## License

By contributing, you agree your contributions will be licensed under the [Apache License 2.0](LICENSE).
