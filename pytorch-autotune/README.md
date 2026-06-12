# pytorch-autotune

> Stop guessing your batch size. Let an LLM find your optimal training config in 20 trials.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

`pytorch-autotune` is a thin Python library that autotunes **systems-level** training knobs — batch size, precision, `torch.compile` mode, dataloader workers, gradient accumulation, flash attention — using an LLM as the recommender. It plugs into Optuna and HuggingFace Trainer as a drop-in sampler.

## The problem

Existing HPO tools (Optuna, Ray Tune, W&B Sweeps) focus on **quality knobs**: learning rate, weight decay, dropout. Practitioners under-tune the **systems knobs** that govern training throughput — they pick from folklore, copy a blog post, or stop at the first config that doesn't OOM. `pytorch-autotune` closes that gap.

## Install

```bash
pip install pytorch-autotune
```

Optional extras:
```bash
pip install "pytorch-autotune[llm]"   # Anthropic + OpenAI SDKs
pip install "pytorch-autotune[ray]"   # Ray Tune integration
pip install "pytorch-autotune[hf]"    # HuggingFace Trainer integration
pip install "pytorch-autotune[all]"   # everything
```

## 30-second example

```python
from autotune import tune, grid

def train(config):
    # Your training step. Return samples/second.
    ...

result = tune(
    train,
    sweep=grid(
        batch_size=[16, 32, 64, 128],
        precision=["fp32", "bf16"],
        compile_mode=["default", "reduce-overhead"],
    ),
    recommender="claude",        # or "gpt-4o", "heuristic", "random"
    n_trials=20,
    constraints={"max_vram_gb": 24},
)

print(result.best_config)
print(result.recommendation.reasoning)
```

## Status

Pre-alpha. v0.1.0 target: Q3 2026. See [CHANGELOG.md](CHANGELOG.md) and the [execution plan](../README.md) for the roadmap.

## License

[Apache 2.0](LICENSE).
