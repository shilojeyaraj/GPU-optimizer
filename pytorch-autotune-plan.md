# pytorch-autotune: Full Execution Plan

> Living document. Last updated 2026-05-25.
> Goal: ship a credible OSS library + NeurIPS workshop paper as a solo independent researcher with Cohere affiliation and ~$100–120 compute budget.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Before You Write a Line of Code](#2-before-you-write-a-line-of-code)
3. [Repository Setup & OSS Hygiene](#3-repository-setup--oss-hygiene)
4. [Phase 0 — Compute & Scope Lock (Week 1)](#4-phase-0--compute--scope-lock-week-1)
5. [Phase 1 — Library Skeleton (Weeks 2–4)](#5-phase-1--library-skeleton-weeks-24)
6. [Phase 2 — LLM Recommender + Optuna Integration (Weeks 5–6)](#6-phase-2--llm-recommender--optuna-integration-weeks-56)
7. [Phase 3 — Experiments (Weeks 7–12)](#7-phase-3--experiments-weeks-712)
8. [Phase 4 — Paper + v0.1.0 Launch (Weeks 13–16)](#8-phase-4--paper--v010-launch-weeks-1316)
9. [Phase 5 — Community & Adoption (Ongoing)](#9-phase-5--community--adoption-ongoing)
10. [Edge Cases & Failure Modes](#10-edge-cases--failure-modes)
11. [Making It Properly Open Source](#11-making-it-properly-open-source)
12. [Budget Tracker](#12-budget-tracker)
13. [Key Decisions Log](#13-key-decisions-log)

---

## 1. Project Overview

### What you're building

`pytorch-autotune` — a thin Python library that autotunes **systems-level** training knobs (batch size, precision, `torch.compile` mode, dataloader workers, `pin_memory`, `prefetch_factor`, gradient accumulation, flash attention, TF32) using an LLM as the recommender layer, plugging into Optuna and HuggingFace Trainer as a drop-in sampler.

### What makes it different from Optuna/Ray Tune

Those tools target **quality knobs** (learning rate, weight decay, dropout). Nobody has built a good tool for systems knobs, and nobody has rigorously evaluated whether LLMs can recommend them better than heuristics. That's the gap.

### The paper thesis (one sentence)

> An LLM with access to PyTorch documentation and measured sweep results recommends systems-level training configurations better than expert heuristics but worse than exhaustive search — and the cost gap is closing.

### Success metrics

- **Library**: 500 GitHub stars + 5 non-trivial external contributors within 12 months of v0.1.0
- **Paper**: arXiv preprint within 6 months, NeurIPS workshop submission within 9 months

---

## 2. Before You Write a Line of Code

These decisions, if deferred, will force rewrites. Lock them first.

### 2.1 Decide the minimum viable experiment

You cannot run the full 6-workload, 11-dimension design solo on a limited budget. The minimum credible experiment for a workshop paper is:

| Workload | Why keep it |
|---|---|
| ResNet-50 | Canonical CNN baseline, cheap to run, reviewers expect it |
| BERT-base fine-tune | Most common transformer workload, everyone has run this |
| LLaMA-7B LoRA | Proves the tool is relevant to modern LLM fine-tuning |

Cut: Whisper, Stable Diffusion LoRA, ViT. Add them post-launch if the community asks.

Search dimensions to keep (6 of your 11):

| Knob | Why it's in |
|---|---|
| `batch_size` | Biggest throughput lever, universally relevant |
| `precision` | fp32 / bf16 / fp16 — huge impact, easy to measure |
| `compile_mode` | `default` / `reduce-overhead` / `max-autotune` |
| `use_flash_attention` | True/False — major memory + speed impact on transformers |
| `gradient_accumulation_steps` | 1 / 2 / 4 / 8 |
| `dataloader_workers` | 0 / 2 / 4 / 8 |

Cut for now: `pin_memory`, `prefetch_factor`, `persistent_workers`, `ddp_backend`, `allow_tf32`. These are secondary and complicate the search space without adding much to the paper's core claim.

### 2.2 Decide your LLM provider strategy

The recommender needs to be **provider-agnostic** in the library but you need to pick one for the paper experiments. Use Claude Sonnet (claude-sonnet-4-6) as the primary — it's the most capable mid-tier model and you have obvious access. Run a GPT-4o comparison on one workload for the paper to show provider-agnosticism.

Pin: model version, temperature=0, seed where supported. Log every prompt and response verbatim. This is non-negotiable for reproducibility.

### 2.3 Decide your baseline set

| Baseline | Implementation |
|---|---|
| Random search | `random.choice` from grid |
| Expert heuristic | Hardcoded rules table (see Section 5.3) |
| Optuna TPE | `optuna.samplers.TPESampler` out of the box |
| LLM (no docs) | Claude with sweep history only, no PyTorch docs in prompt |
| LLM (with docs) | Claude with docs + sweep history |

The LLM-no-docs vs LLM-with-docs split is your key ablation. Do not skip it.

### 2.4 Confirm compute access

Before touching the repo, do this:

1. Apply to Lambda Labs academic compute program: https://lambdalabs.com/research
2. Apply to Google TPU Research Cloud: https://sites.research.google/trc/
3. Set up a Kaggle account — 30 free GPU-hours/week (T4/P100), useful for development
4. Price out your full experiment on Vast.ai and Thunder Compute; set a hard spend ceiling of $150

---

## 3. Repository Setup & OSS Hygiene

Do this on Day 1. Good OSS hygiene from the start is much easier than retrofitting it.

### 3.1 Repo structure

```
pytorch-autotune/
├── autotune/
│   ├── __init__.py             # Public API: tune(), sweep
│   ├── sweep.py                # Grid/random sweep definitions
│   ├── profile.py              # torch.profiler + NVML wrapper
│   ├── backends/
│   │   ├── local.py            # subprocess fan-out (default)
│   │   ├── ray.py              # ray.tune integration
│   │   └── slurm.py            # submitit integration
│   ├── recommenders/
│   │   ├── base.py             # Recommender protocol (ABC)
│   │   ├── llm.py              # Provider-agnostic LLM recommender
│   │   ├── heuristic.py        # Expert rules baseline
│   │   └── random.py           # Random baseline
│   └── integrations/
│       ├── optuna.py           # LLMSampler
│       ├── ray_tune.py         # LLMSearcher
│       └── hf_trainer.py       # HuggingFace Trainer hook
├── experiments/                # Paper experiment scripts (reproducible)
│   ├── workloads/
│   │   ├── resnet50.py
│   │   ├── bert_finetune.py
│   │   └── llama_lora.py
│   ├── run_experiment.py       # Single entrypoint for all paper experiments
│   ├── config.yaml             # Search space + constraints
│   └── results/                # Raw results (committed to repo)
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
├── examples/
│   ├── quickstart.py
│   └── hf_trainer_example.py
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── release.yml
│   └── ISSUE_TEMPLATE/
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE                     # Apache 2.0
```

### 3.2 License

Use **Apache 2.0**. Not MIT. Apache 2.0 includes an explicit patent grant which matters for corporate adoption — companies are much more comfortable contributing to Apache-licensed projects. PyTorch itself is BSD + Patent grant for the same reason.

### 3.3 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pytorch-autotune"
version = "0.0.1"
description = "LLM-powered systems-knob autotuning for PyTorch training"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0",
    "optuna>=3.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
llm = ["anthropic>=0.20", "openai>=1.0"]
ray = ["ray[tune]>=2.0"]
hf = ["transformers>=4.30"]
dev = ["pytest", "pytest-cov", "ruff", "mypy", "pre-commit"]
all = ["pytorch-autotune[llm,ray,hf]"]
```

Key choices here:
- `torch` is a dependency but NOT pinned — don't break people's existing environments
- LLM providers are optional extras so the library installs cleanly without API keys
- `pydantic` for config validation — catches bad configs before they waste GPU time

### 3.4 CI from day one

`.github/workflows/ci.yml` — runs on every PR:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ --cov=autotune
      - run: ruff check autotune/
      - run: mypy autotune/
```

Note: unit tests must run without a GPU. Use `torch.device("cpu")` in tests and mock NVML calls. If your test suite requires a GPU it won't run in CI and will rot immediately.

### 3.5 Pre-commit hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
```

---

## 4. Phase 0 — Compute & Scope Lock (Week 1)

**Goal**: Know exactly what you're going to run and how much it costs before writing library code.

### 4.1 Lock the experiment budget

Write a spreadsheet with:
- Workload × config count × repeat trials × estimated trial duration × GPU $/hr
- Separate line for development runs (expect 2–3x your estimate for mistakes)
- Hard ceiling: $150 total

Estimated breakdown:

| Item | GPU | Duration | Cost |
|---|---|---|---|
| ResNet-50 sweep (90 configs × 3 repeats × ~8 min) | RTX 4090 @ $0.46/hr | ~36 hrs | ~$17 |
| BERT fine-tune sweep | RTX 4090 | ~48 hrs | ~$22 |
| LLaMA-7B LoRA sweep | RTX 4090 | ~90 hrs | ~$41 |
| Ablation (LLM w/ vs w/o docs, ResNet-50 only) | RTX 4090 | ~15 hrs | ~$7 |
| Cross-GPU validation (ResNet-50 on A100) | A100 @ $0.78/hr | ~12 hrs | ~$9 |
| Buffer (reruns, mistakes, OOMs) | — | — | ~$30 |
| **Total** | | | **~$126** |

### 4.2 Set up Habitat

Install and test Habitat (https://github.com/geoffxy/habitat) on whatever hardware you have access to (even a consumer GPU or borrowed machine). Habitat lets you profile a few iterations locally then predict throughput on a target GPU with ~12% mean error. This is how you validate A100 numbers without paying A100 prices for your full sweep.

```bash
git clone https://github.com/geoffxy/habitat
cd habitat
# Follow Docker setup in their README
# Run extract-models.sh, setup.sh, start.sh
```

Habitat requires GPU performance counter access — note this needs specific Docker/driver permissions. Test this early; don't discover it's broken on your cloud instance after you've paid for it.

### 4.3 Write the pre-registration document

Before running any experiments, write a 1-page document stating:
- Exact hypothesis
- Exact baselines
- Exact metrics
- Exact workloads and configs
- Expected direction of results

Commit this to the repo as `experiments/preregistration.md` with today's date. This matters for the paper — reviewers at systems venues are increasingly skeptical of post-hoc framing, and having a dated pre-registration is a simple credibility signal.

---

## 5. Phase 1 — Library Skeleton (Weeks 2–4)

**Goal**: `pip install pytorch-autotune` works, the API surface is defined, and everything runs on CPU for testing.

### 5.1 The Recommender protocol

This is the most important design decision in the library. Everything plugs into this.

```python
# autotune/recommenders/base.py
from typing import Protocol, Any
from dataclasses import dataclass

@dataclass
class Suggestion:
    config: dict[str, Any]
    reasoning: str | None = None
    confidence: float | None = None

class Recommender(Protocol):
    def suggest(
        self,
        search_space: dict[str, list],
        history: list[tuple[dict, float]],  # (config, throughput) pairs
        constraints: dict[str, Any],
    ) -> Suggestion:
        ...
```

Keep this protocol minimal. It's the contract that all recommenders (random, heuristic, LLM, future ones you haven't thought of) implement. Don't put provider-specific logic here.

### 5.2 The sweep definitions

```python
# autotune/sweep.py
from dataclasses import dataclass, field
from typing import Any
import itertools
import random

@dataclass
class SweepGrid:
    params: dict[str, list]

    def all_configs(self) -> list[dict]:
        keys = list(self.params.keys())
        values = list(self.params.values())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def sample(self, n: int, seed: int | None = None) -> list[dict]:
        rng = random.Random(seed)
        all_configs = self.all_configs()
        return rng.sample(all_configs, min(n, len(all_configs)))

def grid(**kwargs) -> SweepGrid:
    return SweepGrid(params=kwargs)
```

### 5.3 The expert heuristic baseline

This is the baseline that matters most for the paper. Encode what a senior ML engineer would actually do:

```python
# autotune/recommenders/heuristic.py

GPU_RULES = {
    # (gpu_name_fragment, vram_gb_min): config overrides
    "A100": {"precision": "bf16", "use_flash_attention": True, "compile_mode": "reduce-overhead"},
    "H100": {"precision": "bf16", "use_flash_attention": True, "compile_mode": "max-autotune"},
    "4090": {"precision": "bf16", "use_flash_attention": True, "compile_mode": "reduce-overhead"},
    "3090": {"precision": "fp16", "use_flash_attention": True, "compile_mode": "default"},
    "T4":   {"precision": "fp16", "use_flash_attention": False, "compile_mode": "default"},
}

BATCH_SIZE_RULES = {
    # vram_gb -> safe starting batch size for different model families
    "resnet50": {80: 256, 40: 128, 24: 64, 16: 32},
    "bert":     {80: 64,  40: 32,  24: 16, 16: 8},
    "llama7b":  {80: 8,   40: 4,   24: 2,  16: 1},
}
```

Document your sources for these rules — blog posts, NVIDIA tuning guides, MLPerf reports. The paper reviewer will ask "where do these rules come from."

### 5.4 The local backend

The trial runner is the trickiest part to get right. Each trial must be isolated — a crash or OOM in one trial must not kill the whole sweep.

```python
# autotune/backends/local.py
import subprocess
import sys
import json
import tempfile

def run_trial(train_fn_path: str, config: dict, timeout_seconds: int = 3600) -> dict:
    """
    Run a single trial in a fresh subprocess.
    Returns {"throughput": float, "error": str | None, "oom": bool}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-m", "autotune._trial_runner", train_fn_path, config_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if result.returncode != 0:
            oom = "CUDA out of memory" in result.stderr
            return {"throughput": None, "error": result.stderr[-2000:], "oom": oom}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"throughput": None, "error": "timeout", "oom": False}
```

**Critical**: the OOM detection (`"CUDA out of memory" in result.stderr`) is important for your paper's OOM rate metric. Log every OOM with the full config — this data is interesting in itself.

### 5.5 Config validation with VRAM estimation

This is where you need to be careful about what you promise. VRAM estimation before running is hard. Be conservative and call it a "VRAM floor estimate" not a guarantee.

```python
# autotune/profile.py

VRAM_ESTIMATES_GB = {
    # (model_family, batch_size, precision) -> rough lower bound in GB
    # These are conservative — real usage will be higher
    ("resnet50",   32, "fp32"): 4.0,
    ("resnet50",   32, "bf16"): 2.5,
    ("bert_base",  16, "fp32"): 6.0,
    ("bert_base",  16, "bf16"): 3.5,
    ("llama_7b",    1, "bf16"): 16.0,  # LoRA only
    ("llama_7b",    2, "bf16"): 20.0,
}

def estimate_vram_gb(model_family: str, config: dict) -> float | None:
    """Returns a lower-bound VRAM estimate, or None if unknown."""
    key = (model_family, config.get("batch_size"), config.get("precision"))
    return VRAM_ESTIMATES_GB.get(key)
```

When the estimate is unavailable, don't block the trial — just warn and let it run. Better to OOM and log it than to silently skip configs.

### 5.6 Main `tune()` entrypoint

```python
# autotune/__init__.py
from autotune.sweep import SweepGrid, grid
from autotune.backends.local import run_trial
from autotune.recommenders.llm import LLMRecommender
from autotune.recommenders.heuristic import HeuristicRecommender
from autotune.recommenders.random import RandomRecommender
from dataclasses import dataclass, field

@dataclass
class TuneResult:
    best_config: dict
    best_throughput: float
    all_results: list[dict]
    recommendation: object | None = None

def tune(
    train_fn,
    sweep: SweepGrid,
    recommender: str | None = "heuristic",
    backend: str = "local",
    n_trials: int = 20,
    constraints: dict | None = None,
    seed: int = 42,
) -> TuneResult:
    """
    Main entrypoint. Runs a sweep of training configs and returns the best.

    Args:
        train_fn: callable that takes a config dict and returns samples/sec
        sweep: SweepGrid defining the search space
        recommender: "claude", "gpt-4o", "heuristic", "random", or None (grid)
        backend: "local", "ray", or "slurm"
        n_trials: number of configs to evaluate
        constraints: dict of hard constraints e.g. {"max_vram_gb": 24}
        seed: random seed for reproducibility
    """
    ...
```

### 5.7 Testing strategy (no GPU required)

Write tests that:
1. Mock `run_trial` to return fake throughput values
2. Verify the sweep samples the right number of configs
3. Verify the heuristic recommender returns valid configs
4. Verify OOM configs are logged correctly
5. Verify `TuneResult` contains expected fields

```python
# tests/unit/test_sweep.py
def test_grid_generates_correct_configs():
    s = grid(batch_size=[16, 32], precision=["fp32", "bf16"])
    configs = s.all_configs()
    assert len(configs) == 4
    assert {"batch_size": 16, "precision": "fp32"} in configs

# tests/unit/test_local_backend.py
def test_oom_is_detected(monkeypatch):
    def mock_run(*args, **kwargs):
        return MockResult(returncode=1, stderr="CUDA out of memory: tried to allocate 2GB")
    monkeypatch.setattr(subprocess, "run", mock_run)
    result = run_trial("dummy_fn", {"batch_size": 512})
    assert result["oom"] is True
```

---

## 6. Phase 2 — LLM Recommender + Optuna Integration (Weeks 5–6)

**Goal**: The LLM recommender works end-to-end. The Optuna `LLMSampler` is functional. Smoke test passes on ResNet-50.

### 6.1 The LLM recommender

The prompt design is where the paper lives. This is not boilerplate — iterate on it carefully.

```python
# autotune/recommenders/llm.py

SYSTEM_PROMPT = """You are an expert PyTorch training engineer specializing in systems-level performance optimization. Your job is to recommend the next configuration to try in a hyperparameter sweep, optimizing for training throughput (samples/second).

You have access to:
1. The current search space (what values are available for each knob)
2. The history of already-tried configurations and their measured throughput
3. Hardware constraints (VRAM budget, GPU type if known)

Rules:
- Only suggest configs within the provided search space
- Never suggest a config identical to one already tried
- Explain your reasoning briefly (1-2 sentences)
- Return valid JSON matching the schema provided

PyTorch systems optimization knowledge:
- bf16 is preferred over fp16 on A100/H100/RTX 30xx+ for stability and speed
- torch.compile reduce-overhead mode benefits small models; max-autotune benefits large ones
- Flash attention dramatically reduces memory bandwidth for attention layers (transformers only)
- Gradient accumulation trades throughput for effective batch size — only use when batch_size is memory-constrained
- DataLoader workers: 4 is usually optimal; more than 8 rarely helps and can cause CPU contention
- pin_memory=True helps when CPU→GPU transfer is a bottleneck (usually when workers > 0)
"""

def build_prompt(search_space, history, constraints):
    history_str = "\n".join([
        f"  Config: {config} → Throughput: {throughput:.1f} samples/sec"
        for config, throughput in history
    ])
    return f"""Search space: {search_space}

Constraints: {constraints}

Trial history ({len(history)} trials so far):
{history_str if history else "  (no trials yet)"}

Suggest the next configuration to maximize throughput.
Respond with valid JSON only:
{{
  "config": {{"batch_size": ..., "precision": ..., ...}},
  "reasoning": "..."
}}"""
```

**Important for the paper**: the `SYSTEM_PROMPT` docstring content is part of your "LLM with docs" condition. The "LLM without docs" condition removes everything after "Rules:" and just keeps the task description. This is your ablation.

### 6.2 Provider-agnostic implementation

```python
import anthropic
import openai
import json

class LLMRecommender:
    def __init__(self, provider: str = "claude", model: str | None = None, temperature: float = 0.0):
        self.provider = provider
        self.temperature = temperature
        if provider == "claude":
            self.client = anthropic.Anthropic()
            self.model = model or "claude-sonnet-4-6"
        elif provider == "gpt-4o":
            self.client = openai.OpenAI()
            self.model = model or "gpt-4o"
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'gpt-4o'.")

    def suggest(self, search_space, history, constraints):
        prompt = build_prompt(search_space, history, constraints)
        
        # Log the full prompt — critical for reproducibility
        self._log_prompt(prompt)
        
        if self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
        elif self.provider == "gpt-4o":
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content

        self._log_response(raw)
        
        try:
            parsed = json.loads(raw)
            return Suggestion(config=parsed["config"], reasoning=parsed.get("reasoning"))
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to random if LLM returns garbage
            self._log_error(f"LLM parse error: {e}, raw: {raw}")
            return RandomRecommender().suggest(search_space, history, constraints)
```

The fallback to random on parse failure is important — don't let a malformed LLM response crash a sweep in progress.

### 6.3 Logging (non-negotiable)

Every LLM interaction must be logged. This is both for reproducibility and for the paper's appendix.

```python
import json
import datetime
from pathlib import Path

class LLMLogger:
    def __init__(self, log_dir: str = "autotune_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, event_type: str, data: dict):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": event_type,
            **data,
        }
        with open(self.log_dir / f"{self.session_id}.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
```

Log: prompt sent, raw response received, parsed config, whether fallback was used, trial result when it comes back. You want to be able to reconstruct every decision the LLM made after the fact.

### 6.4 Optuna LLMSampler

```python
# autotune/integrations/optuna.py
import optuna
from autotune.recommenders.llm import LLMRecommender
from autotune.recommenders.base import Suggestion

class LLMSampler(optuna.samplers.BaseSampler):
    def __init__(self, provider: str = "claude", include_docs: bool = True):
        self.recommender = LLMRecommender(provider=provider)
        self._include_docs = include_docs  # ablation flag

    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(study)

    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}
        
        history = [
            (t.params, t.value)
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        
        suggestion = self.recommender.suggest(
            search_space=search_space,
            history=history,
            constraints={},
        )
        
        # Validate that suggestion is within search space
        validated = {}
        for param, dist in search_space.items():
            if param in suggestion.config:
                val = suggestion.config[param]
                if isinstance(dist, optuna.distributions.CategoricalDistribution):
                    if val in dist.choices:
                        validated[param] = val
        
        return validated

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Fallback for params not covered by sample_relative
        return optuna.samplers.RandomSampler().sample_independent(
            study, trial, param_name, param_distribution
        )
```

### 6.5 Smoke test

Write a minimal smoke test that runs the full pipeline on CPU with a toy train function:

```python
# tests/integration/test_smoke.py
def toy_train(config):
    """Fake train function that returns throughput proportional to batch_size."""
    import time
    time.sleep(0.1)  # simulate work
    # Bigger batch = more throughput (simplified)
    return config["batch_size"] * (2.0 if config["precision"] == "bf16" else 1.0)

def test_tune_smoke():
    from autotune import tune, grid
    result = tune(
        toy_train,
        sweep=grid(batch_size=[16, 32], precision=["fp32", "bf16"]),
        recommender="random",
        n_trials=4,
    )
    assert result.best_config is not None
    assert result.best_throughput > 0
    assert len(result.all_results) == 4
```

---

## 7. Phase 3 — Experiments (Weeks 7–12)

**Goal**: Run the full paper experiment matrix. Generate the numbers the paper is built on.

### 7.1 Workload implementations

Each workload implements the same interface: takes a config dict, returns samples/sec. This is also what users of the library will write.

```python
# experiments/workloads/resnet50.py
import torch
import torchvision.models as models
import time

def train(config: dict) -> float:
    """Returns samples/sec over WARMUP+MEASURE steps."""
    WARMUP_STEPS = 5
    MEASURE_STEPS = 20
    BATCH_SIZE = config["batch_size"]

    device = torch.device("cuda")
    model = models.resnet50().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Apply config
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[config["precision"]]
    if config.get("compile_mode") and config["compile_mode"] != "none":
        model = torch.compile(model, mode=config["compile_mode"])

    fake_input = torch.randn(BATCH_SIZE, 3, 224, 224, device=device, dtype=dtype)
    fake_labels = torch.randint(0, 1000, (BATCH_SIZE,), device=device)

    # Warmup
    for _ in range(WARMUP_STEPS):
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = torch.nn.functional.cross_entropy(model(fake_input), fake_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = torch.nn.functional.cross_entropy(model(fake_input), fake_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    return (BATCH_SIZE * MEASURE_STEPS) / elapsed  # samples/sec
```

**Critical benchmarking note**: always call `torch.cuda.synchronize()` before starting and stopping your timer. Without it, CUDA operations are async and your timing will be wrong. Always do warmup steps before measurement.

### 7.2 The experiment runner

```python
# experiments/run_experiment.py
"""
Single entrypoint for all paper experiments.
Usage: python run_experiment.py --workload resnet50 --baseline llm_with_docs --n-trials 30 --seed 42
"""
import argparse
import json
import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["resnet50", "bert", "llama_lora"])
    parser.add_argument("--baseline", choices=["random", "heuristic", "optuna_tpe", "llm_no_docs", "llm_with_docs"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat index (1, 2, or 3)")
    args = parser.parse_args()

    # Results go here
    run_id = f"{args.workload}_{args.baseline}_seed{args.seed}_rep{args.repeat}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_path = Path("experiments/results") / f"{run_id}.json"
    results_path.parent.mkdir(exist_ok=True)

    # ... run the sweep ...
    
    # Save everything
    with open(results_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "args": vars(args),
            "results": results,
            "best_config": best_config,
            "best_throughput": best_throughput,
        }, f, indent=2)

    print(f"Results saved to {results_path}")
```

**Run with `--repeat 1`, `--repeat 2`, `--repeat 3` for each config**. Three repeats per (workload, baseline) combination. This is the minimum for statistical credibility.

### 7.3 Measurement checklist (run before each experiment session)

Before starting any benchmarking session on a cloud instance, do this:

- [ ] Verify you're on a dedicated instance, not shared (check `nvidia-smi` — no other processes using GPU)
- [ ] Set GPU to persistence mode: `sudo nvidia-smi -pm 1`
- [ ] Lock GPU clocks if possible: `sudo nvidia-smi --lock-gpu-clocks=<min>,<max>`
- [ ] Record exact GPU model, driver version, CUDA version, PyTorch version
- [ ] Run 3 warmup trials before starting measurement trials
- [ ] Record ambient GPU temperature at start (`nvidia-smi --query-gpu=temperature.gpu`)
- [ ] Save `nvidia-smi -q` output to a file alongside results

These details go in the paper's experimental setup section. Reviewers check for this.

### 7.4 What to log per trial

Every trial result must include:

```json
{
  "config": {"batch_size": 32, "precision": "bf16", "compile_mode": "reduce-overhead", ...},
  "throughput_samples_per_sec": 847.3,
  "trial_duration_seconds": 42.1,
  "oom": false,
  "error": null,
  "gpu_memory_used_mb": 12480,
  "gpu_utilization_pct": 94,
  "cuda_version": "12.1",
  "torch_version": "2.3.0",
  "gpu_model": "NVIDIA RTX 4090",
  "timestamp": "2026-06-15T14:23:01Z",
  "baseline": "llm_with_docs",
  "workload": "resnet50",
  "repeat": 1,
  "llm_prompt": "...",
  "llm_response": "..."
}
```

Commit all raw results to the repo. This is your reproducibility artifact.

### 7.5 The key paper figures to aim for

**Figure 1 (main result)**: For each workload, a line chart showing throughput vs. number of trials for each baseline. X-axis: trial number 1–30. Y-axis: best throughput found so far. Five lines: random, heuristic, Optuna TPE, LLM-no-docs, LLM-with-docs.

**Figure 2 (ablation)**: Bar chart on ResNet-50 only. Same metric at trial N=10, N=20, N=30. Shows the docs matter (or don't).

**Table 1**: OOM rate per baseline per workload. If LLM-with-docs has a lower OOM rate than random, that's an interesting secondary result.

**Table 2**: Time-to-90%-of-best-config. How many trials does each method need?

### 7.6 Statistical reporting

Do not report a single number. Report mean ± std across your 3 repeats. If a result differs across repeats by more than 5%, investigate before including it in the paper — noisy measurements are a common rejection reason.

---

## 8. Phase 4 — Paper + v0.1.0 Launch (Weeks 13–16)

**Goal**: Paper draft ready, library released simultaneously.

### 8.1 Paper structure (NeurIPS workshop format, ~6 pages)

```
1. Introduction (0.5 pages)
   - The systems knob gap
   - Why LLMs are a natural fit
   - Paper contributions (bullet list)

2. Background & Related Work (0.75 pages)
   - Existing HPO tools and their quality-knob focus
   - LLMs as optimizers literature
   - GPU performance prediction (Habitat, roofline)

3. pytorch-autotune (1 page)
   - Library design (figure: architecture diagram)
   - The recommender protocol
   - Optuna LLMSampler
   - Reproducibility artifacts

4. Experimental Setup (0.75 pages)
   - Workloads
   - Search space
   - Baselines
   - Hardware
   - Measurement methodology

5. Results (1.5 pages)
   - Figure 1: main throughput vs trials
   - Table 1: OOM rates
   - Table 2: time-to-90%-of-best
   - Figure 2: ablation

6. Discussion (0.5 pages)
   - When does the LLM help most?
   - Failure modes observed
   - Cost analysis

7. Conclusion (0.25 pages)
```

### 8.2 Writing the paper honestly

The result may not be "LLM wins." That's fine. The three honest framings:

- **LLM wins**: "LLM-driven recommendation is a viable lightweight alternative for systems knobs"
- **LLM loses**: "We provide a failure taxonomy; current LLMs are not yet reliable systems advisors"
- **LLM ties heuristic**: "Expert rule tables remain competitive; LLM cost is not justified for this task"

Each of these is a real contribution. The pre-registration document you wrote in Phase 0 protects you — you committed to the question before seeing the answer.

### 8.3 v0.1.0 release checklist

- [ ] All public API has docstrings
- [ ] README has a 5-line quickstart that works
- [ ] README has benchmark table (even one workload is fine)
- [ ] CHANGELOG.md exists and documents v0.1.0
- [ ] CONTRIBUTING.md explains how to add a new workload or recommender
- [ ] `pip install pytorch-autotune` works cleanly in a fresh virtualenv
- [ ] GitHub Actions CI passes on Python 3.9, 3.10, 3.11
- [ ] PyPI release is tagged and published
- [ ] arXiv preprint is live with the same day as PyPI release
- [ ] GitHub README links to arXiv paper

### 8.4 The README structure that drives adoption

The README is your most important marketing document. Structure:

```markdown
# pytorch-autotune

> Stop guessing your batch size. Let an LLM find your optimal training config in 20 trials.

[badges: PyPI version | CI status | License | arXiv]

## The problem
[2 sentences: systems knobs are undertued, existing tools don't help]

## Install
pip install pytorch-autotune

## 30-second example
[5-line code block showing the happy path]

## Benchmark results
[table: your main figure in markdown form]

## How it works
[1 paragraph + architecture diagram]

## Integrations
[Optuna | HuggingFace Trainer | Ray Tune]

## Paper
[cite your arXiv preprint]
```

---

## 9. Phase 5 — Community & Adoption (Ongoing)

### 9.1 Launch sequence

Do these in order, same week as v0.1.0:

1. **Kaggle notebook** — a runnable demo using Kaggle's free T4 GPU. This is your lowest-friction demo; anyone can try it with zero setup.
2. **HuggingFace post** — post on HF community forums under "Research" category. The HF audience is exactly your target user.
3. **Hacker News "Show HN"** — title: "Show HN: pytorch-autotune – LLM-powered systems-knob autotuning for PyTorch". Post on a Tuesday or Wednesday morning.
4. **PyTorch forums** — post in the "Tools" category
5. **r/MachineLearning** — after HN, cross-post

Do NOT do all of these on the same day. HN first, then one per day for the following week.

### 9.2 The HuggingFace Trainer PR

This is your biggest adoption multiplier. File a PR to `huggingface/transformers` adding an example showing `LLMSampler` with `trainer.hyperparameter_search()`. Even if it's just documentation, it exposes the library to every HF user who reads the HPO tutorial.

### 9.3 First 5 issues to open yourself

Seed the issue tracker with well-defined, approachable contribution opportunities:

1. `[good first issue]` Add support for `ollama` local LLMs as a provider
2. `[good first issue]` Add ViT-B/16 as a benchmark workload
3. `[enhancement]` Implement `LLMSearcher` for Ray Tune (analog to `LLMSampler`)
4. `[enhancement]` Add SLURM backend via `submitit`
5. `[research]` Evaluate on AMD GPU (MI250/MI300) — hardware diversity for the paper follow-up

### 9.4 What makes a good external contributor

The 5-contributor goal requires genuine engagement. Make it easy:
- CONTRIBUTING.md explains the Recommender protocol so adding a new recommender is a 1-file change
- Dev setup is `pip install -e ".[dev]" && pre-commit install` — nothing exotic
- Every PR gets a response within 48 hours
- Be specific in issue descriptions — "implement `sample_relative` following the pattern in `llm.py`" beats "add new sampler"

---

## 10. Edge Cases & Failure Modes

These will happen. Have a plan.

### 10.1 LLM returns an invalid config

**Scenario**: LLM suggests `{"batch_size": 512, "precision": "fp8"}` where fp8 is not in your search space.

**Handling**: Validate every LLM suggestion against the search space before running it. If validation fails, log the invalid suggestion and fall back to random for that trial. Never skip the trial silently — count it as a trial and log the fallback.

### 10.2 OOM mid-sweep

**Scenario**: Trial 23 of 30 crashes with CUDA OOM, killing the process.

**Handling**: Subprocess isolation (Phase 1, Section 5.4) prevents this from killing the sweep. The trial returns `{"oom": True}`, gets logged, and the sweep continues. The recommender should use OOM results as signal — a config that OOMed tells the LLM something important about memory constraints.

### 10.3 Stochastic throughput measurements

**Scenario**: Same config returns 847 samples/sec on run 1 and 791 samples/sec on run 2 (7% variance).

**Handling**: This is normal, especially for `torch.compile` which has JIT warm-up effects. Use 3 repeat runs and report mean ± std. If variance exceeds 10% on a stable config, investigate: check for thermal throttling, background processes, or GPU clock state inconsistency. Do not paper over it.

### 10.4 LLM API rate limits or downtime

**Scenario**: 50 trials into a sweep, the Anthropic API starts returning 429s.

**Handling**: Implement exponential backoff in the LLM client. Cache responses keyed by (prompt_hash, model, temperature) so if you restart a sweep you don't re-query for identical situations. Fall back to heuristic recommender after 3 failed retries.

### 10.5 `torch.compile` failures

**Scenario**: A specific model + compile_mode combination throws a compilation error, not an OOM.

**Handling**: Catch `torch._dynamo.exc.BackendCompilerFailed` and similar compile errors separately from OOM. Log them with their full stack trace. These are interesting failure modes — a compile failure is different from an OOM and should be reported separately in the paper.

### 10.6 User's train function has side effects

**Scenario**: User's `train()` function writes checkpoints, modifies global state, or requires a specific working directory.

**Handling**: Document clearly that `train_fn` should be stateless and idempotent. In the subprocess runner, set `cwd` to a temp directory and clean it up after each trial. Can't prevent all side effects but can minimize blast radius.

### 10.7 Windows support

Your library will get Windows issues. The subprocess backend uses Unix-style process management and NVML may not be available.

**Handling**: Don't promise Windows support in v0.1.0. Put `Platform: Linux, macOS` in the README. Open a GitHub issue titled "Windows support tracking" to acknowledge it exists and invite contributions. WSL2 is a reasonable workaround to document.

---

## 11. Making It Properly Open Source

"Properly open source" means more than a public GitHub repo. Here's the full checklist.

### 11.1 Legal

- [ ] **License file**: `LICENSE` in repo root, full Apache 2.0 text
- [ ] **Copyright header**: add to every `.py` file: `# Copyright 2026 [Your Name]. Apache 2.0 License.`
- [ ] **NOTICE file**: required by Apache 2.0 for attribution. List any dependencies and their licenses.
- [ ] **No proprietary dependencies**: every dependency in `pyproject.toml` must be OSS-compatible. Check each one.
- [ ] **CLA decision**: decide whether you want a Contributor License Agreement. For a small project, it's overkill and discourages contributions. Skip it for now.

### 11.2 Documentation

Minimum viable docs for v0.1.0:

- [ ] **README.md**: quickstart, benchmark table, how it works, integrations
- [ ] **CONTRIBUTING.md**: dev setup, how to add a workload, how to add a recommender, PR process
- [ ] **CHANGELOG.md**: v0.1.0 entry with what's included
- [ ] **API reference**: use `pdoc` or `mkdocs` with auto-generated docs from docstrings. Don't hand-write API docs initially — they go stale.
- [ ] **examples/**: at least two working examples: standalone `tune()` and HF Trainer integration

### 11.3 Versioning

Use [Semantic Versioning](https://semver.org/). For a research project:
- `0.x.y` = pre-stable; breaking changes allowed between minor versions
- `1.0.0` = stable API, breaking changes require major version bump

Start at `0.1.0`. Don't start at `1.0.0` — it sets expectations you don't need yet.

### 11.4 Reproducibility as a first-class feature

This is what makes it an open science contribution, not just a library:

- [ ] `experiments/` directory with all paper experiment scripts committed
- [ ] `experiments/results/` with raw JSON results committed (they're small)
- [ ] One-command reproduction: `python experiments/run_experiment.py --workload resnet50 --baseline llm_with_docs` should reproduce paper results on the same hardware
- [ ] `experiments/README.md` documenting exact hardware, software versions, and commands used for paper results
- [ ] Pin all experiment dependencies: `experiments/requirements.txt` with exact versions

### 11.5 Community health files

GitHub shows these as "community standards" on the Insights tab:

- [ ] `README.md`
- [ ] `LICENSE`
- [ ] `CONTRIBUTING.md`
- [ ] `CODE_OF_CONDUCT.md` — use the Contributor Covenant template, 2 minutes to add
- [ ] `SECURITY.md` — even minimal: "report security issues to [email]"
- [ ] Issue templates (`.github/ISSUE_TEMPLATE/bug_report.md`, `feature_request.md`)
- [ ] PR template (`.github/pull_request_template.md`)

### 11.6 PyPI publishing

```toml
# pyproject.toml additions
[project.urls]
Homepage = "https://github.com/yourusername/pytorch-autotune"
Documentation = "https://pytorch-autotune.readthedocs.io"
Repository = "https://github.com/yourusername/pytorch-autotune"
"Bug Tracker" = "https://github.com/yourusername/pytorch-autotune/issues"
Paper = "https://arxiv.org/abs/XXXX.XXXXX"
```

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ["v*"]
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

Use `gh release create v0.1.0` to trigger the release workflow.

---

## 12. Budget Tracker

Update this as you spend.

| Item | Estimated | Actual | Notes |
|---|---|---|---|
| Lambda Labs academic grant application | $0 | — | Apply week 1 |
| Kaggle development (free tier) | $0 | $0 | 30 hrs/week T4 |
| ResNet-50 experiments (RTX 4090, Vast.ai) | $17 | — | |
| BERT experiments (RTX 4090, Vast.ai) | $22 | — | |
| LLaMA-7B LoRA experiments (RTX 4090, Vast.ai) | $41 | — | |
| Ablation experiments | $7 | — | |
| A100 cross-validation (Thunder Compute) | $9 | — | |
| LLM API costs (Claude + GPT-4o) | $5 | — | |
| Buffer | $30 | — | |
| **Total ceiling** | **$131** | — | Hard stop at $150 |

---

## 13. Key Decisions Log

Record decisions here as you make them. This doubles as a methods section for the paper.

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-25 | Cut to 3 workloads (ResNet50, BERT, LLaMA-7B LoRA) | Budget and time constraints |
| 2026-05-25 | Cut to 6 search dimensions | Simplify for v0.1.0; expand post-launch |
| 2026-05-25 | Target NeurIPS workshop, not MLSys | Achievable solo; use as stepping stone |
| 2026-05-25 | Apache 2.0 license | Corporate adoption; patent grant |
| 2026-05-25 | Claude Sonnet as primary LLM, GPT-4o for one workload comparison | Access + provider-agnosticism signal |
| — | — | — |

---

*Next action: Apply to Lambda Labs academic compute program and set up Vast.ai account with $20 test credit. Run the Habitat smoke test. Then start the repo.*
