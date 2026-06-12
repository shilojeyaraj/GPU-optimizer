"""
Expert-rules baseline. The most important baseline for the paper — encodes
what a senior ML engineer with current PyTorch knowledge would actually pick.

Sources for these rules:
- NVIDIA Deep Learning Performance Guide
  https://docs.nvidia.com/deeplearning/performance/
- PyTorch official tuning guide
  https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- MLPerf Training v4.0 reference implementations (config rationale documented per submission)
- PyTorch torch.compile blog posts (2023-2024)
- Empirical adoption patterns from public HuggingFace model cards (2024-2025)
"""
from __future__ import annotations

from typing import Any

from autotune.recommenders.base import Recommender, Suggestion


# GPU model fragment -> recommended overrides.
# We match by substring of torch.cuda.get_device_name(), longest-first.
GPU_RULES: dict[str, dict[str, Any]] = {
    "H100":  {"precision": "bf16", "use_flash_attention": True, "compile_mode": "max-autotune"},
    "A100":  {"precision": "bf16", "use_flash_attention": True, "compile_mode": "reduce-overhead"},
    "L40":   {"precision": "bf16", "use_flash_attention": True, "compile_mode": "reduce-overhead"},
    "4090":  {"precision": "bf16", "use_flash_attention": True, "compile_mode": "reduce-overhead"},
    "3090":  {"precision": "fp16", "use_flash_attention": True, "compile_mode": "default"},
    "T4":    {"precision": "fp16", "use_flash_attention": False, "compile_mode": "default"},
    "P100":  {"precision": "fp16", "use_flash_attention": False, "compile_mode": "default"},
}


# Recommended starting batch size by (model_family, vram_gb_tier).
# Tier = floor of available VRAM in GB; pick the highest tier <= available.
BATCH_SIZE_RULES: dict[str, dict[int, int]] = {
    "resnet50":  {80: 256, 40: 128, 24: 64, 16: 32, 8: 16},
    "bert_base": {80: 64,  40: 32,  24: 16, 16: 8,  8: 4},
    "llama_7b":  {80: 8,   40: 4,   24: 2,  16: 1,  8: 1},
}


# Heuristic defaults — applied when the GPU/model isn't recognized.
DEFAULTS = {
    "dataloader_workers": 4,
    "gradient_accumulation_steps": 1,
    "use_flash_attention": False,
    "compile_mode": "default",
    "precision": "fp32",
}


def detect_gpu_overrides() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        name = torch.cuda.get_device_name(0)
    except Exception:
        return {}

    for fragment, overrides in sorted(
        GPU_RULES.items(), key=lambda kv: -len(kv[0])
    ):
        if fragment in name:
            return dict(overrides)
    return {}


def recommend_batch_size(model_family: str, vram_gb: float) -> int | None:
    rules = BATCH_SIZE_RULES.get(model_family)
    if rules is None:
        return None
    eligible = [(tier, bs) for tier, bs in rules.items() if tier <= vram_gb]
    if not eligible:
        return None
    return max(eligible, key=lambda tb: tb[0])[1]


class HeuristicRecommender(Recommender):
    """
    Picks the first config from rules, then iterates through the search space
    in a deterministic order. Does not use trial history — that's the point;
    a static rules table is the contrast we're measuring the LLM against.

    Args:
        model_family: one of BATCH_SIZE_RULES keys, or None to skip batch sizing
        vram_gb: available VRAM (for batch size selection); inferred if None
    """

    def __init__(
        self, model_family: str | None = None, vram_gb: float | None = None
    ):
        self.model_family = model_family
        self.vram_gb = vram_gb if vram_gb is not None else _detect_vram_gb()
        self._enumeration: list[dict] | None = None
        self._cursor = 0

    def suggest(
        self,
        search_space: dict[str, list],
        history: list[tuple[dict, float]],
        constraints: dict[str, Any],
    ) -> Suggestion:
        if not history:
            return Suggestion(
                config=self._initial_config(search_space),
                reasoning="heuristic: GPU + model-family defaults",
            )

        if self._enumeration is None:
            self._enumeration = _enumerate_priority(search_space, self._initial_config(search_space))

        tried = {tuple(sorted(cfg.items())) for cfg, _ in history}
        while self._cursor < len(self._enumeration):
            candidate = self._enumeration[self._cursor]
            self._cursor += 1
            if tuple(sorted(candidate.items())) not in tried:
                return Suggestion(
                    config=candidate, reasoning="heuristic: next-best by rule ordering"
                )

        # Fallback: walk all configs.
        for cfg in _all_configs(search_space):
            if tuple(sorted(cfg.items())) not in tried:
                return Suggestion(
                    config=cfg, reasoning="heuristic: search-space exhaustion fallback"
                )
        return Suggestion(
            config=self._initial_config(search_space),
            reasoning="heuristic: search space exhausted",
        )

    def _initial_config(self, search_space: dict[str, list]) -> dict:
        overrides = {**DEFAULTS, **detect_gpu_overrides()}
        if self.model_family and self.vram_gb:
            bs = recommend_batch_size(self.model_family, self.vram_gb)
            if bs is not None:
                overrides["batch_size"] = bs

        config: dict = {}
        for key, options in search_space.items():
            preferred = overrides.get(key)
            if preferred in options:
                config[key] = preferred
            else:
                config[key] = options[0]
        return config


def _enumerate_priority(
    search_space: dict[str, list], initial: dict
) -> list[dict]:
    """Walk the space by varying one knob at a time around the initial config."""
    out: list[dict] = []
    seen: set[tuple] = set()

    def add(cfg: dict) -> None:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)

    add(initial)
    for k, options in search_space.items():
        for v in options:
            if v == initial.get(k):
                continue
            variant = dict(initial)
            variant[k] = v
            add(variant)
    return out


def _all_configs(search_space: dict[str, list]) -> list[dict]:
    import itertools

    keys = list(search_space.keys())
    values = list(search_space.values())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def _detect_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
    except Exception:
        pass
    return 0.0
