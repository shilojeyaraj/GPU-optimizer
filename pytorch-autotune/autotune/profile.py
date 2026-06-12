"""
GPU profiling and VRAM estimation utilities.

Carved from the old ml/profiler_utils.py and ml/cuda_utils.py. Self-contained:
no Celery, FastAPI, or pybind11 dependencies. Falls back gracefully when CUDA
or pynvml are unavailable.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


_PRECISION_DTYPES = {
    "fp32": None,
    "fp16": "float16",
    "bf16": "bfloat16",
}

_VALID_COMPILE_MODES = {None, "default", "reduce-overhead", "max-autotune"}


# ───────────────────────── VRAM estimation ─────────────────────────
#
# Conservative lower-bound estimates from MLPerf / NVIDIA tuning guides and
# empirical measurement. Used to pre-filter obviously-infeasible configs
# before a trial runs. When the estimate is unavailable, we let the trial
# run and rely on OOM detection in the backend — better to OOM and log
# than to silently skip a config.

VRAM_ESTIMATES_GB: dict[tuple[str, int, str], float] = {
    # ResNet-50 (25M params)
    ("resnet50", 16, "fp32"): 2.5,
    ("resnet50", 32, "fp32"): 4.0,
    ("resnet50", 64, "fp32"): 7.0,
    ("resnet50", 128, "fp32"): 13.0,
    ("resnet50", 256, "fp32"): 24.0,
    ("resnet50", 32, "bf16"): 2.5,
    ("resnet50", 64, "bf16"): 4.0,
    ("resnet50", 128, "bf16"): 7.0,
    ("resnet50", 256, "bf16"): 13.0,
    # BERT-base (110M params)
    ("bert_base", 8, "fp32"): 4.0,
    ("bert_base", 16, "fp32"): 6.0,
    ("bert_base", 32, "fp32"): 10.0,
    ("bert_base", 16, "bf16"): 3.5,
    ("bert_base", 32, "bf16"): 6.0,
    ("bert_base", 64, "bf16"): 10.0,
    # LLaMA-7B LoRA (frozen base + small adapters)
    ("llama_7b", 1, "bf16"): 16.0,
    ("llama_7b", 2, "bf16"): 20.0,
    ("llama_7b", 4, "bf16"): 28.0,
    ("llama_7b", 8, "bf16"): 44.0,
}


def estimate_vram_gb(model_family: str, config: dict) -> float | None:
    """
    Lower-bound VRAM estimate for (model_family, batch_size, precision).
    Returns None when no estimate exists — caller should let the trial run anyway.
    """
    key = (model_family, config.get("batch_size"), config.get("precision"))
    return VRAM_ESTIMATES_GB.get(key)


def fits_in_budget(model_family: str, config: dict, max_vram_gb: float) -> bool:
    """
    True when we have an estimate AND it fits; True when we have no estimate
    (don't over-filter). Only returns False when we have an estimate that
    exceeds the budget.
    """
    est = estimate_vram_gb(model_family, config)
    if est is None:
        return True
    return est <= max_vram_gb


# ───────────────────────── GPU memory helpers ─────────────────────────


def get_memory_snapshot(gpu_index: int = 0) -> tuple[int, int] | None:
    """Returns (allocated_bytes, reserved_minus_allocated_bytes) or None if no CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.set_device(gpu_index)
        allocated = torch.cuda.memory_allocated(gpu_index)
        reserved = torch.cuda.memory_reserved(gpu_index)
        return (int(allocated), int(reserved - allocated))
    except Exception:
        return None


def get_gpu_info(gpu_index: int = 0) -> dict[str, Any]:
    """
    GPU model + driver + CUDA info for the experiment log. Best-effort.
    """
    info: dict[str, Any] = {
        "gpu_model": None,
        "driver_version": None,
        "cuda_version": None,
        "torch_version": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_model"] = torch.cuda.get_device_name(gpu_index)
    except ImportError:
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        info["gpu_model"] = info["gpu_model"] or pynvml.nvmlDeviceGetName(handle)
    except Exception:
        pass

    return info


# ───────────────────────── Trial timing helpers ─────────────────────────


def cuda_synchronize() -> None:
    """No-op when CUDA isn't available. Use before/after measurement windows."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def reset_peak_memory(gpu_index: int = 0) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(gpu_index)
    except ImportError:
        pass


def peak_memory_mb(gpu_index: int = 0) -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(gpu_index) / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


# ───────────────────────── Validation ─────────────────────────


def validate_config(config: dict) -> None:
    """Raise ValueError for any config value out of contract. Pure, no I/O."""
    precision = config.get("precision")
    if precision is not None and precision not in _PRECISION_DTYPES:
        raise ValueError(
            f"precision must be one of {sorted(_PRECISION_DTYPES)}, got {precision!r}"
        )
    compile_mode = config.get("compile_mode")
    if compile_mode is not None and compile_mode not in _VALID_COMPILE_MODES:
        raise ValueError(
            f"compile_mode must be one of {sorted(m for m in _VALID_COMPILE_MODES if m)}, "
            f"got {compile_mode!r}"
        )
