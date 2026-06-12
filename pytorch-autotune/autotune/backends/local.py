"""
Local subprocess backend. Each trial runs in a fresh Python process so that
OOM, segfault, or torch.compile error in one trial can't kill the sweep.

OOM detection works by stderr-scraping for "CUDA out of memory" — the OOM rate
metric is part of the paper's reporting, not just an error handler.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrialResult:
    config: dict
    throughput: float | None
    error: str | None = None
    oom: bool = False
    compile_failure: bool = False
    timeout: bool = False
    duration_seconds: float | None = None
    memory_mb: float | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "throughput_samples_per_sec": self.throughput,
            "error": self.error,
            "oom": self.oom,
            "compile_failure": self.compile_failure,
            "timeout": self.timeout,
            "trial_duration_seconds": self.duration_seconds,
            "memory_mb": self.memory_mb,
            "extra": self.extra or {},
        }


def run_trial(
    train_fn_module_path: str,
    train_fn_name: str,
    config: dict,
    timeout_seconds: int = 3600,
) -> TrialResult:
    """
    Run a single trial in a fresh subprocess.

    Args:
        train_fn_module_path: importable module path containing the train fn
            (e.g. "experiments.workloads.resnet50")
        train_fn_name: function name inside that module (e.g. "train")
        config: config dict passed to train_fn
        timeout_seconds: kill the subprocess after this many seconds

    The subprocess writes a JSON result blob to stdout and exits 0 on success.
    Any non-zero exit is classified as oom / compile_failure / generic error
    based on stderr content.
    """
    config_path = _write_temp_json(config)

    # Propagate parent sys.path so the subprocess can import the user's
    # train_fn from wherever it lives (pytest-collected modules, editable
    # installs, scripts run from arbitrary dirs).
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "autotune._trial_runner",
                train_fn_module_path,
                train_fn_name,
                str(config_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return TrialResult(
            config=config,
            throughput=None,
            error="timeout",
            timeout=True,
        )
    finally:
        try:
            config_path.unlink()
        except OSError:
            pass

    if result.returncode != 0:
        stderr_tail = result.stderr[-4000:] if result.stderr else ""
        is_oom = "CUDA out of memory" in stderr_tail or "OutOfMemoryError" in stderr_tail
        is_compile_failure = (
            "BackendCompilerFailed" in stderr_tail
            or "torch._dynamo.exc" in stderr_tail
        )
        return TrialResult(
            config=config,
            throughput=None,
            error=stderr_tail,
            oom=is_oom,
            compile_failure=is_compile_failure,
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return TrialResult(
            config=config,
            throughput=None,
            error=f"trial_runner did not return valid JSON: {exc}",
        )

    return TrialResult(
        config=config,
        throughput=payload.get("throughput"),
        duration_seconds=payload.get("duration_seconds"),
        memory_mb=payload.get("memory_mb"),
        extra=payload.get("extra"),
    )


def _write_temp_json(config: dict) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(config, f)
        return Path(f.name)
