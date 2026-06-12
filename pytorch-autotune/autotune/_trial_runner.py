"""
Subprocess entrypoint for a single trial.

Invoked by autotune.backends.local.run_trial as:
    python -m autotune._trial_runner <module_path> <fn_name> <config_json_path>

Imports the user's training function dynamically, calls it with the config,
and writes a JSON result blob to stdout. Any exception propagates as a
non-zero exit code with the traceback on stderr — the parent process
classifies it (OOM, compile failure, generic error).
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import traceback
from pathlib import Path

from autotune.profile import peak_memory_mb, reset_peak_memory


def _load_train_fn(module_path: str, fn_name: str):
    module = importlib.import_module(module_path)
    if not hasattr(module, fn_name):
        raise AttributeError(
            f"module '{module_path}' has no attribute '{fn_name}'"
        )
    return getattr(module, fn_name)


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: python -m autotune._trial_runner <module_path> <fn_name> <config_json>",
            file=sys.stderr,
        )
        return 2

    module_path, fn_name, config_path_str = sys.argv[1:]
    config = json.loads(Path(config_path_str).read_text(encoding="utf-8"))

    try:
        train_fn = _load_train_fn(module_path, fn_name)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1

    reset_peak_memory()
    start = time.perf_counter()
    try:
        throughput = train_fn(config)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    duration = time.perf_counter() - start

    payload = {
        "throughput": float(throughput) if throughput is not None else None,
        "duration_seconds": duration,
        "memory_mb": peak_memory_mb(),
        "extra": {},
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
