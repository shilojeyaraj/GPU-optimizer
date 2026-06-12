"""
Live LLM smoke test -- one Claude API call per ablation arm.

Run this BEFORE spending any GPU budget. It validates the end-to-end LLM path:

  - ANTHROPIC_API_KEY (or OPENAI_API_KEY) is set, SDK is installed
  - LLMRecommender.suggest() returns a real LLM-grounded config -- not the
    random fallback that fires on API/parse/validation errors
  - Both ablation arms (include_docs True/False) hit distinct system prompts
  - Token usage and cost are recoverable from the JSONL log (a paper metric)

Cost on claude-sonnet-4-6 is ~$0.01-$0.05 for the two calls -- well under the
$100 ceiling. Exit code is 0 only if both arms returned LLM-grounded configs.

Usage:
    cd pytorch-autotune
    pip install -e ".[llm]"
    export ANTHROPIC_API_KEY=...      # or set in PowerShell: $env:ANTHROPIC_API_KEY = "..."
    python experiments/smoke_llm.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from autotune.recommenders.llm import LLMRecommender

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = REPO_ROOT / "experiments" / "results" / "smoke_logs"

# Per-million-token pricing, claude-sonnet-4-6, as of 2026-05. Sources can drift;
# the JSONL log keeps raw token counts so cost can be recomputed at any time.
PRICE_INPUT_PER_M = 3.00
PRICE_OUTPUT_PER_M = 15.00
PRICE_CACHE_WRITE_PER_M = 3.75
PRICE_CACHE_READ_PER_M = 0.30

SMOKE_SEARCH_SPACE: dict[str, list] = {
    "batch_size": [8, 16, 32, 64, 128],
    "precision": ["fp32", "fp16", "bf16"],
    "compile_mode": ["default", "reduce-overhead", "max-autotune"],
    "use_flash_attention": [True, False],
    "gradient_accumulation_steps": [1, 2, 4],
    "dataloader_workers": [0, 2, 4, 8],
}
SMOKE_CONSTRAINTS = {"max_vram_gb": 24}


def estimate_cost_usd(usage: dict | None) -> float:
    if not usage:
        return 0.0
    input_t = usage.get("input_tokens") or 0
    output_t = usage.get("output_tokens") or 0
    cache_create = usage.get("cache_creation_input_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    return (
        input_t * PRICE_INPUT_PER_M
        + output_t * PRICE_OUTPUT_PER_M
        + cache_create * PRICE_CACHE_WRITE_PER_M
        + cache_read * PRICE_CACHE_READ_PER_M
    ) / 1_000_000


def _read_session_events(log_dir: Path) -> list[dict]:
    """Read all JSONL entries from the most recent session file in this dir."""
    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        return []
    lines = files[-1].read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_arm(
    provider: str, model: str | None, include_docs: bool, log_dir: Path
) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    rec = LLMRecommender(
        provider=provider, model=model, include_docs=include_docs, log_dir=log_dir
    )

    t0 = time.perf_counter()
    suggestion = rec.suggest(
        search_space=SMOKE_SEARCH_SPACE, history=[], constraints=SMOKE_CONSTRAINTS
    )
    latency_s = time.perf_counter() - t0

    events = _read_session_events(log_dir)
    response_events = [e for e in events if e.get("event") == "response"]
    fallback_events = [
        e
        for e in events
        if e.get("event") in {"error", "parse_failure", "invalid_config"}
    ]
    usage = response_events[-1].get("usage") if response_events else None

    return {
        "include_docs": include_docs,
        "model": rec.model,
        "config": suggestion.config,
        "reasoning": suggestion.reasoning,
        "usage": usage,
        "cost_usd": estimate_cost_usd(usage),
        "latency_s": latency_s,
        "fell_back": bool(fallback_events),
        "log_file": str(sorted(log_dir.glob("*.jsonl"))[-1]) if log_dir.glob("*.jsonl") else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="claude", choices=["claude", "gpt-4o"])
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model id; defaults to the provider's default",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help=f"Where to write JSONL session logs. Default: {DEFAULT_LOG_DIR}",
    )
    args = parser.parse_args()

    key_env = "ANTHROPIC_API_KEY" if args.provider == "claude" else "OPENAI_API_KEY"
    if not os.getenv(key_env):
        print(f"error: {key_env} not set in environment", file=sys.stderr)
        return 2

    log_root = Path(args.log_dir)
    n_cells = math.prod(len(v) for v in SMOKE_SEARCH_SPACE.values())

    print(f"=== Live LLM smoke ({args.provider}) ===")
    print(f"search space: {len(SMOKE_SEARCH_SPACE)} dims, {n_cells} cells")
    print(f"logging to:   {log_root}")
    print()

    arms = []
    for include_docs in (True, False):
        label = "with_docs" if include_docs else "no_docs"
        print(f"--- arm: {label} ---")
        arm = run_arm(args.provider, args.model, include_docs, log_root / label)
        arms.append((label, arm))
        print(f"  model:     {arm['model']}")
        print(f"  config:    {arm['config']}")
        print(f"  reasoning: {arm['reasoning']}")
        print(f"  usage:     {arm['usage']}")
        print(f"  cost:      ${arm['cost_usd']:.5f}")
        print(f"  latency:   {arm['latency_s']:.2f}s")
        print(f"  fell back: {arm['fell_back']}")
        print(f"  log:       {arm['log_file']}")
        print()

    total_cost = sum(a["cost_usd"] for _, a in arms)
    any_fallback = any(a["fell_back"] for _, a in arms)
    print("=== summary ===")
    print(f"total cost:   ${total_cost:.5f}")
    print(f"any fallback: {any_fallback}")

    if any_fallback:
        print(
            "\nFAIL -- at least one arm fell back to random. Inspect the JSONL logs.",
            file=sys.stderr,
        )
        return 1
    if arms[0][1]["config"] == arms[1][1]["config"]:
        print(
            "\nWARN -- both arms returned the identical config. Possible at "
            "temperature=0, but eyeball the two reasonings to confirm the docs "
            "section is actually being used."
        )
    print("\nOK -- both arms returned LLM-grounded configs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
