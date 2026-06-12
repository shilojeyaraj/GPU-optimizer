"""
LLM-based recommender. Provider-agnostic (Claude + GPT-4o).

The SYSTEM_PROMPT split into TASK + DOCS sections is the paper's core
ablation: `include_docs=False` runs the LLM-without-docs baseline, isolating
whether the LLM's general reasoning helps or whether the encoded PyTorch
knowledge in the prompt does.
"""
# ruff: noqa: E501  -- prompt strings are intentionally long; breaking them risks subtle text drift.
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from autotune.recommenders.base import Recommender, Suggestion
from autotune.recommenders.random import RandomRecommender

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TASK = """You are an expert PyTorch training engineer specializing in systems-level performance optimization. Your job is to recommend the next configuration to try in a hyperparameter sweep, optimizing for training throughput (samples/second).

You have access to:
1. The current search space (what values are available for each knob)
2. The history of already-tried configurations and their measured throughput
3. Hardware constraints (VRAM budget, GPU type if known)

Rules:
- Only suggest configs within the provided search space
- Never suggest a config identical to one already tried
- Explain your reasoning briefly (1-2 sentences)
- Return valid JSON matching the schema provided"""


SYSTEM_PROMPT_DOCS = """

PyTorch systems optimization knowledge:
- bf16 is preferred over fp16 on A100/H100/RTX 30xx+ for stability and speed
- torch.compile reduce-overhead mode benefits small models; max-autotune benefits large ones
- Flash attention dramatically reduces memory bandwidth for attention layers (transformers only)
- Gradient accumulation trades throughput for effective batch size — only use when batch_size is memory-constrained
- DataLoader workers: 4 is usually optimal; more than 8 rarely helps and can cause CPU contention
- pin_memory=True helps when CPU->GPU transfer is a bottleneck (usually when workers > 0)"""


def build_system_prompt(include_docs: bool = True) -> str:
    return SYSTEM_PROMPT_TASK + (SYSTEM_PROMPT_DOCS if include_docs else "")


def build_user_prompt(
    search_space: dict[str, list],
    history: list[tuple[dict, float]],
    constraints: dict,
) -> str:
    if history:
        history_lines = "\n".join(
            f"  Config: {json.dumps(cfg)} -> Throughput: {tput:.1f} samples/sec"
            for cfg, tput in history
        )
    else:
        history_lines = "  (no trials yet)"

    return f"""Search space: {json.dumps(search_space)}

Constraints: {json.dumps(constraints)}

Trial history ({len(history)} trials so far):
{history_lines}

Suggest the next configuration to maximize throughput.
Respond with valid JSON only:
{{
  "config": {{"batch_size": ..., "precision": ..., ...}},
  "reasoning": "..."
}}"""


class LLMLogger:
    """JSONL log of every prompt/response. Required for paper reproducibility."""

    def __init__(self, log_dir: str | Path = "autotune_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"{self.session_id}.jsonl"

    def log(self, event_type: str, data: dict) -> None:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": event_type,
            **data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


class LLMRecommender(Recommender):
    """
    Recommender backed by Claude or GPT-4o.

    Args:
        provider: "claude" or "gpt-4o"
        model: model identifier; defaults to claude-sonnet-4-6 / gpt-4o
        temperature: sampler temperature (0.0 for best reproducibility)
        include_docs: if False, omit the PyTorch knowledge section from the
            system prompt. This is the paper's central ablation.
        log_dir: directory for prompt/response logs
    """

    def __init__(
        self,
        provider: str = "claude",
        model: str | None = None,
        temperature: float = 0.0,
        include_docs: bool = True,
        log_dir: str | Path = "autotune_logs",
    ):
        self.provider = provider
        self.temperature = temperature
        self.include_docs = include_docs
        self.system_prompt = build_system_prompt(include_docs=include_docs)
        self.logger = LLMLogger(log_dir=log_dir)
        self._fallback = RandomRecommender()
        # Populated by _call_provider on each call; surfaced through the JSONL
        # response log so cost-per-trial is recoverable for the paper.
        self._last_usage: dict[str, int | None] | None = None

        if provider == "claude":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    'install with `pip install "pytorch-autotune[llm]"`'
                ) from exc
            self.client = anthropic.Anthropic()
            self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        elif provider == "gpt-4o":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    'install with `pip install "pytorch-autotune[llm]"`'
                ) from exc
            self.client = OpenAI()
            self.model = model or "gpt-4o"
        else:
            raise ValueError(
                f"unknown provider {provider!r}; use 'claude' or 'gpt-4o'"
            )

    def suggest(
        self,
        search_space: dict[str, list],
        history: list[tuple[dict, float]],
        constraints: dict[str, Any],
    ) -> Suggestion:
        user_prompt = build_user_prompt(search_space, history, constraints)
        prompt_hash = hashlib.sha256(
            (self.system_prompt + user_prompt).encode("utf-8")
        ).hexdigest()[:16]

        self.logger.log(
            "prompt",
            {
                "provider": self.provider,
                "model": self.model,
                "include_docs": self.include_docs,
                "prompt_hash": prompt_hash,
                "system_prompt": self.system_prompt,
                "user_prompt": user_prompt,
            },
        )

        try:
            raw = self._call_provider(user_prompt)
        except Exception as exc:  # noqa: BLE001
            self.logger.log(
                "error", {"prompt_hash": prompt_hash, "error": str(exc)}
            )
            logger.warning("LLM provider call failed; falling back to random: %s", exc)
            return self._fallback.suggest(search_space, history, constraints)

        self.logger.log(
            "response",
            {
                "prompt_hash": prompt_hash,
                "raw_response": raw,
                "usage": self._last_usage,
            },
        )

        parsed = _parse_response(raw)
        if parsed is None or "config" not in parsed:
            self.logger.log(
                "parse_failure", {"prompt_hash": prompt_hash, "raw_response": raw}
            )
            logger.warning("LLM response parse failure; falling back to random")
            return self._fallback.suggest(search_space, history, constraints)

        config = _validate_config(parsed["config"], search_space)
        if config is None:
            self.logger.log(
                "invalid_config",
                {"prompt_hash": prompt_hash, "raw_response": raw, "search_space": search_space},
            )
            logger.warning("LLM proposed out-of-space config; falling back to random")
            return self._fallback.suggest(search_space, history, constraints)

        return Suggestion(config=config, reasoning=parsed.get("reasoning"))

    def _call_provider(self, user_prompt: str) -> str:
        if self.provider == "claude":
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=self.temperature,
                system=[
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            u = getattr(msg, "usage", None)
            self._last_usage = {
                "input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None),
                "cache_creation_input_tokens": getattr(
                    u, "cache_creation_input_tokens", None
                ),
                "cache_read_input_tokens": getattr(
                    u, "cache_read_input_tokens", None
                ),
            }
            return "".join(
                block.text
                for block in msg.content
                if getattr(block, "type", None) == "text"
            )
        else:  # gpt-4o
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            u = getattr(response, "usage", None)
            self._last_usage = {
                "input_tokens": getattr(u, "prompt_tokens", None),
                "output_tokens": getattr(u, "completion_tokens", None),
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
            }
            return response.choices[0].message.content or ""


def _parse_response(raw: str) -> dict | None:
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def _validate_config(config: dict, search_space: dict[str, list]) -> dict | None:
    """Return the config if every value is in the search space, else None."""
    if not isinstance(config, dict):
        return None
    cleaned: dict = {}
    for key, options in search_space.items():
        if key not in config:
            return None
        value = config[key]
        if value not in options:
            return None
        cleaned[key] = value
    return cleaned
