"""
Recommender protocol — the contract every recommender (random, heuristic, LLM)
implements. Plugin point for the entire library.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Suggestion:
    config: dict[str, Any]
    reasoning: str | None = None
    confidence: float | None = None


class Recommender(Protocol):
    def suggest(
        self,
        search_space: dict[str, list],
        history: list[tuple[dict, float]],
        constraints: dict[str, Any],
    ) -> Suggestion: ...
