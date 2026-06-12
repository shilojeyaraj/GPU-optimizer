"""Random sampler. Paper baseline."""
from __future__ import annotations

import random
from typing import Any

from autotune.recommenders.base import Recommender, Suggestion


class RandomRecommender(Recommender):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def suggest(
        self,
        search_space: dict[str, list],
        history: list[tuple[dict, float]],
        constraints: dict[str, Any],
    ) -> Suggestion:
        tried = {tuple(sorted(cfg.items())) for cfg, _ in history}
        for _ in range(100):
            config = {k: self.rng.choice(v) for k, v in search_space.items()}
            if tuple(sorted(config.items())) not in tried:
                return Suggestion(config=config, reasoning="random sample")
        # All combos seen — return any random one and let the runner dedupe upstream.
        config = {k: self.rng.choice(v) for k, v in search_space.items()}
        return Suggestion(config=config, reasoning="random (search space exhausted)")
