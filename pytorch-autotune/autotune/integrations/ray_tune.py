"""
Ray Tune integration: LLMSearcher. Planned for v0.2.

The pattern will mirror autotune.integrations.optuna.LLMSampler — subclass
ray.tune.search.Searcher and route suggest() through a Recommender. Stubbed
now so the import surface is documented.
"""
from __future__ import annotations


class LLMSearcher:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Ray Tune integration ships in v0.2. "
            "Contributions welcome — see issue tracker."
        )
