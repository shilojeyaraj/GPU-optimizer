"""Sweep grid definitions."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass


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

    def size(self) -> int:
        size = 1
        for v in self.params.values():
            size *= len(v)
        return size


def grid(**kwargs) -> SweepGrid:
    return SweepGrid(params=kwargs)
