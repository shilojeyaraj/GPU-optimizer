from autotune.sweep import SweepGrid, grid


def test_grid_generates_correct_configs():
    s = grid(batch_size=[16, 32], precision=["fp32", "bf16"])
    configs = s.all_configs()
    assert len(configs) == 4
    assert {"batch_size": 16, "precision": "fp32"} in configs
    assert {"batch_size": 32, "precision": "bf16"} in configs


def test_grid_size_matches_enumeration():
    s = grid(a=[1, 2, 3], b=["x", "y"])
    assert s.size() == 6
    assert len(s.all_configs()) == 6


def test_sample_is_deterministic_with_seed():
    s = grid(a=[1, 2, 3, 4], b=["x", "y", "z"])
    first = s.sample(5, seed=42)
    second = s.sample(5, seed=42)
    assert first == second


def test_sample_caps_at_grid_size():
    s = grid(a=[1, 2], b=["x"])
    assert len(s.sample(100, seed=0)) == 2
