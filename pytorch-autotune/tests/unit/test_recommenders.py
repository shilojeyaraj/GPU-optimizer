from autotune.recommenders.heuristic import HeuristicRecommender
from autotune.recommenders.random import RandomRecommender


SEARCH_SPACE = {
    "batch_size": [16, 32, 64],
    "precision": ["fp32", "bf16"],
    "compile_mode": ["default", "reduce-overhead"],
}


def test_random_recommender_returns_valid_config():
    r = RandomRecommender(seed=0)
    s = r.suggest(SEARCH_SPACE, history=[], constraints={})
    assert s.config["batch_size"] in SEARCH_SPACE["batch_size"]
    assert s.config["precision"] in SEARCH_SPACE["precision"]
    assert s.config["compile_mode"] in SEARCH_SPACE["compile_mode"]


def test_random_recommender_avoids_already_tried():
    r = RandomRecommender(seed=0)
    tried_cfg = {"batch_size": 16, "precision": "fp32", "compile_mode": "default"}
    s = r.suggest(SEARCH_SPACE, history=[(tried_cfg, 100.0)], constraints={})
    assert s.config != tried_cfg


def test_heuristic_first_suggestion_is_in_space():
    r = HeuristicRecommender(model_family="resnet50", vram_gb=24)
    s = r.suggest(SEARCH_SPACE, history=[], constraints={})
    for key, options in SEARCH_SPACE.items():
        assert s.config[key] in options


def test_heuristic_advances_after_history():
    r = HeuristicRecommender(model_family="resnet50", vram_gb=24)
    first = r.suggest(SEARCH_SPACE, history=[], constraints={}).config
    second = r.suggest(
        SEARCH_SPACE, history=[(first, 100.0)], constraints={}
    ).config
    assert second != first
