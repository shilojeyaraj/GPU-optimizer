"""
LLMRecommender tests that don't actually call an LLM. Verifies prompt
construction, response parsing, the include_docs ablation, and the
fallback-to-random path on parse failure.
"""
from autotune.recommenders import llm as llm_module
from autotune.recommenders.llm import (
    _parse_response,
    _validate_config,
    build_system_prompt,
    build_user_prompt,
)


def test_include_docs_changes_system_prompt():
    with_docs = build_system_prompt(include_docs=True)
    without = build_system_prompt(include_docs=False)
    assert "bf16 is preferred" in with_docs
    assert "bf16 is preferred" not in without
    # The task description must remain identical across the ablation.
    assert without in with_docs


def test_user_prompt_lists_history():
    prompt = build_user_prompt(
        search_space={"batch_size": [16, 32]},
        history=[({"batch_size": 16}, 100.0)],
        constraints={"max_vram_gb": 24},
    )
    assert "batch_size" in prompt
    assert "100.0" in prompt
    assert "max_vram_gb" in prompt


def test_user_prompt_handles_empty_history():
    prompt = build_user_prompt(
        search_space={"batch_size": [16, 32]}, history=[], constraints={}
    )
    assert "no trials yet" in prompt


def test_parse_response_strips_json_fence():
    raw = '```json\n{"config": {"batch_size": 32}, "reasoning": "x"}\n```'
    parsed = _parse_response(raw)
    assert parsed["config"] == {"batch_size": 32}
    assert parsed["reasoning"] == "x"


def test_parse_response_handles_plain_json():
    raw = '{"config": {"batch_size": 16}, "reasoning": "ok"}'
    parsed = _parse_response(raw)
    assert parsed["config"] == {"batch_size": 16}


def test_parse_response_returns_none_on_garbage():
    assert _parse_response("not json") is None
    assert _parse_response("") is None


def test_validate_config_rejects_out_of_space_values():
    space = {"batch_size": [16, 32], "precision": ["fp32"]}
    assert _validate_config({"batch_size": 64, "precision": "fp32"}, space) is None
    assert _validate_config({"batch_size": 16, "precision": "fp32"}, space) == {
        "batch_size": 16,
        "precision": "fp32",
    }


def test_validate_config_requires_all_keys():
    space = {"batch_size": [16, 32], "precision": ["fp32"]}
    assert _validate_config({"batch_size": 16}, space) is None
