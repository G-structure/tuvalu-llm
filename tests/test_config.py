"""Test config loading and validation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tv.common.config import load_config, merge_config, get_repo_root
from scripts.build_stage_a_mt_data import flatten_build_config
from scripts.eval_stage_a_translation import flatten_eval_config
from scripts.train_stage_a_translation import flatten_train_config


def test_load_config():
    """Test loading a JSON config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"key": "value", "nested": {"a": 1}}, f)
        f.flush()
        config = load_config(f.name)
    assert config["key"] == "value"
    assert config["nested"]["a"] == 1


def test_load_config_not_found():
    """Test that missing config raises FileNotFoundError."""
    try:
        load_config("/nonexistent/path/config.json")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass


def test_merge_config():
    """Test deep merging of configs."""
    base = {"a": 1, "nested": {"x": 10, "y": 20}, "b": 2}
    overrides = {"a": 99, "nested": {"x": 99}}
    result = merge_config(base, overrides)
    assert result["a"] == 99
    assert result["nested"]["x"] == 99
    assert result["nested"]["y"] == 20
    assert result["b"] == 2


def test_merge_config_new_keys():
    """Test merging adds new keys."""
    base = {"a": 1}
    overrides = {"b": 2, "c": {"d": 3}}
    result = merge_config(base, overrides)
    assert result == {"a": 1, "b": 2, "c": {"d": 3}}


def test_get_repo_root():
    """Test repo root detection."""
    root = get_repo_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_all_stage_configs_valid():
    """Test that all stage config files parse as valid JSON."""
    configs_dir = get_repo_root() / "configs"
    for config_path in configs_dir.glob("*.json"):
        config = load_config(config_path)
        assert isinstance(config, dict), f"{config_path.name} not a dict"


# --- Task 1 tests: wrapper config flattening ---

SAMPLE_NESTED_CONFIG = {
    "stage": "stage_a_translation",
    "model": {"name": "Qwen/Qwen3-30B-A3B-Base", "type": "base"},
    "training": {
        "mode": "lora", "lora_rank": 32, "max_length": 2048,
        "batch_size": 64, "learning_rate": 2e-4, "epochs": 2,
        "save_every": 100, "seed": 17,
        "train_on_what": "ALL_ASSISTANT_MESSAGES", "ttl_seconds": 604800,
    },
    "data": {
        "input_dir": "data/aligned",
        "output_dir": "data/finetune/stage_a_mt",
        "min_confidence": 0.8, "min_chars": 10, "max_chars": 4096,
        "ratio_min": 0.4, "ratio_max": 2.5,
        "bible_max_train_share": 0.7,
        "non_bible_val_frac": 0.05, "non_bible_test_frac": 0.05,
        "validation_books": [31, 63, 64], "test_books": [8, 57, 65],
        "train_file": "train_balanced.jsonl",
    },
    "eval": {
        "max_tokens": 512, "temperature": 0.0,
        "test_file": "test.jsonl",
        "out_dir": "logs/tinker/evals/stage_a",
    },
    "logs": {"base_dir": "logs/tinker/stage_a"},
}


def test_build_wrapper_flattens_nested_config():
    """Verify build wrapper correctly flattens nested config."""
    flat = flatten_build_config(SAMPLE_NESTED_CONFIG)
    assert flat["input_dir"] == "data/aligned"
    assert flat["output_dir"] == "data/finetune/stage_a_mt"
    assert flat["min_confidence"] == 0.8
    assert flat["min_chars"] == 10
    assert flat["max_chars"] == 4096
    assert flat["ratio_min"] == 0.4
    assert flat["ratio_max"] == 2.5
    assert flat["bible_max_train_share"] == 0.7
    assert flat["non_bible_val_frac"] == 0.05
    assert flat["non_bible_test_frac"] == 0.05
    assert flat["validation_books"] == [31, 63, 64]
    assert flat["test_books"] == [8, 57, 65]
    assert flat["seed"] == 17
    # Should NOT contain nested sections
    assert "data" not in flat
    assert "training" not in flat
    assert "model" not in flat


def test_eval_wrapper_flattens_nested_config():
    """Verify eval wrapper correctly flattens nested config."""
    flat = flatten_eval_config(SAMPLE_NESTED_CONFIG)
    assert flat["data"] == "data/finetune/stage_a_mt/test.jsonl"
    assert flat["model_name"] == "Qwen/Qwen3-30B-A3B-Base"
    assert flat["max_tokens"] == 512
    assert flat["temperature"] == 0.0
    assert flat["out_dir"] == "logs/tinker/evals/stage_a"


def test_config_driven_book_holdouts():
    """Verify build_data uses config books when provided."""
    from tv.training.stage_a_mt.build_data import _assign_split

    bible_row = {"content_type": "bible_verse", "book_num": 1}

    # Default: book 1 is train
    assert _assign_split(bible_row, non_bible_val_frac=0.05, non_bible_test_frac=0.05) == "train"

    # Custom: book 1 as test
    assert _assign_split(
        bible_row, non_bible_val_frac=0.05, non_bible_test_frac=0.05,
        test_books={1}, validation_books={2},
    ) == "test"

    # Custom: book 1 as validation
    assert _assign_split(
        bible_row, non_bible_val_frac=0.05, non_bible_test_frac=0.05,
        test_books={99}, validation_books={1},
    ) == "validation"


def test_pilot_config_selects_pilot_model():
    """Verify pilot config loads correct model."""
    root = get_repo_root()
    pilot_path = root / "configs" / "stage_a_translation_qwen30b_base_pilot_2m_1epoch.json"
    config = load_config(pilot_path)
    assert config["model"]["name"] == "Qwen/Qwen3-30B-A3B-Base"
    assert config["data"]["pilot_token_budget"] == 2000000
    assert config["data"]["train_file"] == "train_pilot_2m.jsonl"
    assert config["training"]["epochs"] == 1


def test_stage_b_qwen30b_config_valid():
    """Stage B Qwen3 config loads and has correct model."""
    root = get_repo_root()
    config_path = root / "configs" / "stage_b_agent_qwen30b.json"
    config = load_config(config_path)
    assert config["model"]["name"] == "Qwen/Qwen3-30B-A3B"
    assert config["stage"] == "stage_b_agent"
    assert config["training"]["train_on_what"] == "ALL_ASSISTANT_MESSAGES"
    assert config["training"]["lora_rank"] == 32


# --- Task 2 tests: pilot subset ---

def _make_fake_examples(n: int, chars_per_msg: int = 200) -> list[dict]:
    """Create fake chat examples for pilot subset testing."""
    examples = []
    for i in range(n):
        examples.append({
            "id": f"test_pair_{i:04d}::tvl_to_en",
            "messages": [
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": "x" * chars_per_msg},
                {"role": "assistant", "content": "y" * chars_per_msg},
            ],
            "metadata": {"direction": "tvl_to_en", "content_type": "bible_verse"},
        })
    return examples


def test_pilot_subset_deterministic():
    """Two runs with same config produce same subset."""
    from tv.training.stage_a_mt.build_data import build_pilot_subset

    examples = _make_fake_examples(100)
    run1 = build_pilot_subset(examples, token_budget=5000)
    run2 = build_pilot_subset(examples, token_budget=5000)
    assert [e["id"] for e in run1] == [e["id"] for e in run2]


def test_pilot_subset_within_budget():
    """Total tokens within 10% of budget."""
    from tv.training.stage_a_mt.build_data import build_pilot_subset
    from tv.common.token_estimates import estimate_dataset_tokens

    examples = _make_fake_examples(200, chars_per_msg=100)
    budget = 5000
    subset = build_pilot_subset(examples, token_budget=budget)
    total = estimate_dataset_tokens(subset)
    # Should be at most budget + one example's worth over
    assert total <= budget * 1.1, f"Total {total} exceeds 110% of budget {budget}"
    # Should use a reasonable fraction of the budget
    assert total > budget * 0.5, f"Total {total} is less than 50% of budget {budget}"


def test_pilot_subset_stable_ordering():
    """Examples are selected in stable hash order."""
    from tv.training.stage_a_mt.build_data import build_pilot_subset, _stable_hash

    examples = _make_fake_examples(50)
    subset = build_pilot_subset(examples, token_budget=3000)
    # Verify subset is in stable hash order
    hashes = [_stable_hash(e["id"]) for e in subset]
    assert hashes == sorted(hashes)
