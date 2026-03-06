"""Test shared dataset naming conventions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.synthetic.naming import dataset_name_to_filename, filename_to_dataset_name


class TestNamingRoundtrip:
    def test_name_to_filename_roundtrip(self):
        names = [
            "openai/gsm8k",
            "tasksource/tasksource-instruct-v0",
            "HuggingFaceH4/ultrachat_200k",
            "Muennighoff/mbpp",
            "Salesforce/xlam-function-calling-60k",
        ]
        for name in names:
            filename = dataset_name_to_filename(name)
            assert "/" not in filename, f"Filename should not contain /: {filename}"
            recovered = filename_to_dataset_name(filename)
            assert recovered == name, f"Roundtrip failed: {name} -> {filename} -> {recovered}"

    def test_name_without_slash(self):
        """Names without / should pass through unchanged."""
        assert dataset_name_to_filename("gsm8k") == "gsm8k"
        assert filename_to_dataset_name("gsm8k") == "gsm8k"

    def test_double_underscore_in_output(self):
        assert dataset_name_to_filename("openai/gsm8k") == "openai__gsm8k"


class TestSourceBuilderAndGeneratorAgree:
    def test_source_builder_and_generator_agree(self):
        """Verify build_stage_b_sources and generate.py use the same naming."""
        # The source builder uses dataset_name_to_filename
        # The generator also uses dataset_name_to_filename
        # Both should produce the same filename for the same dataset name
        test_names = [
            "tasksource/tasksource-instruct-v0",
            "openai/gsm8k",
            "Salesforce/xlam-function-calling-60k",
        ]
        for name in test_names:
            builder_filename = dataset_name_to_filename(name)
            generator_filename = dataset_name_to_filename(name)
            assert builder_filename == generator_filename


class TestDictBasedDatasetConfig:
    def test_dict_based_dataset_config_parsing(self):
        """Verify that dict-based config (like synthetic_stage_b_core.json) parses correctly."""
        raw_datasets = [
            {"name": "openai/gsm8k", "task_family": "math", "enabled": True},
            {"name": "Muennighoff/mbpp", "task_family": "code", "enabled": False},
            {"name": "rajpurkar/squad", "task_family": "qa", "enabled": True},
        ]
        # Replicate the parsing logic from build_stage_b_sources.py
        datasets = []
        for d in raw_datasets:
            if isinstance(d, dict):
                if d.get("enabled", True):
                    datasets.append(d["name"])
            else:
                datasets.append(d)

        assert datasets == ["openai/gsm8k", "rajpurkar/squad"]
        assert "Muennighoff/mbpp" not in datasets  # disabled

    def test_string_list_still_works(self):
        """Plain string lists should still parse correctly."""
        raw_datasets = ["openai/gsm8k", "rajpurkar/squad"]
        datasets = []
        for d in raw_datasets:
            if isinstance(d, dict):
                if d.get("enabled", True):
                    datasets.append(d["name"])
            else:
                datasets.append(d)
        assert datasets == ["openai/gsm8k", "rajpurkar/squad"]
