#!/usr/bin/env python3
"""
Unit tests for download_models._verify_model_files
"""
import json
import os
import tempfile

from download_models import _verify_model_files


def _create_dirs(base, subdirs):
    """Helper to create subdirectories."""
    for subdir in subdirs:
        os.makedirs(os.path.join(base, subdir), exist_ok=True)


REQUIRED_SUBDIRS = ["transformer", "vae", "text_encoder", "scheduler", "tokenizer"]


class TestVerifyModelFiles:
    """Tests for _verify_model_files"""

    def test_returns_false_when_directory_missing_subdirs(self, tmp_path):
        wan_dir = str(tmp_path / "model")
        os.makedirs(wan_dir)
        # Only create some subdirs, missing transformer
        _create_dirs(wan_dir, ["vae", "text_encoder", "scheduler", "tokenizer"])
        assert _verify_model_files(wan_dir) is False

    def test_returns_false_when_transformer_empty(self, tmp_path):
        wan_dir = str(tmp_path / "model")
        os.makedirs(wan_dir)
        _create_dirs(wan_dir, REQUIRED_SUBDIRS)
        # transformer dir exists but no model file
        assert _verify_model_files(wan_dir) is False

    def test_returns_true_with_single_model_file(self, tmp_path):
        wan_dir = str(tmp_path / "model")
        os.makedirs(wan_dir)
        _create_dirs(wan_dir, REQUIRED_SUBDIRS)
        # Create single model file (no index)
        model_file = os.path.join(wan_dir, "transformer", "diffusion_pytorch_model.safetensors")
        with open(model_file, "w") as f:
            f.write("dummy")
        assert _verify_model_files(wan_dir) is True

    def test_returns_true_with_index_and_all_shards(self, tmp_path):
        wan_dir = str(tmp_path / "model")
        os.makedirs(wan_dir)
        _create_dirs(wan_dir, REQUIRED_SUBDIRS)
        transformer_dir = os.path.join(wan_dir, "transformer")
        # Create index file referencing two shards
        index_data = {
            "weight_map": {
                "layer.0.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
                "layer.1.weight": "diffusion_pytorch_model-00002-of-00002.safetensors",
            }
        }
        with open(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f)
        # Create both shard files
        for shard in ["diffusion_pytorch_model-00001-of-00002.safetensors",
                       "diffusion_pytorch_model-00002-of-00002.safetensors"]:
            with open(os.path.join(transformer_dir, shard), "w") as f:
                f.write("dummy")
        assert _verify_model_files(wan_dir) is True

    def test_returns_false_with_index_but_missing_shard(self, tmp_path):
        wan_dir = str(tmp_path / "model")
        os.makedirs(wan_dir)
        _create_dirs(wan_dir, REQUIRED_SUBDIRS)
        transformer_dir = os.path.join(wan_dir, "transformer")
        # Create index file referencing two shards
        index_data = {
            "weight_map": {
                "layer.0.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
                "layer.1.weight": "diffusion_pytorch_model-00002-of-00002.safetensors",
            }
        }
        with open(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f)
        # Create only the first shard (missing the second)
        with open(os.path.join(transformer_dir, "diffusion_pytorch_model-00001-of-00002.safetensors"), "w") as f:
            f.write("dummy")
        assert _verify_model_files(wan_dir) is False

    def test_returns_false_with_corrupt_index_file(self, tmp_path):
        wan_dir = str(tmp_path / "model")
        os.makedirs(wan_dir)
        _create_dirs(wan_dir, REQUIRED_SUBDIRS)
        transformer_dir = os.path.join(wan_dir, "transformer")
        # Create a corrupt index file
        with open(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
            f.write("not valid json")
        assert _verify_model_files(wan_dir) is False

    def test_returns_false_when_each_subdir_missing(self, tmp_path):
        """Test each required subdirectory being missing individually."""
        for missing_subdir in REQUIRED_SUBDIRS:
            wan_dir = str(tmp_path / f"model_{missing_subdir}")
            os.makedirs(wan_dir)
            present = [s for s in REQUIRED_SUBDIRS if s != missing_subdir]
            _create_dirs(wan_dir, present)
            assert _verify_model_files(wan_dir) is False, f"Should fail when {missing_subdir}/ is missing"
