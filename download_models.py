#!/usr/bin/env python3
"""
Pre-download Wan2.1-T2V-1.3B model for faster startup.
Downloads the model from HuggingFace to the models directory.
"""
import json
import os
import sys
import time
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_DIR_NAME = "Wan2.1-T2V-1.3B-Diffusers"


def _format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string"""
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def _get_dir_size(path: str) -> int:
    """Get total size of a directory in bytes"""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def _log_disk_space(models_dir: str):
    """Log available disk space"""
    try:
        usage = shutil.disk_usage(models_dir)
        logger.info(
            f"  Disk space: {_format_size(usage.free)} free "
            f"/ {_format_size(usage.total)} total "
            f"({usage.free * 100 / usage.total:.0f}% free)"
        )
    except Exception:
        pass


def _verify_model_files(wan_dir: str) -> bool:
    """Verify that critical model files exist in the model directory.

    Checks for the required subdirectories and key files that the
    diffusers pipeline needs to load the Wan2.1-T2V-1.3B model.

    Returns True if all critical files are present, False otherwise.
    """
    required_subdirs = ["transformer", "vae", "text_encoder", "scheduler", "tokenizer"]
    for subdir in required_subdirs:
        subdir_path = os.path.join(wan_dir, subdir)
        if not os.path.isdir(subdir_path):
            logger.warning(f"  Missing required subdirectory: {subdir}/")
            return False

    # Check that the transformer directory contains the model index file
    # and at least one model shard
    transformer_dir = os.path.join(wan_dir, "transformer")
    index_file = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_file):
        # If an index file exists, verify that referenced shard files are present
        try:
            with open(index_file, "r") as f:
                index_data = json.load(f)
            shard_files = set(index_data.get("weight_map", {}).values())
            for shard_file in shard_files:
                shard_path = os.path.join(transformer_dir, shard_file)
                if not os.path.isfile(shard_path):
                    logger.warning(f"  Missing transformer shard file: {shard_file}")
                    return False
        except Exception as e:
            logger.warning(f"  Failed to verify transformer index: {e}")
            return False
    else:
        # No index file — check for a single model file instead
        single_model = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.isfile(single_model):
            logger.warning(f"  Missing transformer model file(s) in {transformer_dir}/")
            return False

    return True


def download_models():
    """Download the Wan2.1-T2V-1.3B model"""
    from huggingface_hub import snapshot_download

    models_dir = os.getenv("MODELS_DIR", "/workspace/models")
    os.makedirs(models_dir, exist_ok=True)

    total_start = time.time()

    logger.info("=" * 60)
    logger.info("  Wan2.1-T2V-1.3B Model Download")
    logger.info("=" * 60)
    logger.info(f"  Models directory: {models_dir}")
    _log_disk_space(models_dir)
    logger.info("=" * 60)

    try:
        wan_dir = os.path.join(models_dir, WAN_DIR_NAME)
        if os.path.exists(wan_dir) and os.listdir(wan_dir) and _verify_model_files(wan_dir):
            size = _get_dir_size(wan_dir)
            logger.info(
                f"[1/1] ✅ SKIP {WAN_DIR_NAME}/ "
                f"(already exists, {_format_size(size)})"
            )
        else:
            if os.path.exists(wan_dir) and os.listdir(wan_dir):
                logger.info(
                    f"[1/1] ⬇️  COMPLETING {WAN_DIR_NAME}/ "
                    f"(incomplete download detected, resuming from {WAN_MODEL_ID}...)"
                )
            else:
                logger.info(
                    f"[1/1] ⬇️  DOWNLOADING {WAN_DIR_NAME}/ "
                    f"from {WAN_MODEL_ID}... (~3-4 GB)"
                )
            file_start = time.time()
            snapshot_download(
                repo_id=WAN_MODEL_ID,
                local_dir=wan_dir,
                local_dir_use_symlinks=False,
            )
            elapsed = time.time() - file_start
            size = _get_dir_size(wan_dir) if os.path.exists(wan_dir) else 0
            logger.info(
                f"[1/1] ✅ DONE {WAN_DIR_NAME}/ "
                f"({_format_size(size)}, took {_format_duration(elapsed)})"
            )

        total_elapsed = time.time() - total_start

        logger.info("")
        logger.info("=" * 60)
        logger.info("  ✅ Model check completed successfully!")
        logger.info(f"  Total time: {_format_duration(total_elapsed)}")
        logger.info(f"  Models directory: {models_dir}")
        _log_disk_space(models_dir)
        logger.info("")
        logger.info("  Files in models directory:")
        if os.path.exists(wan_dir) and os.listdir(wan_dir):
            size = _format_size(_get_dir_size(wan_dir))
            logger.info(f"    ✅ {WAN_DIR_NAME}/ ({size})")
        else:
            logger.info(f"    ❌ {WAN_DIR_NAME}/ (MISSING)")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        total_elapsed = time.time() - total_start
        logger.error("")
        logger.error("=" * 60)
        logger.error(f"  ❌ FAILED to download model after {_format_duration(total_elapsed)}")
        logger.error(f"  Error: {str(e)}")
        logger.error("")
        logger.error("  Possible causes:")
        logger.error("    - No internet connection")
        logger.error("    - Not enough disk space")
        logger.error("    - HuggingFace Hub is down")
        _log_disk_space(models_dir)
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(download_models())
