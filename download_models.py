#!/usr/bin/env python3
"""
Pre-download LTX-2 models for faster startup.
Downloads all required model files from HuggingFace to the models directory.
"""
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

LTX2_REPO_ID = "Lightricks/LTX-2"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"
# Base Gemma repo for downloading tokenizer.model (SentencePiece) if missing from QAT variant
GEMMA_BASE_REPO_ID = "google/gemma-3-12b-it"

# LTX-2 model files to download
LTX2_FILES = [
    "ltx-2-19b-distilled-fp8.safetensors",
    "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "ltx-2-19b-distilled-lora-384.safetensors",
]

GEMMA_DIR_NAME = "gemma-3-12b-it-qat-q4_0-unquantized"

ALL_MODELS = LTX2_FILES + [GEMMA_DIR_NAME + "/"]


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


def download_models():
    """Download all required LTX-2 models"""
    from huggingface_hub import hf_hub_download, snapshot_download

    # Check for HuggingFace token (required for gated models like Gemma)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        logger.warning("⚠️  No HF_TOKEN found. Gated models (like Gemma) will fail to download.")
        logger.warning("   Get a token at: https://huggingface.co/settings/tokens")
        logger.warning("   Request access to: https://huggingface.co/google/gemma-3-12b-it")
    
    models_dir = os.getenv("MODELS_DIR", "/workspace/models")
    os.makedirs(models_dir, exist_ok=True)

    total_start = time.time()

    logger.info("=" * 60)
    logger.info("  LTX-2 Model Download")
    logger.info("=" * 60)
    logger.info(f"  Models directory: {models_dir}")
    _log_disk_space(models_dir)
    logger.info(f"  Models to check: {len(ALL_MODELS)}")
    logger.info("=" * 60)

    downloaded_count = 0
    skipped_count = 0

    try:
        # Download LTX-2 model files
        for i, filename in enumerate(LTX2_FILES, 1):
            dest_path = os.path.join(models_dir, filename)
            if os.path.exists(dest_path):
                size = os.path.getsize(dest_path)
                logger.info(
                    f"[{i}/{len(ALL_MODELS)}] ✅ SKIP {filename} "
                    f"(already exists, {_format_size(size)})"
                )
                skipped_count += 1
                continue

            logger.info(
                f"[{i}/{len(ALL_MODELS)}] ⬇️  DOWNLOADING {filename} "
                f"from {LTX2_REPO_ID}..."
            )
            file_start = time.time()
            hf_hub_download(
                repo_id=LTX2_REPO_ID,
                filename=filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
                token=hf_token,
            )
            elapsed = time.time() - file_start
            size = os.path.getsize(dest_path)
            logger.info(
                f"[{i}/{len(ALL_MODELS)}] ✅ DONE {filename} "
                f"({_format_size(size)}, took {_format_duration(elapsed)})"
            )
            downloaded_count += 1

        # Download Gemma text encoder
        gemma_index = len(LTX2_FILES) + 1
        gemma_dir = os.path.join(models_dir, GEMMA_DIR_NAME)
        if os.path.exists(gemma_dir) and os.listdir(gemma_dir):
            size = _get_dir_size(gemma_dir)
            logger.info(
                f"[{gemma_index}/{len(ALL_MODELS)}] ✅ SKIP gemma-3-12b-it-qat-q4_0-unquantized/ "
                f"(already exists, {_format_size(size)})"
            )
            skipped_count += 1
        else:
            logger.info(
                f"[{gemma_index}/{len(ALL_MODELS)}] ⬇️  DOWNLOADING gemma-3-12b-it-qat-q4_0-unquantized/ "
                f"from {GEMMA_REPO_ID}... (this is the largest download, ~15 GB)"
            )
            file_start = time.time()
            snapshot_download(
                repo_id=GEMMA_REPO_ID,
                local_dir=gemma_dir,
                local_dir_use_symlinks=False,
                token=hf_token,
            )
            elapsed = time.time() - file_start
            size = _get_dir_size(gemma_dir) if os.path.exists(gemma_dir) else 0
            logger.info(
                f"[{gemma_index}/{len(ALL_MODELS)}] ✅ DONE gemma-3-12b-it-qat-q4_0-unquantized/ "
                f"({_format_size(size)}, took {_format_duration(elapsed)})"
            )
            downloaded_count += 1

        # Ensure all required configuration files exist in the Gemma directory.
        # The QAT variant may not include these files, so download from the base Gemma repo.
        required_files = [
            "tokenizer.model",           # SentencePiece tokenizer model
            "preprocessor_config.json",  # Preprocessor configuration (required by transformers)
            "tokenizer_config.json",     # Tokenizer configuration
            "special_tokens_map.json",   # Special tokens mapping
        ]

        for filename in required_files:
            file_path = os.path.join(gemma_dir, filename)
            if not os.path.exists(file_path):
                logger.info(
                    f"⬇️  DOWNLOADING {filename} from "
                    f"{GEMMA_BASE_REPO_ID} (required by LTX-2 text encoder)..."
                )
                file_start = time.time()
                try:
                    hf_hub_download(
                        repo_id=GEMMA_BASE_REPO_ID,
                        filename=filename,
                        local_dir=gemma_dir,
                        local_dir_use_symlinks=False,
                        token=hf_token,
                    )
                    elapsed = time.time() - file_start
                    logger.info(
                        f"✅ DONE {filename} (took {_format_duration(elapsed)})"
                    )
                except Exception as e:
                    # Handle download failures gracefully - special_tokens_map.json may be optional
                    # for some models, but preprocessor_config.json and tokenizer files are critical
                    logger.warning(f"⚠️  Could not download {filename}: {str(e)}")
            else:
                logger.info(f"✅ {filename} already present")

        total_elapsed = time.time() - total_start

        logger.info("")
        logger.info("=" * 60)
        logger.info("  ✅ Model check completed successfully!")
        logger.info(f"  Downloaded: {downloaded_count} model(s)")
        logger.info(f"  Skipped (already present): {skipped_count} model(s)")
        logger.info(f"  Total time: {_format_duration(total_elapsed)}")
        logger.info(f"  Models directory: {models_dir}")
        _log_disk_space(models_dir)
        logger.info("")
        logger.info("  Files in models directory:")
        for filename in LTX2_FILES:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                size = _format_size(os.path.getsize(filepath))
                logger.info(f"    ✅ {filename} ({size})")
            else:
                logger.info(f"    ❌ {filename} (MISSING)")
        if os.path.exists(gemma_dir) and os.listdir(gemma_dir):
            size = _format_size(_get_dir_size(gemma_dir))
            logger.info(f"    ✅ gemma-3-12b-it-qat-q4_0-unquantized/ ({size})")
        else:
            logger.info(f"    ❌ gemma-3-12b-it-qat-q4_0-unquantized/ (MISSING)")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        total_elapsed = time.time() - total_start
        logger.error("")
        logger.error("=" * 60)
        logger.error(f"  ❌ FAILED to download models after {_format_duration(total_elapsed)}")
        logger.error(f"  Error: {str(e)}")
        logger.error("")
        logger.error("  Possible causes:")
        logger.error("    - No internet connection")
        logger.error("    - Not enough disk space")
        logger.error("    - HuggingFace Hub is down")
        logger.error("    - Missing or invalid HuggingFace token (for gated models)")
        logger.error("")
        logger.error("  FOR GATED MODELS (Gemma):")
        logger.error("    1. Request access: https://huggingface.co/google/gemma-3-12b-it")
        logger.error("    2. Get token: https://huggingface.co/settings/tokens")
        logger.error("    3. Set environment variable: HF_TOKEN=hf_...")
        _log_disk_space(models_dir)
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(download_models())
