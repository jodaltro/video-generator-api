#!/usr/bin/env python3
"""
Pre-download LTX-2 models for faster startup.
Downloads all required model files from HuggingFace to the models directory.
"""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LTX2_REPO_ID = "Lightricks/LTX-2"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# LTX-2 model files to download
LTX2_FILES = [
    "ltx-2-19b-distilled-fp8.safetensors",
    "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "ltx-2-19b-distilled-lora-384.safetensors",
]


def download_models():
    """Download all required LTX-2 models"""
    from huggingface_hub import hf_hub_download, snapshot_download

    models_dir = os.getenv("MODELS_DIR", "/workspace/models")
    os.makedirs(models_dir, exist_ok=True)

    logger.info(f"Models directory: {models_dir}")
    logger.info("Downloading LTX-2 models. This may take 10-30 minutes...")

    try:
        # Download LTX-2 model files
        for filename in LTX2_FILES:
            dest_path = os.path.join(models_dir, filename)
            if os.path.exists(dest_path):
                logger.info(f"✓ {filename} already exists, skipping")
                continue
            logger.info(f"Downloading {filename} from {LTX2_REPO_ID}...")
            hf_hub_download(
                repo_id=LTX2_REPO_ID,
                filename=filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f"✓ {filename} downloaded")

        # Download Gemma text encoder
        gemma_dir = os.path.join(models_dir, "gemma-3-12b-it-qat-q4_0-unquantized")
        if os.path.exists(gemma_dir) and os.listdir(gemma_dir):
            logger.info("✓ Gemma text encoder already exists, skipping")
        else:
            logger.info(f"Downloading Gemma text encoder from {GEMMA_REPO_ID}...")
            snapshot_download(
                repo_id=GEMMA_REPO_ID,
                local_dir=gemma_dir,
                local_dir_use_symlinks=False,
            )
            logger.info("✓ Gemma text encoder downloaded")

        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ All models downloaded successfully!")
        logger.info(f"  Models directory: {models_dir}")
        logger.info("")
        logger.info("Downloaded files:")
        for filename in LTX2_FILES:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                size_gb = os.path.getsize(filepath) / (1024**3)
                logger.info(f"  - {filename} ({size_gb:.1f} GB)")
        logger.info(f"  - gemma-3-12b-it-qat-q4_0-unquantized/ (directory)")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"✗ Failed to download models: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(download_models())
