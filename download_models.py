#!/usr/bin/env python3
"""
Pre-download models for faster startup
Run this script to download models before the first API request
"""
import os
import sys
import logging
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_ID = "Lightricks/LTX-Video"


def download_models():
    """Download LTX-Video models"""
    logger.info(f"Downloading models from {MODEL_ID}...")
    logger.info("This may take 10-20 minutes depending on your connection...")
    
    try:
        # Download the model
        cache_dir = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        
        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        
        logger.info("✓ Models downloaded successfully!")
        logger.info(f"Cache location: {cache_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"✗ Failed to download models: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(download_models())
