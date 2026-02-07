#!/bin/bash
set -e

echo ""
echo "=========================================="
echo " Video Generator API (LTX-2) - Starting"
echo "=========================================="
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [START] Initializing pod..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  MODELS_DIR=${MODELS_DIR:-/workspace/models}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  PORT=${PORT:-8000}"
echo ""

# Download models if not already present
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 1/2] Checking and downloading models..."
echo ""
if python download_models.py; then
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 1/2] ✅ Model check/download completed successfully."
else
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 1/2] ❌ ERROR: Model download failed! Check the logs above for details."
    exit 1
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 2/2] Starting API server on port ${PORT:-8000}..."
echo ""
exec python main.py
