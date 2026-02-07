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

# Install Python dependencies on first startup
DEPS_MARKER="/workspace/.deps_installed"
# Verify that key packages are actually importable; if not, force reinstall
if [ -f "$DEPS_MARKER" ] && ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]  Marker file exists but packages missing. Reinstalling..."
    rm -f "$DEPS_MARKER"
fi
if [ ! -f "$DEPS_MARKER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 1/3] Installing Python dependencies (first run)..."
    echo ""

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  Installing PyTorch with CUDA support..."
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  Cloning and installing LTX-2..."
    if [ ! -d "/opt/LTX-2" ]; then
        git clone https://github.com/Lightricks/LTX-2.git /opt/LTX-2
    fi
    (cd /opt/LTX-2 && pip install --no-cache-dir -e packages/ltx-core && pip install --no-cache-dir -e packages/ltx-pipelines)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  Installing requirements..."
    pip install --no-cache-dir -r requirements.txt

    touch "$DEPS_MARKER"
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 1/3] ✅ Dependencies installed successfully."
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 1/3] ✅ Dependencies already installed (skipping)."
fi

echo ""

# Download models if not already present
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 2/3] Checking and downloading models..."
echo ""
if python download_models.py; then
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 2/3] ✅ Model check/download completed successfully."
else
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 2/3] ❌ ERROR: Model download failed! Check the logs above for details."
    exit 1
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP 3/3] Starting API server on port ${PORT:-8000}..."
echo ""
exec python main.py
