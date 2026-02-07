# Use NVIDIA CUDA base image for GPU support with LTX-2
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8000
ENV MODELS_DIR=/workspace/models
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone LTX-2 repository and install its packages
RUN git clone https://github.com/Lightricks/LTX-2.git /opt/LTX-2 && \
    cd /opt/LTX-2 && \
    pip install --no-cache-dir -e packages/ltx-core && \
    pip install --no-cache-dir -e packages/ltx-pipelines

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY video_generator.py .
COPY config.py .
COPY download_models.py .
COPY start.sh .
RUN chmod +x start.sh

# Create directories for models
RUN mkdir -p /workspace/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application (downloads models on first start, then starts the API)
CMD ["./start.sh"]
