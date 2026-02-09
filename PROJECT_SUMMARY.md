# Project Summary: Video Generator API (Wan2.1)

## Overview

This project implements a video generation API using the lightweight **Wan2.1-T2V-1.3B** model (1.3B parameters) from Wan-AI. The API is designed to be deployed on GPU pods (RunPod, etc.), with models downloaded separately to keep the Docker image manageable. Compared to the previous LTX-2 setup (~30GB), this model requires only ~4GB, making it accessible on much smaller GPUs.

## Key Features

### 🎯 Core Functionality
- **Single endpoint API** (`/generate`) for video generation from text
- **Wan2.1 Model**: 1.3B parameter lightweight text-to-video model
- **HuggingFace Diffusers**: Standard pipeline, no custom packages needed
- **No audio generation**: Video-only output for reduced resource usage
- **No HF token required**: Public model, no authentication needed
- **Swagger documentation**: Interactive API docs at root URL
- **Health check**: Monitor API and model loading status

### 🐳 Docker Optimization
- **NVIDIA CUDA base image**: Full GPU support with CUDA 12.1
- **Lightweight image**: Python dependencies installed at pod startup, not baked into image
- **Standard diffusers pipeline**: No custom package installation needed
- **Separate model download**: Model downloaded via script, not baked into image
- **Persistent volumes**: Dependencies and model files persist across container restarts

### ☁️ Pod/RunPod Ready
- Optimized for GPU deployment (works on 8GB+ VRAM GPUs)
- Environment variable configuration
- Persistent volume support for model storage
- Health checks included
- Comprehensive deployment guide

## Project Structure

```
video-generator-api/
├── main.py                      # FastAPI application
├── video_generator.py           # Video generation service (Wan2.1)
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── download_models.py           # Wan2.1 model download script
├── Dockerfile                   # Docker image definition (CUDA 12.1)
├── docker-compose.yml           # Docker Compose configuration
├── build_and_push.sh            # Local Docker build & push script
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
├── .env.example                # Environment variables template
├── Makefile                    # Common development tasks
├── README.md                   # Main documentation
├── QUICKSTART.md               # Getting started guide
├── RUNPOD_DEPLOYMENT.md        # RunPod deployment guide
├── test_api.py                 # API testing script
└── examples.py                 # Usage examples
```

## Technical Architecture

### API Layer (FastAPI)
- **Framework**: FastAPI with Pydantic models
- **Validation**: Comprehensive request validation
- **Documentation**: Auto-generated OpenAPI/Swagger
- **Error handling**: Graceful error responses with model status

### Video Generation (Wan2.1)
- **Model**: Wan2.1-T2V-1.3B from Wan-AI
- **Pipeline**: WanPipeline (via HuggingFace diffusers)
- **Text Encoder**: Built-in (no separate model needed)
- **Output**: MP4 format (video only, no audio)
- **Performance**: GPU acceleration with FP16

### Docker Image
- **Base**: nvidia/cuda:12.1.1-devel-ubuntu22.04
- **System deps**: ffmpeg, CUDA toolkit, Python 3.10
- **Runtime deps**: PyTorch (CUDA), diffusers, FastAPI (installed at first startup)

## Required Model (~4 GB total)

| File | Size | Description |
|------|------|-------------|
| `Wan2.1-T2V-1.3B-Diffusers/` | ~3-4 GB | Wan2.1 text-to-video model |

## API Endpoints

### POST /generate
Generate a video from text prompt using Wan2.1.

**Parameters:**
- `prompt` (required): Text description
- `duration`: Video length in seconds (1-10)
- `fps`: Frames per second (8-30)
- `width`, `height`: Resolution (256-1280, divisible by 8)
- `num_inference_steps`: Quality (4-100)
- `guidance_scale`: Prompt adherence (1-20)
- `seed`: Reproducibility seed (optional)

**Response:** MP4 video file

### GET /health
Check API status and model loading state.

## Deployment

### Docker (Local)
```bash
./build_and_push.sh
docker run --gpus all -p 8000:8000 \
  -v workspace-data:/workspace \
  -v pip-packages:/usr/local/lib/python3.10/dist-packages \
  -v pip-bin:/usr/local/bin \
  video-generator-api:latest
```

### Docker Hub (Push)
```bash
./build_and_push.sh --push --username your-username
```

### RunPod/Pod
See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for detailed instructions.

## System Requirements

### Minimum
- **GPU**: NVIDIA RTX 3060 (8 GB VRAM)
- **RAM**: 16 GB
- **Storage**: 10 GB
- **CUDA**: 12.1+

### Recommended
- **GPU**: NVIDIA RTX 4070 (12 GB) or better
- **RAM**: 32 GB+
- **Storage**: 20 GB SSD

## License

- **Wan2.1**: Apache 2.0 License (Wan-AI)
- **FastAPI**: MIT License
- **PyTorch**: BSD-style License
