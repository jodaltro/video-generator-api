# Project Summary: Video Generator API (LTX-2)

## Overview

This project implements a video generation API using the state-of-the-art **LTX-2** model (19B parameters) from Lightricks. The API is designed to be deployed on GPU pods (RunPod, etc.), with models downloaded separately to keep the Docker image manageable.

## Key Features

### 🎯 Core Functionality
- **Single endpoint API** (`/generate`) for video generation from text
- **LTX-2 Model**: 19B parameter DiT-based audio-video foundation model
- **DistilledPipeline**: Fast inference with 8 predefined sigmas
- **FP8 Support**: Lower memory footprint for consumer GPUs
- **Swagger documentation**: Interactive API docs at root URL
- **Health check**: Monitor API and model loading status

### 🐳 Docker Optimization
- **NVIDIA CUDA base image**: Full GPU support with CUDA 12.1
- **Lightweight image**: Python dependencies installed at pod startup, not baked into image
- **LTX-2 native pipelines**: Uses official ltx-pipelines from Lightricks
- **Separate model download**: Models downloaded via script, not baked into image
- **Persistent volumes**: Dependencies and model files persist across container restarts

### ☁️ Pod/RunPod Ready
- Optimized for GPU deployment (A40/A100 recommended)
- Environment variable configuration
- Persistent volume support for model storage
- Health checks included
- Comprehensive deployment guide

## Project Structure

```
video-generator-api/
├── main.py                      # FastAPI application
├── video_generator.py           # Video generation service (LTX-2)
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── download_models.py           # LTX-2 model download script
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

### Video Generation (LTX-2)
- **Model**: LTX-2 19B from Lightricks
- **Pipeline**: DistilledPipeline (fastest inference)
- **Text Encoder**: Gemma 3 12B
- **Upscaler**: Spatial upscaler 2x
- **Output**: MP4 format
- **Performance**: GPU acceleration with FP8 support

### Docker Image
- **Base**: nvidia/cuda:12.1.1-devel-ubuntu22.04
- **System deps**: ffmpeg, CUDA toolkit, Python 3.10
- **Runtime deps**: PyTorch (CUDA), ltx-pipelines, ltx-core, FastAPI (installed at first startup)
- **LTX-2 packages**: Installed from official GitHub repo at first startup

## Required Models (~30 GB total)

| File | Size | Description |
|------|------|-------------|
| `ltx-2-19b-distilled-fp8.safetensors` | ~10 GB | Main LTX-2 checkpoint |
| `ltx-2-spatial-upscaler-x2-1.0.safetensors` | ~2 GB | Spatial upscaler |
| `ltx-2-19b-distilled-lora-384.safetensors` | ~1 GB | Distilled LoRA |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | ~15 GB | Gemma 3 text encoder |

## API Endpoints

### POST /generate
Generate a video from text prompt using LTX-2.

**Parameters:**
- `prompt` (required): Text description
- `duration`: Video length in seconds (1-10)
- `fps`: Frames per second (8-60)
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
  -v ltx2-repo:/opt/LTX-2 \
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
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM, FP8 only)
- **RAM**: 32 GB
- **Storage**: 50 GB
- **CUDA**: 12.1+

### Recommended
- **GPU**: NVIDIA A40 (48 GB) or A100 (80 GB)
- **RAM**: 64 GB+
- **Storage**: 100 GB SSD

## License

- **LTX-2**: Apache 2.0 License (Lightricks)
- **Gemma 3**: Google Terms of Service
- **FastAPI**: MIT License
- **PyTorch**: BSD-style License
