# RunPod Deployment Guide (Wan2.1)

## Overview

This guide explains how to deploy the Video Generator API with Wan2.1 on RunPod with GPU support.

## Prerequisites

1. RunPod account with credits
2. Docker Hub account (or other container registry)
3. Docker installed locally for building the image

## Step 1: Build and Push Docker Image

```bash
# Build and push using the provided script
./build_and_push.sh --push --username your-username

# Or with a specific tag
./build_and_push.sh --push --username your-username --tag v3.0
```

## Step 2: Create a RunPod Template

1. Go to RunPod Templates
2. Click "New Template"
3. Configure:
   - **Template Name**: Video Generator API Wan2.1
   - **Container Image**: `your-username/video-generator-api:latest`
   - **Container Disk**: 10 GB
   - **Volume Disk**: 20 GB (for Wan2.1 model ~4 GB total)
   - **Volume Path**: `/workspace`
   - **Expose HTTP Ports**: `8000`
   - **Environment Variables**:
     ```
     PORT=8000
     MODELS_DIR=/workspace/models
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     ```

   > ✅ **No HuggingFace token required** - the Wan2.1 model is public.

## Step 3: Deploy a Pod

1. Go to "Pods" section
2. Click "Deploy"
3. Select your template
4. Choose GPU:
   - **Minimum**: Any NVIDIA GPU with 8GB+ VRAM
   - **Recommended**: RTX 4070 (12GB VRAM) or better
5. Select Storage:
   - **Container Disk**: 10 GB minimum
   - **Volume**: 20 GB recommended (for persistent model storage)
6. Click "Deploy"

## Step 4: Automatic Setup on First Start

When the pod starts for the first time, the container will automatically:

1. **Install Python dependencies** (PyTorch, diffusers, etc.)
2. Check for existing model in `/workspace/models`
3. Download the Wan2.1 model from HuggingFace if missing
4. Start the API server

The download includes:
- `Wan2.1-T2V-1.3B-Diffusers/` (~3-4 GB)

**Total**: ~4 GB
**Estimated time**: 2-5 minutes depending on network speed.

> **Note**: On subsequent restarts, if dependencies and model are stored on a persistent volume, installation/download is skipped and the API starts immediately.

## Step 5: Test the API

Get your pod's public URL from RunPod dashboard, then:

```bash
# Health check
curl https://your-pod-id-8000.proxy.runpod.net/health

# Generate video
curl -X POST "https://your-pod-id-8000.proxy.runpod.net/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing with a ball of yarn in a cozy living room",
    "duration": 2.0,
    "width": 480,
    "height": 320
  }' \
  --output video.mp4
```

## Performance Tips

### GPU Selection

| GPU | VRAM | Cost | Notes |
|-----|------|------|-------|
| RTX 3060 | 8GB | $ | Minimum, works fine |
| RTX 4070 | 12GB | $$ | Recommended |
| RTX 4090 | 24GB | $$$ | Fastest |

### Optimization Settings

For faster generation:
- Use lower resolution (480x320)
- Use fewer inference steps (25)

For better quality:
- Use higher resolution if VRAM allows
- Use more inference steps (50)

### Volume Storage

**Recommended setup:**
- Mount volume at `/workspace`
- Set `MODELS_DIR=/workspace/models`
- Model persists across pod restarts
- Much faster startup after first download

## Troubleshooting

### Pod won't start
- Check Docker image name and tag
- Verify port 8000 is exposed
- Check RunPod logs for errors

### Model download fails
- Check internet connectivity
- Verify disk space (need 10GB+)
- Check HuggingFace Hub status
- Re-run `python download_models.py` (supports resume)

### Out of memory errors
- Reduce resolution (320x256)
- Reduce num_frames (16-24)
- Enable CPU offload: `ENABLE_CPU_OFFLOAD=true`

### "Model files not found" error
- Check the pod logs to see if the automatic download completed successfully
- If the download failed, check internet connectivity and disk space
- You can also manually re-run: `python download_models.py`
- Check that `MODELS_DIR` points to the correct volume path
- Verify model files exist: `ls /workspace/models/`

## API Access

### Public Endpoint
RunPod provides: `https://your-pod-id-8000.proxy.runpod.net`

### Swagger UI
Access at: `https://your-pod-id-8000.proxy.runpod.net/`

## Support

For issues:
1. Check pod logs first
2. Verify Docker image works locally
3. Verify model files are downloaded
4. Contact RunPod support if infrastructure issue
5. Open GitHub issue for API bugs
