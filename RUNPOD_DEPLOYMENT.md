# RunPod Deployment Guide (LTX-2)

## Overview

This guide explains how to deploy the Video Generator API with LTX-2 on RunPod with GPU support.

## Prerequisites

1. RunPod account with credits
2. Docker Hub account (or other container registry)
3. Docker installed locally for building the image

## Step 1: Build and Push Docker Image

```bash
# Build and push using the provided script
./build_and_push.sh --push --username your-username

# Or with a specific tag
./build_and_push.sh --push --username your-username --tag v2.0
```

## Step 2: Create a RunPod Template

1. Go to RunPod Templates
2. Click "New Template"
3. Configure:
   - **Template Name**: Video Generator API LTX-2
   - **Container Image**: `your-username/video-generator-api:latest`
   - **Container Disk**: 20 GB
   - **Volume Disk**: 100 GB (for LTX-2 models ~30 GB total)
   - **Volume Path**: `/workspace`
   - **Expose HTTP Ports**: `8000`
   - **Environment Variables**:
     ```
     PORT=8000
     MODELS_DIR=/workspace/models
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     ```

## Step 3: Deploy a Pod

1. Go to "Pods" section
2. Click "Deploy"
3. Select your template
4. Choose GPU:
   - **Recommended**: NVIDIA A40 (48GB VRAM) or A100 (80GB VRAM)
   - **Minimum**: RTX 4090 (24GB VRAM) with FP8 checkpoint
5. Select Storage:
   - **Container Disk**: 20 GB minimum
   - **Volume**: 100 GB recommended (for persistent model storage)
6. Click "Deploy"

## Step 4: Download Models (First Run)

On first deployment, connect to the pod terminal and download models:

```bash
# Via pod terminal
python download_models.py
```

Or run a one-off container:
```bash
# This downloads models to the persistent volume
docker exec <container-id> python download_models.py
```

The download includes:
- `ltx-2-19b-distilled-fp8.safetensors` (~10 GB)
- `ltx-2-spatial-upscaler-x2-1.0.safetensors` (~2 GB)
- `ltx-2-19b-distilled-lora-384.safetensors` (~1 GB)
- `gemma-3-12b-it-qat-q4_0-unquantized/` (~15 GB)

**Total**: ~30 GB
**Estimated time**: 10-30 minutes depending on network speed.

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
    "duration": 3.0,
    "width": 512,
    "height": 768
  }' \
  --output video.mp4
```

## Performance Tips

### GPU Selection

| GPU | VRAM | Speed | Cost | Notes |
|-----|------|-------|------|-------|
| RTX 4090 | 24GB | ~2-3 min | $$ | FP8 only |
| A40 | 48GB | ~1-2 min | $$$ | Recommended |
| A100 | 80GB | ~1 min | $$$$ | Best performance |

### Optimization Settings

For faster generation:
- Use the DistilledPipeline (default) - fastest with 8 predefined sigmas
- Use FP8 checkpoint (default) - lower memory footprint
- Use `width: 512, height: 768` (default resolution)

For better quality:
- Use `ltx-2-19b-dev` or `ltx-2-19b-dev-fp8` checkpoint
- Use higher resolution if VRAM allows

### Volume Storage

**Recommended setup:**
- Mount volume at `/workspace`
- Set `MODELS_DIR=/workspace/models`
- Models persist across pod restarts
- Much faster startup after first download

## Troubleshooting

### Pod won't start
- Check Docker image name and tag
- Verify port 8000 is exposed
- Check RunPod logs for errors

### Model download fails
- Check internet connectivity
- Verify disk space (need 50GB+)
- Check HuggingFace Hub status
- Re-run `python download_models.py` (supports resume)

### Out of memory errors
- Use FP8 checkpoint (default)
- Reduce resolution (384x512)
- Reduce num_frames (60-80)
- Use A40 or A100 GPU

### "Model files not found" error
- Run `python download_models.py` first
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
