# RunPod Deployment Guide

## Overview

This guide explains how to deploy the Video Generator API on RunPod with GPU support.

## Prerequisites

1. RunPod account with credits
2. Docker Hub account (or other container registry)
3. Docker installed locally for building the image

## Step 1: Build and Push Docker Image

```bash
# Build the image
docker build -t your-username/video-generator-api:latest .

# Login to Docker Hub
docker login

# Push the image
docker push your-username/video-generator-api:latest
```

## Step 2: Create a RunPod Template

1. Go to RunPod Templates
2. Click "New Template"
3. Configure:
   - **Template Name**: Video Generator API
   - **Container Image**: `your-username/video-generator-api:latest`
   - **Container Disk**: 20 GB (for model cache)
   - **Volume Path**: `/workspace/.cache` (optional but recommended)
   - **Expose HTTP Ports**: `8000`
   - **Environment Variables**:
     ```
     PORT=8000
     HF_HOME=/workspace/.cache/huggingface
     ```

## Step 3: Deploy a Pod

1. Go to "Pods" section
2. Click "Deploy"
3. Select your template
4. Choose GPU:
   - **Recommended**: NVIDIA A40 (40GB VRAM)
   - **Minimum**: RTX 3090 (24GB VRAM)
   - **Budget**: RTX 4090 (24GB VRAM)
5. Select Storage:
   - **Container Disk**: 20 GB minimum
   - **Volume**: 50 GB recommended (for persistent model cache)
6. Click "Deploy"

## Step 4: First Run

On the first run:
1. Wait 10-20 minutes for model download
2. Models will be cached in `/workspace/.cache/huggingface/`
3. Check logs to monitor progress: Pod → Logs
4. Once ready, the API will respond to `/health` endpoint

## Step 5: Test the API

Get your pod's public URL from RunPod dashboard, then:

```bash
# Health check
curl https://your-pod-id-8000.proxy.runpod.net/health

# Generate video
curl -X POST "https://your-pod-id-8000.proxy.runpod.net/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing with a ball of yarn",
    "duration": 3.0,
    "width": 512,
    "height": 512
  }' \
  --output video.mp4
```

## Performance Tips

### GPU Selection

| GPU | VRAM | Speed | Cost | Recommended |
|-----|------|-------|------|-------------|
| RTX 3090 | 24GB | ~2-3 min/video | $ | Good for testing |
| RTX 4090 | 24GB | ~1-2 min/video | $$ | Best value |
| A40 | 48GB | ~1-2 min/video | $$$ | Production |
| A100 | 80GB | ~1 min/video | $$$$ | High throughput |

### Optimization Settings

For faster generation:
- Use `num_inference_steps: 20-25` (vs 30-50)
- Use `width: 512, height: 512` (vs 768x768)
- Use `num_frames: 60-80` (vs 120+)

For better quality:
- Use `num_inference_steps: 40-50`
- Use `guidance_scale: 4-5`
- Use higher resolution (768x768 or 1024x1024)

### Volume Storage

**Recommended setup:**
- Mount volume at `/workspace/.cache`
- Set `HF_HOME=/workspace/.cache/huggingface`
- Models persist across pod restarts
- Faster startup after first run

## Cost Estimation

Assuming RTX 4090 at $0.34/hour:

| Videos/hour | Duration | Cost/video |
|-------------|----------|------------|
| 30 | 3s | $0.011 |
| 20 | 5s | $0.017 |
| 10 | 10s | $0.034 |

*Costs vary by GPU type and provider rates*

## Troubleshooting

### Pod won't start
- Check Docker image name and tag
- Verify port 8000 is exposed
- Check RunPod logs for errors

### Model download fails
- Check internet connectivity
- Verify disk space (need 20GB+)
- Check HuggingFace Hub status

### Out of memory errors
- Reduce resolution (256x256 or 384x384)
- Reduce num_frames (40-60)
- Use GPU with more VRAM

### Slow generation
- Check if GPU is being used (should see CUDA in logs)
- Verify GPU is not throttled
- Try different GPU type

## API Access

### Public Endpoint
RunPod provides: `https://your-pod-id-8000.proxy.runpod.net`

### Swagger UI
Access at: `https://your-pod-id-8000.proxy.runpod.net/`

### Security
- Use RunPod's built-in authentication
- Or add your own API key mechanism
- Consider rate limiting for production

## Scaling

### Horizontal Scaling
1. Deploy multiple pods
2. Use load balancer (external)
3. Each pod handles ~20-30 videos/hour

### Auto-scaling
- Not natively supported by RunPod
- Consider Kubernetes alternative
- Or use serverless endpoints

## Monitoring

Check pod metrics:
- GPU utilization (should be 80-100% during generation)
- Memory usage
- Network I/O
- Request latency

## Support

For issues:
1. Check pod logs first
2. Verify Docker image works locally
3. Contact RunPod support if infrastructure issue
4. Open GitHub issue for API bugs
