# Quick Start Guide (LTX-2)

## Docker Deployment (Recommended)

### 1. Build the Docker Image

```bash
./build_and_push.sh
```

### 2. Download Models (First Time)

```bash
docker run --gpus all -v models-cache:/workspace/models video-generator-api:latest python download_models.py
```

This downloads ~30 GB of model files. Takes 10-30 minutes.

### 3. Start the API

```bash
docker run --gpus all -p 8000:8000 -v models-cache:/workspace/models video-generator-api:latest
```

The API will start on http://localhost:8000

### 4. Access Swagger UI

Open your browser and go to: http://localhost:8000

You'll see the interactive API documentation where you can:
- View all endpoints
- Try the API directly from the browser
- See request/response examples

### 5. Generate Your First Video

Using the Swagger UI:
1. Click on `POST /generate`
2. Click "Try it out"
3. Modify the request body (or use defaults)
4. Click "Execute"
5. Download the generated video

Using curl:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing with a ball of yarn in a cozy living room",
    "duration": 3.0,
    "width": 512,
    "height": 768
  }' \
  --output my_video.mp4
```

## Docker Compose

```bash
# Start the service
docker-compose up

# Or in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Important Notes

### First Run
- You must download models before the first video generation
- Run `python download_models.py` inside the container
- Models are ~30 GB total (LTX-2 checkpoint + Gemma encoder + upscaler + LoRA)
- Models persist in the volume across container restarts

### Hardware Requirements
- **GPU (NVIDIA)**: Required - CUDA 12.1+ compatible
- **Minimum VRAM**: 24 GB (RTX 4090, with FP8 checkpoint)
- **Recommended VRAM**: 48 GB (A40) or 80 GB (A100)
- **RAM**: 32 GB minimum
- **Disk space**: 50 GB minimum (for models)

### Performance Tips
For faster generation:
- Use the DistilledPipeline (default) - fastest with predefined sigmas
- Use FP8 checkpoint (default) - lower memory footprint
- Lower resolution (384x512)

For better quality:
- Use `ltx-2-19b-dev` checkpoint
- Higher resolution (768x1024)
- More inference steps

## Build and Push to Docker Hub

```bash
# Build and push
./build_and_push.sh --push --username your-username

# With version tag
./build_and_push.sh --push --username your-username --tag v2.0
```

## Testing

Run the test suite:
```bash
python test_api.py
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for cloud deployment
- Explore the Swagger UI for all API options
