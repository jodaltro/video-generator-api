# Quick Start Guide (Wan2.1)

## Docker Deployment (Recommended)

### 1. Build the Docker Image

```bash
./build_and_push.sh
```

### 2. Start the API

```bash
docker run --gpus all -p 8000:8000 \
  -v workspace-data:/workspace \
  -v pip-packages:/usr/local/lib/python3.10/dist-packages \
  -v pip-bin:/usr/local/bin \
  video-generator-api:latest
```

On first run, the container will automatically:
1. Install Python dependencies (PyTorch, diffusers, etc.)
2. Download ~4 GB model files (takes 2-5 minutes)

Subsequent starts will skip installation if volumes are preserved.

The API will start on http://localhost:8000

### 3. Access Swagger UI

Open your browser and go to: http://localhost:8000

You'll see the interactive API documentation where you can:
- View all endpoints
- Try the API directly from the browser
- See request/response examples

### 4. Generate Your First Video

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
    "duration": 2.0,
    "width": 480,
    "height": 320
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
- Dependencies and model are installed/downloaded automatically on first start
- The image is lightweight; Python packages are installed at runtime
- Model is ~4 GB total (Wan2.1-T2V-1.3B)
- No HuggingFace token required (public model)
- Use persistent volumes to avoid re-downloading on restarts

### Hardware Requirements
- **GPU (NVIDIA)**: Required - CUDA 12.1+ compatible
- **Minimum VRAM**: 8 GB (RTX 3060)
- **Recommended VRAM**: 12 GB (RTX 4070)
- **RAM**: 16 GB minimum
- **Disk space**: 10 GB minimum (for model)

### Performance Tips
For faster generation:
- Use lower resolution (480x320)
- Fewer frames (33)
- Fewer inference steps (25)

For better quality:
- Higher resolution (640x480)
- More frames
- More inference steps (50)

## Build and Push to Docker Hub

```bash
# Build and push
./build_and_push.sh --push --username your-username

# With version tag
./build_and_push.sh --push --username your-username --tag v3.0
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
