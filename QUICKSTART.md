# Quick Start Guide

## Local Development (Without Docker)

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Start the API

```bash
python main.py
```

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
    "prompt": "A cat playing with a ball",
    "duration": 3.0,
    "width": 512,
    "height": 512
  }' \
  --output my_video.mp4
```

## Docker Deployment

### Using Docker Compose (Recommended)

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

### Using Docker Directly

```bash
# Build
docker build -t video-generator-api .

# Run
docker run -p 8000:8000 video-generator-api

# Run with GPU (NVIDIA)
docker run --gpus all -p 8000:8000 video-generator-api

# Run with persistent cache
docker run -p 8000:8000 \
  -v $(pwd)/cache:/root/.cache/huggingface \
  video-generator-api
```

## Important Notes

### First Run
- The first video generation will take 10-20 minutes
- This is because the LTX-Video model (~10-15 GB) needs to download
- Subsequent generations will be much faster
- Models are cached in `~/.cache/huggingface/`

### Hardware Requirements
- **CPU only**: Works but slow (~20-30 min per video)
- **GPU (NVIDIA)**: Fast (~1-3 min per video)
- **Minimum RAM**: 16 GB
- **Recommended RAM**: 32 GB
- **Disk space**: 20 GB minimum (for models)

### GPU Support
The API automatically detects and uses GPU if available:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- Falls back to CPU if no GPU detected

### Performance Tips
For faster generation:
- Lower resolution (256x256 or 384x384)
- Fewer inference steps (20-25 instead of 30-50)
- Shorter videos (2-3 seconds)

For better quality:
- Higher resolution (768x768 or 1024x1024)
- More inference steps (40-50)
- Higher guidance scale (4-5)

## Troubleshooting

### "Connection refused" error
- Make sure the API is running: `python main.py`
- Check the correct port: default is 8000

### "CUDA out of memory"
- Reduce resolution: try 256x256
- Reduce num_frames: try 40-60
- Use a GPU with more VRAM

### Model download is slow
- This is normal on first run
- Depends on your internet speed
- Models are ~10-15 GB total

### "Module not found" errors
- Make sure you installed all requirements: `pip install -r requirements.txt`
- Activate virtual environment if using one

## Testing

Run the test suite:
```bash
python test_api.py
```

Run the examples:
```bash
python examples.py
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for cloud deployment
- Explore the Swagger UI for all API options
- Experiment with different prompts and parameters
