# Project Summary: Video Generator API

## Overview

This project implements a complete video generation API using the state-of-the-art LTX-Video model from Lightricks. The API is designed to be deployed on RunPod with GPU support, with models downloaded on-demand to keep the Docker image lightweight.

## Key Features

### 🎯 Core Functionality
- **Single endpoint API** (`/generate`) for video generation from text
- **Modern AI model**: LTX-Video (Lightricks) - state-of-the-art text-to-video
- **Flexible parameters**: Control duration, resolution, quality, and more
- **Swagger documentation**: Interactive API docs at root URL
- **Health check**: Monitor API and model loading status

### 🐳 Docker Optimization
- **Lightweight base image**: Python 3.10-slim
- **Lazy model loading**: Models downloaded on first run (~10-15 GB)
- **Efficient caching**: Uses HuggingFace cache directory
- **GPU support**: Automatic detection (CUDA/MPS/CPU)
- **Memory optimizations**: CPU offload and VAE slicing enabled

### ☁️ RunPod Ready
- Optimized for GPU deployment
- Environment variable configuration
- Persistent volume support for model cache
- Health checks included
- Comprehensive deployment guide

## Project Structure

```
video-generator-api/
├── main.py                      # FastAPI application
├── video_generator.py           # Video generation service
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
├── .env.example                # Environment variables template
├── Makefile                    # Common development tasks
├── README.md                   # Main documentation
├── QUICKSTART.md               # Getting started guide
├── RUNPOD_DEPLOYMENT.md        # RunPod deployment guide
├── download_models.py          # Pre-download models script
├── test_api.py                 # API testing script
├── examples.py                 # Usage examples
└── .github/
    └── workflows/
        └── docker-build.yml    # CI/CD for Docker images
```

## Technical Architecture

### API Layer (FastAPI)
- **Framework**: FastAPI with Pydantic models
- **Validation**: Comprehensive request validation
- **Documentation**: Auto-generated OpenAPI/Swagger
- **Error handling**: Graceful error responses

### Video Generation (LTX-Video)
- **Model**: Lightricks/LTX-Video from HuggingFace
- **Pipeline**: Diffusers library integration
- **Output**: MP4 format using OpenCV/imageio
- **Performance**: GPU acceleration with fallback to CPU

### Docker Image
- **Base**: python:3.10-slim (minimal footprint)
- **System deps**: ffmpeg, OpenCV dependencies
- **Python deps**: PyTorch, Diffusers, FastAPI
- **Size**: ~2-3 GB (without models)
- **Runtime size**: ~12-18 GB (with cached models)

## API Endpoints

### POST /generate
Generate a video from text prompt.

**Parameters:**
- `prompt` (required): Text description
- `duration`: Video length in seconds (1-10)
- `fps`: Frames per second (8-60)
- `width`, `height`: Resolution (256-1024, divisible by 8)
- `num_inference_steps`: Quality (10-100)
- `guidance_scale`: Prompt adherence (1-20)
- `seed`: Reproducibility seed (optional)

**Response:** MP4 video file (binary)

### GET /health
Check API status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Video Generator API is running"
}
```

## Deployment Options

### 1. Local Development
```bash
pip install -r requirements.txt
python main.py
```

### 2. Docker
```bash
docker build -t video-generator-api .
docker run -p 8000:8000 video-generator-api
```

### 3. Docker Compose
```bash
docker-compose up
```

### 4. RunPod
See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for detailed instructions.

## Performance Characteristics

### Generation Time (per video)
- **CPU only**: 20-30 minutes
- **RTX 3090**: 2-3 minutes
- **RTX 4090**: 1-2 minutes
- **A40/A100**: 1 minute

### Resource Requirements
- **Minimum**: 4 CPU cores, 16 GB RAM, 20 GB disk
- **Recommended**: GPU with 24GB+ VRAM, 32 GB RAM, 50 GB SSD
- **Model size**: ~10-15 GB (downloaded on first run)

### Optimization Tips
**For speed:**
- Lower resolution (256x256, 384x384)
- Fewer inference steps (20-25)
- Shorter duration (2-3 seconds)

**For quality:**
- Higher resolution (768x768, 1024x1024)
- More inference steps (40-50)
- Higher guidance scale (4-5)

## Development Workflow

### Quick Start
```bash
# Install dependencies
make install

# Run API locally
make run

# Run tests
make test

# Run examples
make examples
```

### Docker Workflow
```bash
# Build image
make docker-build

# Run with Docker Compose
make docker-compose-up

# View logs
docker-compose logs -f

# Stop
make docker-compose-down
```

### Testing
```bash
# Basic API test
python test_api.py

# Usage examples
python examples.py

# Health check
curl http://localhost:8000/health
```

## CI/CD

GitHub Actions workflow automatically:
1. Builds Docker image on push to main/master
2. Pushes to Docker Hub (requires secrets)
3. Tags with version/branch/SHA
4. Uses build cache for faster builds

**Required secrets:**
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

## Security Considerations

1. **Model downloads**: From trusted HuggingFace repository
2. **Input validation**: Comprehensive parameter validation
3. **Resource limits**: Request parameters bounded
4. **No authentication**: Add auth layer for production
5. **Rate limiting**: Consider adding for production use

## Future Enhancements

Potential improvements:
- [ ] Image-to-video generation
- [ ] Video-to-video transformation
- [ ] Batch processing endpoint
- [ ] WebSocket for progress updates
- [ ] Authentication/API keys
- [ ] Rate limiting
- [ ] Video preview/thumbnails
- [ ] Multiple model support
- [ ] Custom fine-tuned models

## Dependencies

### Core
- FastAPI 0.104.1 - Web framework
- Uvicorn 0.24.0 - ASGI server
- Pydantic 2.5.0 - Data validation

### AI/ML
- PyTorch 2.1.1 - Deep learning framework
- Diffusers 0.25.0 - Diffusion models
- Transformers 4.36.0 - Model architectures
- Accelerate 0.25.0 - Training/inference optimization

### Video Processing
- OpenCV 4.8.1 - Video encoding
- imageio 2.33.0 - Alternative video I/O
- imageio-ffmpeg 0.4.9 - FFmpeg wrapper

## License

Components:
- **LTX-Video**: Apache 2.0 License (Lightricks)
- **FastAPI**: MIT License
- **Diffusers**: Apache 2.0 License
- **PyTorch**: BSD-style License

## Support & Contributing

- **Issues**: Open GitHub issues for bugs
- **Features**: Submit pull requests
- **Questions**: Use GitHub discussions
- **Documentation**: Help improve docs

## Acknowledgments

- **Lightricks** for the LTX-Video model
- **HuggingFace** for the Diffusers library
- **FastAPI** team for the excellent framework
- **RunPod** for GPU infrastructure

---

**Built with ❤️ for the AI video generation community**
