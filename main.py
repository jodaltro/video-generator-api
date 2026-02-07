"""
Video Generation API using LTX-2 Model
Main FastAPI application
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os

from config import settings
from video_generator import VideoGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Video Generator API (LTX-2)",
    description="API for generating videos from text using the LTX-2 model from Lightricks",
    version="2.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Global video generator instance (lazy loaded)
video_generator: Optional[VideoGenerator] = None


class VideoGenerationRequest(BaseModel):
    """Request model for video generation"""
    prompt: str = Field(
        ...,
        description="Text description of the video to generate",
        examples=["A serene sunset over the ocean with waves gently rolling"]
    )
    duration: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Duration of the video in seconds (1-10)",
        examples=[3.0]
    )
    num_frames: Optional[int] = Field(
        default=None,
        ge=8,
        le=257,
        description="Number of frames to generate (8-257). If not specified, calculated from duration and fps",
        examples=[121]
    )
    fps: int = Field(
        default=25,
        ge=8,
        le=60,
        description="Frames per second (8-60)",
        examples=[25]
    )
    width: int = Field(
        default=512,
        ge=256,
        le=1280,
        description="Video width in pixels (256-1280, must be divisible by 8)",
        examples=[512]
    )
    height: int = Field(
        default=768,
        ge=256,
        le=1280,
        description="Video height in pixels (256-1280, must be divisible by 8)",
        examples=[768]
    )
    num_inference_steps: int = Field(
        default=40,
        ge=4,
        le=100,
        description="Number of denoising steps (4-100). DistilledPipeline uses 8 predefined sigmas",
        examples=[40]
    )
    guidance_scale: float = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="Guidance scale for prompt adherence (1-20). Higher = more prompt following",
        examples=[3.0]
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. If not specified, uses random seed",
        examples=[42]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A serene sunset over the ocean with waves gently rolling",
                "duration": 3.0,
                "fps": 25,
                "width": 512,
                "height": 768,
                "num_inference_steps": 40,
                "guidance_scale": 3.0,
                "seed": 42
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str
    models_dir: str
    message: str


def get_video_generator() -> VideoGenerator:
    """Get or initialize the video generator"""
    global video_generator
    if video_generator is None:
        logger.info("Initializing LTX-2 video generator...")
        video_generator = VideoGenerator(
            checkpoint_path=settings.checkpoint_path,
            spatial_upsampler_path=settings.spatial_upsampler_path,
            gemma_root=settings.gemma_root_path,
            distilled_lora_path=settings.distilled_lora_path,
            enable_fp8=settings.enable_fp8,
        )
    return video_generator


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint

    Returns the API status and whether the model is loaded
    """
    global video_generator
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="healthy",
        model_loaded=video_generator is not None and video_generator.is_loaded(),
        device=device,
        models_dir=settings.models_dir,
        message="Video Generator API (LTX-2) is running"
    )


@app.post(
    "/generate",
    responses={
        200: {
            "content": {"video/mp4": {}},
            "description": "Generated video file"
        }
    },
    tags=["Video Generation"]
)
async def generate_video(request: VideoGenerationRequest):
    """
    Generate a video from text prompt using LTX-2

    This endpoint generates a video based on the provided text prompt and parameters.
    Uses the LTX-2 DistilledPipeline for fast inference.

    - **prompt**: Text description of the video to generate
    - **duration**: Length of video in seconds (1-10)
    - **num_frames**: Number of frames (optional, calculated from duration/fps if not provided)
    - **fps**: Frames per second (8-60)
    - **width**: Video width in pixels (256-1280, divisible by 8)
    - **height**: Video height in pixels (256-1280, divisible by 8)
    - **num_inference_steps**: Number of denoising steps (4-100)
    - **guidance_scale**: Prompt adherence strength (1-20)
    - **seed**: Random seed for reproducibility (optional)

    Returns an MP4 video file
    """
    try:
        logger.info(f"Received video generation request: {request.prompt[:50]}...")

        # Validate dimensions are divisible by 8
        if request.width % 8 != 0 or request.height % 8 != 0:
            raise HTTPException(
                status_code=400,
                detail="Width and height must be divisible by 8"
            )

        # Get video generator instance
        generator = get_video_generator()

        # Calculate num_frames if not provided
        num_frames = request.num_frames
        if num_frames is None:
            num_frames = int(request.duration * request.fps)
            # Ensure it's within valid range
            num_frames = max(8, min(257, num_frames))

        # Generate video
        logger.info(f"Generating video with {num_frames} frames at {request.fps} fps")
        output_path = generator.generate(
            prompt=request.prompt,
            num_frames=num_frames,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            fps=request.fps,
            seed=request.seed
        )

        logger.info("Video generation completed successfully")

        # Return video file
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename="generated_video.mp4",
            background=None,
        )

    except FileNotFoundError as e:
        logger.error(f"Model files not found: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Model files not found. Run 'python download_models.py' first. Details: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate video: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info("=" * 60)
    logger.info("  Video Generator API (LTX-2) - Server Starting")
    logger.info("=" * 60)
    logger.info(f"  Host: 0.0.0.0")
    logger.info(f"  Port: {port}")
    logger.info(f"  Models dir: {settings.models_dir}")
    logger.info(f"  Swagger UI: http://0.0.0.0:{port}/")
    logger.info(f"  Health check: http://0.0.0.0:{port}/health")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
