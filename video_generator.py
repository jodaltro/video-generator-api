"""
Video Generator Service using LTX-2 Model
"""
import os
import logging
import torch
import numpy as np
from typing import Optional
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from huggingface_hub import snapshot_download
import io
import tempfile

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Video generator using LTX-2 model"""
    
    def __init__(self, model_id: str = "Lightricks/LTX-Video"):
        """
        Initialize the video generator
        
        Args:
            model_id: HuggingFace model ID for LTX-2
        """
        self.model_id = model_id
        self.pipeline = None
        self.device = self._get_device()
        logger.info(f"VideoGenerator initialized with device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.pipeline is not None
    
    def _load_model(self):
        """Load the LTX-2 model (lazy loading)"""
        if self.pipeline is not None:
            return
        
        logger.info(f"Loading LTX-2 model from {self.model_id}...")
        logger.info("This may take a while on first run as models are downloaded...")
        
        try:
            # Load the pipeline
            self.pipeline = LTXPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            
            # Move to device
            self.pipeline.to(self.device)
            
            # Enable memory optimizations if on CUDA
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_vae_slicing()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        num_frames: int = 81,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        fps: int = 25,
        seed: Optional[int] = None
    ) -> bytes:
        """
        Generate a video from text prompt
        
        Args:
            prompt: Text description of the video
            num_frames: Number of frames to generate
            width: Video width in pixels
            height: Video height in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for prompt adherence
            fps: Frames per second for output video
            seed: Random seed for reproducibility
            
        Returns:
            Video bytes in MP4 format
        """
        # Load model if not already loaded
        self._load_model()
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            logger.info(f"Using seed: {seed}")
        
        logger.info(f"Generating video: {width}x{height}, {num_frames} frames, {fps} fps")
        
        # Generate video
        output = self.pipeline(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Get video frames
        video_frames = output.frames[0]  # Shape: (num_frames, height, width, channels)
        
        # Convert to MP4 using opencv or imageio
        video_bytes = self._frames_to_mp4(video_frames, fps)
        
        return video_bytes
    
    def _frames_to_mp4(self, frames: np.ndarray, fps: int) -> bytes:
        """
        Convert frames to MP4 bytes
        
        Args:
            frames: Video frames array
            fps: Frames per second
            
        Returns:
            MP4 video bytes
        """
        try:
            import cv2
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Get frame dimensions
                height, width = frames.shape[1:3]
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
                
                # Write frames
                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                
                # Read the video file as bytes
                with open(tmp_path, 'rb') as f:
                    video_bytes = f.read()
                
                return video_bytes
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except ImportError:
            # Fallback to imageio if opencv not available
            logger.warning("OpenCV not available, falling back to imageio")
            return self._frames_to_mp4_imageio(frames, fps)
    
    def _frames_to_mp4_imageio(self, frames: np.ndarray, fps: int) -> bytes:
        """
        Convert frames to MP4 bytes using imageio
        
        Args:
            frames: Video frames array
            fps: Frames per second
            
        Returns:
            MP4 video bytes
        """
        import imageio
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Ensure frames are uint8
            if frames.dtype != np.uint8:
                frames = (frames * 255).astype(np.uint8)
            
            # Write video
            imageio.mimwrite(tmp_path, frames, fps=fps, codec='libx264')
            
            # Read as bytes
            with open(tmp_path, 'rb') as f:
                video_bytes = f.read()
            
            return video_bytes
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
