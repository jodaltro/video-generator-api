"""
Video Generator Service using LTX-2 Model

Uses the official LTX-2 pipelines from the Lightricks/LTX-2 repository.
Supports text-to-video generation with the DistilledPipeline for fast inference.
"""
import os
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Video generator using LTX-2 model"""

    def __init__(
        self,
        checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        distilled_lora_path: Optional[str] = None,
        enable_fp8: bool = True,
        enable_cpu_offload: bool = False,
        clear_cache_before_generation: bool = True,
    ):
        """
        Initialize the video generator with LTX-2 model paths.

        Args:
            checkpoint_path: Path to LTX-2 checkpoint .safetensors file
            spatial_upsampler_path: Path to spatial upscaler .safetensors file
            gemma_root: Path to Gemma text encoder directory
            distilled_lora_path: Path to distilled LoRA .safetensors file (optional)
            enable_fp8: Whether to enable FP8 transformer for lower memory
        """
        self.checkpoint_path = checkpoint_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.gemma_root = gemma_root
        self.distilled_lora_path = distilled_lora_path
        self.enable_fp8 = enable_fp8
        self.enable_cpu_offload = enable_cpu_offload
        self.clear_cache_before_generation = clear_cache_before_generation
        self.pipeline = None
        self.device = self._get_device()
        logger.info(f"VideoGenerator initialized with device: {self.device}")
        logger.info(f"CPU offload: {enable_cpu_offload}, Clear cache: {clear_cache_before_generation}")
        
        # Set PyTorch memory allocation config for better memory management
        if self.device == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _log_gpu_memory(self, stage: str = ""):
        """Log GPU memory usage"""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            logger.info(
                f"GPU Memory {stage}: "
                f"Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB, "
                f"Free: {free:.2f}GB, "
                f"Total: {total:.2f}GB"
            )
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.device == "cuda" and torch.cuda.is_available():
            logger.info("Clearing GPU cache...")
            self._log_gpu_memory("before clear")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self._log_gpu_memory("after clear")

    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.pipeline is not None

    def _validate_model_files(self):
        """Validate that all required model files exist"""
        missing = []
        if not os.path.exists(self.checkpoint_path):
            missing.append(f"Checkpoint: {self.checkpoint_path}")
        if not os.path.exists(self.spatial_upsampler_path):
            missing.append(f"Spatial upsampler: {self.spatial_upsampler_path}")
        if not os.path.isdir(self.gemma_root):
            missing.append(f"Gemma encoder directory: {self.gemma_root}")
        else:
            # LTX-2 text encoder requires several configuration files
            required_gemma_files = [
                ("tokenizer.model", "SentencePiece tokenizer"),
                ("preprocessor_config.json", "Preprocessor configuration"),
                ("tokenizer_config.json", "Tokenizer configuration"),
            ]
            for filename, description in required_gemma_files:
                filepath = os.path.join(self.gemma_root, filename)
                if not os.path.exists(filepath):
                    missing.append(
                        f"Gemma {description}: {filepath} "
                        "(the QAT model variant may not include this file)"
                    )
        if missing:
            msg = "Missing model files:\n" + "\n".join(f"  - {m}" for m in missing)
            msg += "\n\nRun 'python download_models.py' to download them."
            raise FileNotFoundError(msg)

    def _load_model(self):
        """Load the LTX-2 model (lazy loading)"""
        if self.pipeline is not None:
            return

        logger.info("Loading LTX-2 model...")
        self._validate_model_files()

        try:
            from ltx_pipelines.distilled import DistilledPipeline

            self.pipeline = DistilledPipeline(
                checkpoint_path=self.checkpoint_path,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=self.gemma_root,
                loras=[],
                fp8transformer=self.enable_fp8,
            )

            logger.info("LTX-2 model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load LTX-2 model: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        num_frames: int = 121,
        width: int = 512,
        height: int = 768,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.0,
        fps: int = 25,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate a video from text prompt using LTX-2.

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
            Path to generated MP4 file
        """
        import tempfile
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

        # Clear GPU cache before generation if enabled
        if self.clear_cache_before_generation:
            self._clear_gpu_cache()
        
        # Load model if not already loaded
        self._load_model()
        
        self._log_gpu_memory("after model load")

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        logger.info(f"Using seed: {seed}")

        logger.info(
            f"Generating video: {width}x{height}, {num_frames} frames, {fps} fps"
        )

        # Create output path securely
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        # Generate video using LTX-2 DistilledPipeline
        # DistilledPipeline returns (video_tensor, audio_tensor) instead of writing to file
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        video, audio = self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(fps),
            images=[],  # No image conditioning
            tiling_config=tiling_config,
            enhance_prompt=False,
        )

        # Encode the video and audio to output file
        encode_video(
            video=video,
            fps=fps,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

        logger.info(f"Video generated: {output_path}")
        return output_path
