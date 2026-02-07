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
        self.pipeline = None
        self.device = self._get_device()
        logger.info(f"VideoGenerator initialized with device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

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
            # LTX-2 text encoder requires tokenizer.model (SentencePiece format)
            tokenizer_model = os.path.join(self.gemma_root, "tokenizer.model")
            if not os.path.exists(tokenizer_model):
                missing.append(
                    f"Gemma tokenizer.model (SentencePiece): {tokenizer_model} "
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

        # Load model if not already loaded
        self._load_model()

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
        self.pipeline(
            prompt=prompt,
            output_path=output_path,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(fps),
        )

        logger.info(f"Video generated: {output_path}")
        return output_path
