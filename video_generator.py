"""
Video Generator Service using Wan2.1 Model

Uses the lightweight Wan2.1-T2V-1.3B model via HuggingFace diffusers.
Supports text-to-video generation with significantly lower memory requirements
compared to LTX-2 (~4GB vs ~30GB).
"""
import os
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Video generator using Wan2.1-T2V-1.3B model"""

    def __init__(
        self,
        model_path: str,
        hf_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        enable_cpu_offload: bool = False,
        clear_cache_before_generation: bool = True,
    ):
        """
        Initialize the video generator with Wan2.1 model.

        Args:
            model_path: Local path where the model is cached
            hf_model_id: HuggingFace model ID for Wan2.1
            enable_cpu_offload: Whether to enable CPU offloading for lower VRAM usage
            clear_cache_before_generation: Whether to clear GPU cache before generation
        """
        self.model_path = model_path
        self.hf_model_id = hf_model_id
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

    def _load_model(self):
        """Load the Wan2.1 model (lazy loading)"""
        if self.pipeline is not None:
            return

        logger.info("Loading Wan2.1-T2V-1.3B model...")

        try:
            from diffusers import AutoencoderKLWan, WanPipeline
            from diffusers.utils import export_to_video  # noqa: F401

            # Determine the source: use local cache if available, otherwise download
            model_source = self.model_path if os.path.isdir(self.model_path) else self.hf_model_id

            # Use full precision on CPU to avoid poor outputs (brown/static frames)
            model_dtype = torch.float16 if self.device != "cpu" else torch.float32

            # Load the VAE in float32 for better quality
            vae = AutoencoderKLWan.from_pretrained(
                model_source,
                subfolder="vae",
                torch_dtype=torch.float32,
            )

            # Load the full pipeline
            self.pipeline = WanPipeline.from_pretrained(
                model_source,
                vae=vae,
                torch_dtype=model_dtype,
            )

            # Fix UMT5 text encoder weight tying: the shared.weight must be
            # tied to encoder.embed_tokens.weight.  Without this, the text
            # encoder cannot convert tokens into embeddings and the generated
            # video will not reflect the prompt.
            if hasattr(self.pipeline, "text_encoder") and self.pipeline.text_encoder is not None:
                text_encoder = self.pipeline.text_encoder
                try:
                    if hasattr(text_encoder, "shared") and hasattr(text_encoder, "encoder"):
                        # Ensure encoder embeddings share the pretrained "shared" weights
                        text_encoder.encoder.embed_tokens = text_encoder.shared
                        logger.info("Text encoder embeddings synced from shared weights")
                except Exception as sync_error:
                    logger.warning(
                        f"Failed to sync text encoder embeddings "
                        f"({type(sync_error).__name__}): {sync_error}"
                    )

                if hasattr(text_encoder, "tie_weights"):
                    text_encoder.tie_weights()
                    logger.info("Text encoder weights tied successfully")

            # Enable CPU offloading if requested (saves VRAM, CUDA only)
            if self.enable_cpu_offload and self.device == "cuda":
                logger.info("Enabling model CPU offloading...")
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)

            logger.info("Wan2.1-T2V-1.3B model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load Wan2.1 model: {e}")
            raise

    # Default negative prompt to steer the model away from common artifacts
    DEFAULT_NEGATIVE_PROMPT = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, "
        "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "still picture, messy background, three legs, many people in the background, "
        "walking backwards"
    )

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 33,
        width: int = 480,
        height: int = 320,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        fps: int = 16,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate a video from text prompt using Wan2.1.

        Args:
            prompt: Text description of the video
            negative_prompt: Text describing what to avoid in the video.
                Uses a sensible default if not provided.
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
        from diffusers.utils import export_to_video

        # Clear GPU cache before generation if enabled
        if self.clear_cache_before_generation:
            self._clear_gpu_cache()

        # Load model if not already loaded
        self._load_model()

        self._log_gpu_memory("after model load")

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        logger.info(f"Using seed: {seed}")

        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT

        logger.info(
            f"Generating video: {width}x{height}, {num_frames} frames, {fps} fps"
        )

        # Create output path securely
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        # Generate video using Wan2.1 pipeline
        generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # Export video frames to MP4
        export_to_video(output.frames[0], output_path, fps=fps)

        logger.info(f"Video generated: {output_path}")
        return output_path
