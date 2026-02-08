"""
Video Generator Service using LTX-2 Model

Uses the official LTX-2 pipelines from the Lightricks/LTX-2 repository.
Supports text-to-video generation with the DistilledPipeline for fast inference.

Memory optimization modes:
- Standard: Uses DistilledPipeline directly (~23GB GPU for RTX 4090)
- Aggressive offload: Manually orchestrates pipeline stages with sequential
  CPU↔GPU model transfers to reduce peak GPU usage (~12-14GB)
"""
import gc
import os
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def _cleanup_memory():
    """Force garbage collection and clear GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
        enable_aggressive_offload: bool = False,
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
            enable_cpu_offload: Whether to enable basic CPU offloading (limited with DistilledPipeline)
            enable_aggressive_offload: Whether to enable aggressive sequential CPU↔GPU offloading.
                Significantly reduces peak GPU memory (~12-14GB instead of ~23GB) at the cost of
                slower generation due to model transfers between CPU and GPU.
            clear_cache_before_generation: Whether to clear GPU cache before each generation
        """
        self.checkpoint_path = checkpoint_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.gemma_root = gemma_root
        self.distilled_lora_path = distilled_lora_path
        self.enable_fp8 = enable_fp8
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_aggressive_offload = enable_aggressive_offload
        self.clear_cache_before_generation = clear_cache_before_generation
        self.pipeline = None
        self._model_ledger = None
        self.device = self._get_device()
        logger.info(f"VideoGenerator initialized with device: {self.device}")
        logger.info(
            f"CPU offload: {enable_cpu_offload}, "
            f"Aggressive offload: {enable_aggressive_offload}, "
            f"Clear cache: {clear_cache_before_generation}"
        )
        
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
        return self.pipeline is not None or self._model_ledger is not None

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
        if self.enable_aggressive_offload:
            self._load_model_aggressive()
        else:
            self._load_model_standard()

    def _load_model_standard(self):
        """Load the LTX-2 model using the standard DistilledPipeline"""
        if self.pipeline is not None:
            return

        logger.info("Loading LTX-2 model (standard mode)...")
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
            logger.error(f"Failed to load LTX-2 model: {e}")
            raise

    def _load_model_aggressive(self):
        """
        Load the ModelLedger for aggressive offloading mode.

        In this mode, we don't create a DistilledPipeline. Instead, we keep
        only the ModelLedger (which holds builders, not loaded models) and
        build each model component on-demand, moving it to GPU only when
        needed and back to CPU/deleted when done.
        """
        if self._model_ledger is not None:
            return

        logger.info("Loading LTX-2 model (aggressive offload mode)...")
        logger.info(
            "Models will be loaded to GPU one at a time and offloaded after use. "
            "This reduces peak GPU memory but increases generation time."
        )
        self._validate_model_files()

        try:
            from ltx_pipelines.utils import ModelLedger

            self._model_ledger = ModelLedger(
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                checkpoint_path=self.checkpoint_path,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root_path=self.gemma_root,
                loras=[],
                fp8transformer=self.enable_fp8,
            )

            logger.info("LTX-2 ModelLedger initialized (aggressive offload mode)")
        except Exception as e:
            logger.error(f"Failed to initialize ModelLedger: {e}")
            raise

    @torch.inference_mode()
    def _generate_aggressive_offload(
        self,
        prompt: str,
        num_frames: int,
        width: int,
        height: int,
        fps: int,
        seed: int,
        tiling_config,
    ):
        """
        Generate video with aggressive sequential CPU↔GPU offloading.

        Replicates the DistilledPipeline logic but moves each model component
        to GPU only when needed, then deletes it to free GPU memory before
        loading the next component. This reduces peak GPU memory from ~23GB
        to ~12-14GB at the cost of slower generation.

        Pipeline stages:
        1. Text encoding (Gemma on GPU → encode → delete)
        2. Stage 1 denoising (video_encoder + transformer on GPU → denoise → keep for stage 2)
        3. Upsampling (spatial_upsampler on GPU → upsample → delete)
        4. Stage 2 denoising (reuse transformer + video_encoder → denoise → delete both)
        5. Video decoding (video_decoder on GPU → decode → delete)
        6. Audio decoding (audio_decoder + vocoder on GPU → decode → delete)
        """
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import (
            DISTILLED_SIGMA_VALUES,
            STAGE_2_DISTILLED_SIGMA_VALUES,
        )
        from ltx_pipelines.utils.helpers import (
            assert_resolution,
            denoise_audio_video,
            euler_denoising_loop,
            image_conditionings_by_replacing_latent,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.types import PipelineComponents

        gpu = torch.device("cuda")
        dtype = torch.bfloat16

        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=gpu).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        pipeline_components = PipelineComponents(dtype=dtype, device=gpu)

        # --- Stage: Text Encoding (load text encoder to GPU, encode, then free) ---
        logger.info("[Aggressive Offload] Loading text encoder to GPU...")
        self._log_gpu_memory("before text encoder")
        text_encoder = self._model_ledger.text_encoder()
        text_encoder = text_encoder.to(gpu)
        self._log_gpu_memory("after text encoder load")

        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        torch.cuda.synchronize()
        del text_encoder
        _cleanup_memory()
        logger.info("[Aggressive Offload] Text encoder freed from GPU")
        self._log_gpu_memory("after text encoder free")

        # --- Stage 1: Initial low-res video generation ---
        logger.info("[Aggressive Offload] Loading video encoder to GPU...")
        video_encoder = self._model_ledger.video_encoder()
        video_encoder = video_encoder.to(gpu)
        self._log_gpu_memory("after video encoder load")

        logger.info("[Aggressive Offload] Loading transformer to GPU...")
        transformer = self._model_ledger.transformer()
        transformer = transformer.to(gpu)
        self._log_gpu_memory("after transformer load")

        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(gpu)

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=float(fps),
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=[],
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=gpu,
        )

        logger.info("[Aggressive Offload] Running Stage 1 denoising...")
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=pipeline_components,
            dtype=dtype,
            device=gpu,
        )
        self._log_gpu_memory("after stage 1")

        # --- Upsampling: offload transformer, load upsampler ---
        logger.info("[Aggressive Offload] Offloading transformer to CPU for upsampling...")
        transformer = transformer.to(torch.device("cpu"))
        _cleanup_memory()
        self._log_gpu_memory("after transformer offload")

        logger.info("[Aggressive Offload] Loading spatial upsampler to GPU...")
        spatial_upsampler = self._model_ledger.spatial_upsampler()
        spatial_upsampler = spatial_upsampler.to(gpu)
        self._log_gpu_memory("after upsampler load")

        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=spatial_upsampler,
        )

        # Free upsampler
        del spatial_upsampler
        _cleanup_memory()
        logger.info("[Aggressive Offload] Spatial upsampler freed")
        self._log_gpu_memory("after upsampler free")

        # --- Stage 2: Refine at higher resolution ---
        logger.info("[Aggressive Offload] Moving transformer back to GPU for Stage 2...")
        transformer = transformer.to(gpu)
        self._log_gpu_memory("after transformer reload")

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(gpu)
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=float(fps),
        )
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=[],
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=gpu,
        )

        logger.info("[Aggressive Offload] Running Stage 2 denoising...")
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=pipeline_components,
            dtype=dtype,
            device=gpu,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )
        self._log_gpu_memory("after stage 2")

        # Free transformer and video encoder
        del transformer
        del video_encoder
        _cleanup_memory()
        logger.info("[Aggressive Offload] Transformer and video encoder freed")
        self._log_gpu_memory("after denoising cleanup")

        # --- Decode video ---
        logger.info("[Aggressive Offload] Loading video decoder to GPU...")
        video_decoder = self._model_ledger.video_decoder()
        video_decoder = video_decoder.to(gpu)
        self._log_gpu_memory("after video decoder load")

        decoded_video = vae_decode_video(
            video_state.latent, video_decoder, tiling_config, generator
        )

        del video_decoder
        _cleanup_memory()
        logger.info("[Aggressive Offload] Video decoder freed")
        self._log_gpu_memory("after video decode")

        # --- Decode audio ---
        logger.info("[Aggressive Offload] Loading audio decoder and vocoder to GPU...")
        audio_decoder = self._model_ledger.audio_decoder()
        audio_decoder = audio_decoder.to(gpu)
        vocoder = self._model_ledger.vocoder()
        vocoder = vocoder.to(gpu)
        self._log_gpu_memory("after audio decoder load")

        decoded_audio = vae_decode_audio(
            audio_state.latent, audio_decoder, vocoder
        )

        del audio_decoder
        del vocoder
        _cleanup_memory()
        logger.info("[Aggressive Offload] Audio decoder and vocoder freed")
        self._log_gpu_memory("after audio decode")

        return decoded_video, decoded_audio

    def generate(
        self,
        prompt: str,
        num_frames: int = 33,
        width: int = 320,
        height: int = 512,
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

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        if self.enable_aggressive_offload:
            # Use aggressive offloading: manually orchestrate pipeline stages
            # with sequential CPU↔GPU model transfers
            logger.info("Using aggressive offload mode for reduced GPU memory")
            video, audio = self._generate_aggressive_offload(
                prompt=prompt,
                num_frames=num_frames,
                width=width,
                height=height,
                fps=fps,
                seed=seed,
                tiling_config=tiling_config,
            )
        else:
            # Use standard DistilledPipeline
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
