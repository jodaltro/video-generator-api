"""
Configuration settings for Video Generator API (LTX-2)
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    app_name: str = "Video Generator API (LTX-2)"
    port: int = 8000
    host: str = "0.0.0.0"

    # Model Settings - LTX-2
    models_dir: str = os.getenv("MODELS_DIR", "/workspace/models")

    # LTX-2 checkpoint (choose one)
    ltx2_checkpoint: str = "ltx-2-19b-distilled-fp8.safetensors"

    # Spatial upscaler (required for two-stage pipelines)
    ltx2_spatial_upscaler: str = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

    # Distilled LoRA (required for TI2VidTwoStagesPipeline)
    ltx2_distilled_lora: str = "ltx-2-19b-distilled-lora-384.safetensors"

    # Gemma text encoder directory
    gemma_dir: str = "gemma-3-12b-it-qat-q4_0-unquantized"

    # HuggingFace repo for downloading models
    hf_repo_id: str = "Lightricks/LTX-2"
    hf_gemma_repo_id: str = "google/gemma-3-12b-it-qat-q4_0-unquantized"
    # Base (non-QAT) Gemma repo for tokenizer.model, which the QAT variant may lack
    hf_gemma_base_repo_id: str = "google/gemma-3-12b-it"

    # Generation Defaults
    default_fps: int = 25
    default_width: int = 512
    default_height: int = 768
    default_num_frames: int = 121
    default_num_inference_steps: int = 40
    default_guidance_scale: float = 3.0

    # Performance Settings
    enable_fp8: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.models_dir, self.ltx2_checkpoint)

    @property
    def spatial_upsampler_path(self) -> str:
        return os.path.join(self.models_dir, self.ltx2_spatial_upscaler)

    @property
    def distilled_lora_path(self) -> str:
        return os.path.join(self.models_dir, self.ltx2_distilled_lora)

    @property
    def gemma_root_path(self) -> str:
        return os.path.join(self.models_dir, self.gemma_dir)


# Global settings instance
settings = Settings()
