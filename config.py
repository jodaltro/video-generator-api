"""
Configuration settings for Video Generator API (Wan2.1)
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    app_name: str = "Video Generator API (Wan2.1)"
    port: int = 8000
    host: str = "0.0.0.0"

    # Model Settings - Wan2.1
    models_dir: str = os.getenv("MODELS_DIR", "/workspace/models")

    # HuggingFace model ID for Wan2.1 text-to-video (1.3B – lightweight)
    hf_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # Generation Defaults (safe for 8GB+ GPU)
    default_fps: int = 16
    default_width: int = 480
    default_height: int = 320
    default_num_frames: int = 33
    default_num_inference_steps: int = 25
    default_guidance_scale: float = 5.0

    # Performance Settings
    enable_cpu_offload: bool = os.getenv("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
    clear_cache_before_generation: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def model_path(self) -> str:
        return os.path.join(self.models_dir, "Wan2.1-T2V-1.3B-Diffusers")


# Global settings instance
settings = Settings()
