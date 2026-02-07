"""
Configuration settings for Video Generator API
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "Video Generator API"
    port: int = 8000
    host: str = "0.0.0.0"
    
    # Model Settings
    model_id: str = "Lightricks/LTX-Video"
    model_cache_dir: Optional[str] = None
    
    # Generation Defaults
    default_fps: int = 25
    default_width: int = 512
    default_height: int = 512
    default_num_inference_steps: int = 30
    default_guidance_scale: float = 3.0
    
    # Performance Settings
    enable_model_offload: bool = True
    enable_vae_slicing: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
