"""Configuration settings for the Image → Story V2 system."""
import os
from dataclasses import dataclass, field
from typing import Any

from ..domain.schemas import PipelineConfig
from ..domain.enums import PipelineMode


@dataclass
class Settings:
    """Global settings."""
    
    # Model cache
    hf_home: str = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_hub_offline: bool = True
    transformers_offline: bool = True
    
    # Default device
    device: str = "cpu"
    
    # Default pipeline config
    default_mode: PipelineMode = PipelineMode.STANDARD
    
    # Paths
    artifacts_dir: str = "artifacts"
    configs_dir: str = "configs"
    evals_dir: str = "evals"
    
    # Reproducibility
    default_seed: int = 0
    
    def apply_env(self) -> None:
        """Apply environment variables."""
        os.environ.setdefault("HF_HOME", self.hf_home)
        if self.hf_hub_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if self.transformers_offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# Global settings instance
settings = Settings()


def get_pipeline_config(mode: str | PipelineMode = "standard") -> PipelineConfig:
    """Get pipeline configuration for a mode."""
    if isinstance(mode, str):
        mode = PipelineMode(mode)
    return PipelineConfig.from_mode(mode.value)


def load_config_from_yaml(path: str) -> PipelineConfig:
    """Load pipeline config from YAML file."""
    import yaml
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return PipelineConfig(**data)