"""Settings and mode-preset helpers for the Image -> Story system."""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..domain.enums import PipelineMode
from ..domain.schemas import PipelineConfig


@dataclass
class Settings:
    """Process-level settings (paths, device, offline mode, seed)."""

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
    benchmarks_dir: str = "benchmarks"

    # Reproducibility
    default_seed: int = 0

    def apply_env(self) -> None:
        """Apply environment variables."""
        os.environ.setdefault("HF_HOME", self.hf_home)
        if self.hf_hub_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if self.transformers_offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


#: Global settings instance
settings = Settings()


def get_pipeline_config(mode: str | PipelineMode = "standard") -> PipelineConfig:
    """Get pipeline configuration for a mode."""
    if isinstance(mode, str):
        mode = PipelineMode(mode)
    return PipelineConfig.from_mode(mode.value)


def load_config_from_yaml(path: str) -> PipelineConfig:
    """Load pipeline config from a YAML file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return PipelineConfig(**data)
