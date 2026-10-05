"""Collections module for V2.3A: Scalable image collection processing."""
from .pipeline import (
    CollectionPipeline,
    create_story_session,
    CollectionPipelineConfig,
    ProcessingConfig,
)

__all__ = [
    "CollectionPipeline",
    "create_story_session",
    "CollectionPipelineConfig",
    "ProcessingConfig",
]