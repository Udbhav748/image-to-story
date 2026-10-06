"""Story generation: generator abstraction, Qwen adapter and prompt templates."""
from .base import QwenGenerator, StoryGenerationPipeline, StoryGenerator, create_generator
from .prompts import (
    SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
    build_beat_prompt,
    build_multi_image_prompt,
    build_single_image_prompt,
)

__all__ = [
    "StoryGenerator",
    "QwenGenerator",
    "StoryGenerationPipeline",
    "create_generator",
    "SYSTEM_PROMPT",
    "VERIFICATION_PROMPT",
    "build_single_image_prompt",
    "build_multi_image_prompt",
    "build_beat_prompt",
]
