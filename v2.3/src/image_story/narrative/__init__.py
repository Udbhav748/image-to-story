"""Narrative module initialization."""
from .character import CharacterProfile, CharacterSystem
from .conflict import ConflictEngine
from .humor import HumorEngine
from .surprise import SurpriseEngine
from .planner import CreativePlanner
from .prompts import (
    SYSTEM_PROMPT,
    build_single_image_prompt,
    build_multi_image_prompt,
    build_beat_prompt,
    VERIFICATION_PROMPT,
)

__all__ = [
    "CharacterProfile",
    "CharacterSystem",
    "ConflictEngine",
    "HumorEngine",
    "SurpriseEngine",
    "CreativePlanner",
    "SYSTEM_PROMPT",
    "build_single_image_prompt",
    "build_multi_image_prompt",
    "build_beat_prompt",
    "VERIFICATION_PROMPT",
]