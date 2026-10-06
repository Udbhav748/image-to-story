"""Vision layer: Florence-2 perception, GroundingDINO grounding and OCR.

Claim verification against images is `verification.visual.VisualVerifier`, not a
vision-model responsibility.
"""
from .base import VisionModel, VisionModelRegistry
from .florence import Florence2Model
from .grounding import GroundingDINOModel, create_grounding_dino
from .ocr import OCRModel, create_ocr

__all__ = [
    "VisionModel",
    "VisionModelRegistry",
    "Florence2Model",
    "GroundingDINOModel",
    "create_grounding_dino",
    "OCRModel",
    "create_ocr",
]
