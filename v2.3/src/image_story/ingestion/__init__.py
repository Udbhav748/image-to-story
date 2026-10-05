"""Ingestion module for V2.3A: Image collection processing."""
from .validation import ImageValidator
from .hashing import ImageHasher, Deduplicator
from .cache import VisionCache
from .pipeline import IngestionPipeline

__all__ = [
    "ImageValidator",
    "ImageHasher", 
    "Deduplicator",
    "VisionCache",
    "IngestionPipeline",
]