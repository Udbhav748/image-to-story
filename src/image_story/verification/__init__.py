"""Verification layer: claim extraction/verification and visual re-checking.

`visual.VisualVerifier` is a forward check (does the image support this claim?).
Backward verification (does the story cover the evidence?) is a V3 concern and
has no implementation yet -- see docs/architecture/roadmap.md.
"""
from .claims import ClaimExtractor, ClaimVerifier
from .visual import VisualVerifier

__all__ = [
    "VisualVerifier",
    "ClaimExtractor",
    "ClaimVerifier",
]
