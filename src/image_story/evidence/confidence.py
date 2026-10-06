"""Confidence banding for evidence.

The banding thresholds live in `EvidenceRecord.__post_init__`; this module is
the single place that answers "what band is this score?" and what a usable
threshold is for a given requirement.
"""
from __future__ import annotations

from ..domain.enums import EvidenceConfidence

HIGH_THRESHOLD = 0.85
MEDIUM_THRESHOLD = 0.5


def classify(confidence: float) -> EvidenceConfidence:
    """Return the confidence band for a raw score."""
    if confidence >= HIGH_THRESHOLD:
        return EvidenceConfidence.HIGH
    if confidence >= MEDIUM_THRESHOLD:
        return EvidenceConfidence.MEDIUM
    return EvidenceConfidence.LOW


def meets(confidence: float, required: EvidenceConfidence) -> bool:
    """Whether a raw score reaches the given band."""
    order = {
        EvidenceConfidence.LOW: 0,
        EvidenceConfidence.MEDIUM: 1,
        EvidenceConfidence.HIGH: 2,
    }
    return order[classify(confidence)] >= order[required]
