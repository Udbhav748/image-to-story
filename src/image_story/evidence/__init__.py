"""Evidence layer: text projection, provenance records and confidence banding."""
from .builder import evidence_to_text
from .confidence import classify as classify_confidence
from .confidence import meets as confidence_meets
from .provenance import IndexedEvidenceRecord, SceneEvidenceLink

__all__ = [
    "evidence_to_text",
    "IndexedEvidenceRecord",
    "SceneEvidenceLink",
    "classify_confidence",
    "confidence_meets",
]
