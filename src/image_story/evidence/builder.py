"""Evidence utilities: canonical text projection, provenance and confidence.

`evidence_to_text` was previously duplicated in four modules with three
different formats. There is now one function; callers that need a different
rendering pass explicit arguments.
"""
from __future__ import annotations

from ..domain.schemas import EvidenceRecord


def evidence_to_text(
    evidence: EvidenceRecord,
    *,
    include_frame: bool = True,
    include_confidence: bool = False,
    separator: str = " | ",
    text_limit: int | None = None,
) -> str:
    """Render an evidence record as text.

    The canonical projection used for embedding and for context building.

    Args:
        include_frame: append the frame index (useful for retrieval keys).
        include_confidence: append the confidence band.
        separator: joins the parts.
        text_limit: truncate `evidence_text` to this many characters.
    """
    parts: list[str] = []
    if evidence.entity:
        parts.append(f"Entity: {evidence.entity}")
    if evidence.type:
        parts.append(f"Type: {evidence.type.value}")
    if evidence.action:
        parts.append(f"Action: {evidence.action}")
    if evidence.relationship:
        parts.append(f"Relation: {evidence.relationship}")
    if evidence.evidence_text:
        parts.append(evidence.evidence_text[:text_limit] if text_limit else evidence.evidence_text)
    if include_frame and evidence.frame_id >= 0:
        parts.append(f"Frame: {evidence.frame_id}")
    if include_confidence and evidence.confidence_class:
        parts.append(f"Confidence: {evidence.confidence_class.value}")
    return separator.join(parts)
