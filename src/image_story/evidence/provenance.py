"""Evidence provenance helpers.

`IndexedEvidenceRecord` and `SceneEvidenceLink` were defined in
`memory/hierarchical.py` and imported from there by the FAISS store. They are
provenance records, so they live here; `memory/hierarchical.py` re-exports the
names for existing call sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain.schemas import EvidenceRecord


@dataclass
class IndexedEvidenceRecord:
    """Evidence record with full provenance for FAISS indexing."""

    evidence: EvidenceRecord
    image_id: str
    scene_id: str
    frame_id: int
    collection_id: str
    index_position: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.to_dict(),
            "image_id": self.image_id,
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "collection_id": self.collection_id,
            "index_position": self.index_position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexedEvidenceRecord:
        return cls(
            evidence=EvidenceRecord.from_dict(data["evidence"]),
            image_id=data["image_id"],
            scene_id=data["scene_id"],
            frame_id=data["frame_id"],
            collection_id=data["collection_id"],
            index_position=data.get("index_position", -1),
        )


@dataclass
class SceneEvidenceLink:
    """Link between scene and evidence for provenance tracking."""

    scene_id: str
    evidence_id: str
    evidence_index: int  # position in FAISS index
    relevance_score: float = 1.0
