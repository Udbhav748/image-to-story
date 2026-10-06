"""Hierarchical memory schemas for scene and collection-level organization."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import uuid
import time
import numpy as np

from ..domain.schemas import EvidenceRecord, WorldEntity, BoundingBox
from ..domain.enums import EvidenceType, InformationClass


@dataclass
class SceneSummary:
    """Compact representation of a scene with provenance."""
    scene_id: str
    image_ids: list[str] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    dominant_entities: list[str] = field(default_factory=list)
    important_objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    location: str = ""
    environment: str = ""
    recurring_entities: list[str] = field(default_factory=list)
    key_evidence_ids: list[str] = field(default_factory=list)
    summary_text: str = ""
    embedding: np.ndarray | None = None
    evidence_count: int = 0
    start_frame: int = 0
    end_frame: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "image_ids": self.image_ids,
            "frame_indices": self.frame_indices,
            "dominant_entities": self.dominant_entities,
            "important_objects": self.important_objects,
            "actions": self.actions,
            "location": self.location,
            "environment": self.environment,
            "recurring_entities": self.recurring_entities,
            "key_evidence_ids": self.key_evidence_ids,
            "summary_text": self.summary_text,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "evidence_count": self.evidence_count,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneSummary:
        embedding = None
        if "embedding" in data and data["embedding"] is not None:
            embedding = np.array(data["embedding"], dtype=np.float32)
        return cls(
            scene_id=data["scene_id"],
            image_ids=data.get("image_ids", []),
            frame_indices=data.get("frame_indices", []),
            dominant_entities=data.get("dominant_entities", []),
            important_objects=data.get("important_objects", []),
            actions=data.get("actions", []),
            location=data.get("location", ""),
            environment=data.get("environment", ""),
            recurring_entities=data.get("recurring_entities", []),
            key_evidence_ids=data.get("key_evidence_ids", []),
            summary_text=data.get("summary_text", ""),
            embedding=embedding,
            evidence_count=data.get("evidence_count", 0),
            start_frame=data.get("start_frame", 0),
            end_frame=data.get("end_frame", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class CollectionMemory:
    """Collection-level memory aggregating scenes and global state."""
    collection_id: str
    scene_summaries: list[SceneSummary] = field(default_factory=list)
    global_entities: dict[str, "EntityMemory"] = field(default_factory=dict)
    state_transitions: list["StateTransition"] = field(default_factory=list)
    narrative_elements: list["NarrativeElement"] = field(default_factory=list)
    total_images: int = 0
    total_evidence: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "scene_summaries": [s.to_dict() for s in self.scene_summaries],
            "global_entities": {k: v.to_dict() for k, v in self.global_entities.items()},
            "state_transitions": [t.to_dict() for t in self.state_transitions],
            "narrative_elements": [n.to_dict() for n in self.narrative_elements],
            "total_images": self.total_images,
            "total_evidence": self.total_evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def get_scene(self, scene_id: str) -> SceneSummary | None:
        for scene in self.scene_summaries:
            if scene.scene_id == scene_id:
                return scene
        return None
    
    def get_scenes_by_entity(self, entity_label: str) -> list[SceneSummary]:
        return [s for s in self.scene_summaries if entity_label in s.dominant_entities or entity_label in s.recurring_entities]


@dataclass
class EntityMemory:
    """Persistent entity memory across scenes."""
    entity_id: str
    normalized_label: str
    aliases: list[str] = field(default_factory=list)
    entity_type: str = "object"  # character, object, location
    first_seen_scene: str = ""
    last_seen_scene: str = ""
    scene_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    state: str = "present"  # present, disappeared, reappeared, moved
    attributes: dict[str, Any] = field(default_factory=dict)
    bounding_boxes: dict[str, list[float]] = field(default_factory=dict)  # scene_id -> bbox
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "normalized_label": self.normalized_label,
            "aliases": self.aliases,
            "entity_type": self.entity_type,
            "first_seen_scene": self.first_seen_scene,
            "last_seen_scene": self.last_seen_scene,
            "scene_ids": self.scene_ids,
            "image_ids": self.image_ids,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
            "state": self.state,
            "attributes": self.attributes,
            "bounding_boxes": self.bounding_boxes,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityMemory:
        return cls(
            entity_id=data["entity_id"],
            normalized_label=data["normalized_label"],
            aliases=data.get("aliases", []),
            entity_type=data.get("entity_type", "object"),
            first_seen_scene=data.get("first_seen_scene", ""),
            last_seen_scene=data.get("last_seen_scene", ""),
            scene_ids=data.get("scene_ids", []),
            image_ids=data.get("image_ids", []),
            evidence_ids=data.get("evidence_ids", []),
            confidence=data.get("confidence", 0.0),
            state=data.get("state", "present"),
            attributes=data.get("attributes", {}),
            bounding_boxes=data.get("bounding_boxes", {}),
        )


@dataclass
class StateTransition:
    """Detected state change for narrative relevance."""
    transition_id: str = field(default_factory=lambda: f"trans_{uuid.uuid4().hex[:8]}")
    entity_id: str = ""
    entity_label: str = ""
    transition_type: str = ""  # appeared, disappeared, moved, changed_scene, changed_relationship, changed_state
    from_scene: str = ""
    to_scene: str = ""
    from_frame: int = -1
    to_frame: int = -1
    description: str = ""
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "entity_id": self.entity_id,
            "entity_label": self.entity_label,
            "transition_type": self.transition_type,
            "from_scene": self.from_scene,
            "to_scene": self.to_scene,
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "description": self.description,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "timestamp": self.timestamp,
        }


@dataclass
class NarrativeElement:
    """Memorable narrative element for callback retrieval."""
    element_id: str = field(default_factory=lambda: f"narr_{uuid.uuid4().hex[:8]}")
    element_type: str = ""  # object, event, motif, character_detail, open_loop
    label: str = ""
    description: str = ""
    first_scene: str = ""
    last_scene: str = ""
    scene_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    recurrence_count: int = 0
    narrative_importance: float = 0.0  # 0-1 score
    associated_entities: list[str] = field(default_factory=list)
    potential_callback: bool = False
    callback_payoff_scene: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "label": self.label,
            "description": self.description,
            "first_scene": self.first_scene,
            "last_scene": self.last_scene,
            "scene_ids": self.scene_ids,
            "evidence_ids": self.evidence_ids,
            "recurrence_count": self.recurrence_count,
            "narrative_importance": self.narrative_importance,
            "associated_entities": self.associated_entities,
            "potential_callback": self.potential_callback,
            "callback_payoff_scene": self.callback_payoff_scene,
        }


@dataclass
class SceneEvidenceLink:
    """Link between scene and evidence for provenance tracking."""
    scene_id: str
    evidence_id: str
    evidence_index: int  # position in FAISS index
    relevance_score: float = 1.0


@dataclass
class RetrievalResult:
    """Result of hierarchical retrieval."""
    scenes: list[SceneSummary] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    entities: list[EntityMemory] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    narrative_elements: list[NarrativeElement] = field(default_factory=list)
    query: str = ""
    retrieval_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for artifact persistence (provenance chain)."""
        return {
            "scenes": [s.to_dict() for s in self.scenes],
            "evidence": [e.to_dict() for e in self.evidence],
            "entities": [e.to_dict() for e in self.entities],
            "transitions": [t.to_dict() for t in self.transitions],
            "narrative_elements": [n.to_dict() for n in self.narrative_elements],
            "query": self.query,
            "retrieval_time_ms": self.retrieval_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        return cls(
            scenes=[SceneSummary.from_dict(s) for s in data.get("scenes", [])],
            evidence=[EvidenceRecord.from_dict(e) for e in data.get("evidence", [])],
            entities=[EntityMemory.from_dict(e) for e in data.get("entities", [])],
            query=data.get("query", ""),
            retrieval_time_ms=data.get("retrieval_time_ms", 0.0),
        )


class MemoryLevel(str):
    """Memory abstraction levels."""
    EVIDENCE = "evidence"
    ENTITY = "entity"
    SCENE = "scene"
    EVENT = "event"
    NARRATIVE = "narrative"
    COLLECTION = "collection"


# Provenance-preserving evidence record extension
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


# Type alias for retrieval filters
RetrievalFilter = dict[str, Any]


def create_retrieval_filter(
    scene_ids: list[str] | None = None,
    entity_labels: list[str] | None = None,
    evidence_types: list[str] | None = None,
    information_classes: list[str] | None = None,
    min_confidence: float | None = None,
    frame_range: tuple[int, int] | None = None,
) -> RetrievalFilter:
    """Create a structured retrieval filter."""
    filter_dict = {}
    if scene_ids:
        filter_dict["scene_ids"] = scene_ids
    if entity_labels:
        filter_dict["entity_labels"] = entity_labels
    if evidence_types:
        filter_dict["evidence_types"] = evidence_types
    if information_classes:
        filter_dict["information_classes"] = information_classes
    if min_confidence is not None:
        filter_dict["min_confidence"] = min_confidence
    if frame_range:
        filter_dict["frame_range"] = frame_range
    return filter_dict


def apply_retrieval_filter(evidence: EvidenceRecord, filter_dict: RetrievalFilter) -> bool:
    """Apply retrieval filter to evidence record."""
    if not filter_dict:
        return True
    
    if "entity_labels" in filter_dict:
        if evidence.entity not in filter_dict["entity_labels"]:
            return False
    
    if "evidence_types" in filter_dict:
        if evidence.type.value not in filter_dict["evidence_types"]:
            return False
    
    if "information_classes" in filter_dict:
        if evidence.information_class.value not in filter_dict["information_classes"]:
            return False
    
    if "min_confidence" in filter_dict:
        if evidence.confidence < filter_dict["min_confidence"]:
            return False
    
    if "frame_range" in filter_dict:
        start, end = filter_dict["frame_range"]
        if not (start <= evidence.frame_id <= end):
            return False
    
    return True