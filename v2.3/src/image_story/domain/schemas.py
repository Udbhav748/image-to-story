"""Domain schemas for the Image → Story V2 system.

Structured, typed data contracts for all pipeline stages.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Literal
import uuid
import time
from datetime import datetime

from .enums import (
    EvidenceType,
    EvidenceConfidence,
    InformationClass,
    SourceModel,
)


@dataclass
class BoundingBox:
    """Bounding box in [x1, y1, x2, y2] format."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, lst: list[float]) -> BoundingBox:
        return cls(lst[0], lst[1], lst[2], lst[3])


@dataclass
class EvidenceRecord:
    """A single piece of visual evidence with full provenance."""
    id: str = field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}")
    entity: str = ""
    type: EvidenceType = EvidenceType.OBJECT
    action: str | None = None
    relationship: str | None = None
    frame_id: int = 0
    bbox: BoundingBox | None = None
    confidence: float = 0.0
    confidence_class: EvidenceConfidence = EvidenceConfidence.LOW
    source: SourceModel = SourceModel.FLORENCE2
    evidence_text: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence_position: int = 0
    information_class: InformationClass = InformationClass.HARD_FACT
    
    def __post_init__(self):
        if self.confidence >= 0.85:
            self.confidence_class = EvidenceConfidence.HIGH
        elif self.confidence >= 0.5:
            self.confidence_class = EvidenceConfidence.MEDIUM
        else:
            self.confidence_class = EvidenceConfidence.LOW
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity": self.entity,
            "type": self.type.value,
            "action": self.action,
            "relationship": self.relationship,
            "frame_id": self.frame_id,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "confidence": self.confidence,
            "confidence_class": self.confidence_class.value,
            "source": self.source.value,
            "evidence_text": self.evidence_text,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "sequence_position": self.sequence_position,
            "information_class": self.information_class.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRecord:
        bbox = BoundingBox.from_list(data["bbox"]) if data.get("bbox") else None
        return cls(
            id=data.get("id", f"obs_{uuid.uuid4().hex[:8]}"),
            entity=data.get("entity", ""),
            type=EvidenceType(data.get("type", "object")),
            action=data.get("action"),
            relationship=data.get("relationship"),
            frame_id=data.get("frame_id", 0),
            bbox=bbox,
            confidence=data.get("confidence", 0.0),
            source=SourceModel(data.get("source", "florence2")),
            evidence_text=data.get("evidence_text", ""),
            provenance=data.get("provenance", {}),
            timestamp=data.get("timestamp", time.time()),
            sequence_position=data.get("sequence_position", 0),
            information_class=InformationClass(data.get("information_class", "hard_fact")),
        )


@dataclass
class VisualObservations:
    """Structured output from the vision perception stage."""
    image_id: str
    frame_id: int
    scene: str = ""
    detailed_caption: str = ""
    objects: list[str] = field(default_factory=list)
    od_labels: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    spatial_relations: list[str] = field(default_factory=list)
    region_descriptions: list[str] = field(default_factory=list)
    style_or_mood: str = ""
    ocr_text: str = ""
    grounding_detections: list[dict[str, Any]] = field(default_factory=list)
    evidence_records: list[EvidenceRecord] = field(default_factory=list)
    runtime_s: float = 0.0
    model_load_s: float = 0.0
    models_used: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "frame_id": self.frame_id,
            "scene": self.scene,
            "detailed_caption": self.detailed_caption,
            "objects": self.objects,
            "od_labels": self.od_labels,
            "characters": self.characters,
            "actions": self.actions,
            "spatial_relations": self.spatial_relations,
            "region_descriptions": self.region_descriptions,
            "style_or_mood": self.style_or_mood,
            "ocr_text": self.ocr_text,
            "grounding_detections": self.grounding_detections,
            "evidence_records": [e.to_dict() for e in self.evidence_records],
            "runtime_s": self.runtime_s,
            "model_load_s": self.model_load_s,
            "models_used": self.models_used,
        }


@dataclass
class SceneSummary:
    """Summary of a scene grouping multiple observations."""
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
    embedding: Any = None
    evidence_count: int = 0
    start_frame: int = 0
    end_frame: int = 0
    timestamp: float = 0.0


@dataclass
class WorldEntity:
    """An entity tracked across frames."""
    id: str
    label: str
    entity_type: str  # person, object, location
    first_frame: int
    last_frame: int
    frames_present: list[int] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    bounding_boxes: dict[int, BoundingBox] = field(default_factory=dict)
    associated_entities: list[str] = field(default_factory=list)
    is_recurring: bool = False
    disappearance_frame: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "entity_type": self.entity_type,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames_present": self.frames_present,
            "attributes": self.attributes,
            "bounding_boxes": {k: v.to_list() for k, v in self.bounding_boxes.items()},
            "associated_entities": self.associated_entities,
            "is_recurring": self.is_recurring,
            "disappearance_frame": self.disappearance_frame,
        }


@dataclass
class WorldState:
    """Persistent world state across multiple frames."""
    characters: list[WorldEntity] = field(default_factory=list)
    objects: list[WorldEntity] = field(default_factory=list)
    locations: list[WorldEntity] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    open_loops: list[dict[str, Any]] = field(default_factory=list)
    previous_events: list[dict[str, Any]] = field(default_factory=list)
    frame_count: int = 0
    
    def get_all_entities(self) -> list[WorldEntity]:
        return self.characters + self.objects + self.locations
    
    def get_entity_by_label(self, label: str) -> WorldEntity | None:
        for entity in self.get_all_entities():
            if entity.label.lower() == label.lower():
                return entity
        return None
    
    def get_disappeared_entities(self) -> list[WorldEntity]:
        """Get entities that have disappeared (disappearance_frame is set)."""
        return [e for e in self.get_all_entities() if e.disappearance_frame is not None]
    
    def get_recurring_entities(self) -> list[WorldEntity]:
        """Get entities that appear in multiple frames."""
        return [e for e in self.get_all_entities() if e.is_recurring]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "characters": [e.to_dict() for e in self.characters],
            "objects": [e.to_dict() for e in self.objects],
            "locations": [e.to_dict() for e in self.locations],
            "goals": self.goals,
            "relationships": self.relationships,
            "open_loops": self.open_loops,
            "previous_events": self.previous_events,
            "frame_count": self.frame_count,
        }


@dataclass
class RetrievedEvidence:
    """Evidence retrieved from FAISS with ranking metadata."""
    record: EvidenceRecord
    semantic_similarity: float = 0.0
    rank_score: float = 0.0
    rank_factors: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "semantic_similarity": self.semantic_similarity,
            "rank_score": self.rank_score,
            "rank_factors": self.rank_factors,
        }


@dataclass
class CreativePlan:
    """Structured creative plan for story generation."""
    genre: str = "whimsical"
    tone: str = "comedic"
    creativity_level: float = 0.7
    surprise_level: float = 0.6
    humor_level: float = 0.5
    mystery_level: float = 0.4
    emotion_level: float = 0.5
    dialogue_level: float = 0.4
    metaphor_level: float = 0.3
    characters: list[dict[str, Any]] = field(default_factory=list)
    central_conflict: dict[str, Any] | None = None
    open_loops: list[str] = field(default_factory=list)
    foreshadowing_elements: list[dict[str, Any]] = field(default_factory=list)
    callback_plan: list[dict[str, Any]] = field(default_factory=list)
    narrative_arc: list[dict[str, Any]] = field(default_factory=list)
    # V2.1: Grounded creativity fields
    hard_facts: list[str] = field(default_factory=list)
    soft_inferences: list[str] = field(default_factory=list)
    locked_facts: list[str] = field(default_factory=list)
    # V2.2: Risk-aware creative budget
    creative_budget: dict[str, Any] = field(default_factory=lambda: {
        "safe_creative": {"max": 8, "used": 0},      # personality, humor, dialogue, metaphor
        "risky_inferred": {"max": 3, "used": 0},    # motivations, uncertain actions
        "forbidden_visual": {"max": 0, "used": 0},  # new objects, colors, materials, people
    })
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "genre": self.genre,
            "tone": self.tone,
            "creativity_level": self.creativity_level,
            "surprise_level": self.surprise_level,
            "humor_level": self.humor_level,
            "mystery_level": self.mystery_level,
            "emotion_level": self.emotion_level,
            "dialogue_level": self.dialogue_level,
            "metaphor_level": self.metaphor_level,
            "characters": self.characters,
            "central_conflict": self.central_conflict,
            "open_loops": self.open_loops,
            "foreshadowing_elements": self.foreshadowing_elements,
            "callback_plan": self.callback_plan,
            "narrative_arc": self.narrative_arc,
            "hard_facts": self.hard_facts,
            "soft_inferences": self.soft_inferences,
            "locked_facts": self.locked_facts,
            "creative_budget": self.creative_budget,
        }


@dataclass
class StoryBeat:
    """A single beat in the story plan."""
    beat_number: int
    beat_type: str  # setup, goal, conflict, escalation, surprise, resolution, callback
    description: str
    key_entities: list[str] = field(default_factory=list)
    key_evidence_ids: list[str] = field(default_factory=list)
    creative_elements: list[str] = field(default_factory=list)
    target_words: int = 30


@dataclass
class StoryPlan:
    """Complete story plan with beats."""
    beats: list[StoryBeat] = field(default_factory=list)
    creative_plan: CreativePlan | None = None
    target_total_words: int = 250
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "beats": [
                {
                    "beat_number": b.beat_number,
                    "beat_type": b.beat_type,
                    "description": b.description,
                    "key_entities": b.key_entities,
                    "key_evidence_ids": b.key_evidence_ids,
                    "creative_elements": b.creative_elements,
                    "target_words": b.target_words,
                }
                for b in self.beats
            ],
            "creative_plan": self.creative_plan.to_dict() if self.creative_plan else None,
            "target_total_words": self.target_total_words,
        }


# V2.3A: Collection and ingestion models
@dataclass
class ImageRecord:
    """A single image record in a collection with validation and deduplication info."""
    image_id: str = field(default_factory=lambda: f"img_{uuid.uuid4().hex[:8]}")
    path: str = ""
    sequence_index: int = 0
    content_hash: str = ""  # SHA256 of file content
    perceptual_hash: str = ""  # perceptual hash for near-duplicate detection
    file_size: int = 0
    width: int = 0
    height: int = 0
    mime_type: str = ""
    validation_status: str = "pending"  # "valid", "invalid", "skipped", "duplicate"
    processing_status: str = "pending"  # "pending", "queued", "processing", "completed", "failed", "skipped", "cancelled"
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Deduplication
    duplicate_of: str | None = None  # image_id of the duplicate original
    similar_to: list[str] = field(default_factory=list)  # list of similar image_ids
    # Processing
    processing_status_detail: str = "pending"
    error_message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if isinstance(self.duplicate_of, str) and self.duplicate_of == "":
            self.duplicate_of = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "path": self.path,
            "sequence_index": self.sequence_index,
            "content_hash": self.content_hash,
            "perceptual_hash": self.perceptual_hash,
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "mime_type": self.mime_type,
            "validation_status": self.validation_status,
            "processing_status": self.processing_status,
            "error": self.error,
            "metadata": self.metadata,
            "duplicate_of": self.duplicate_of,
            "similar_to": self.similar_to,
            "processing_status_detail": self.processing_status_detail,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=data.get("image_id", f"img_{uuid.uuid4().hex[:8]}"),
            path=data.get("path", ""),
            sequence_index=data.get("sequence_index", 0),
            content_hash=data.get("content_hash", ""),
            perceptual_hash=data.get("perceptual_hash", ""),
            file_size=data.get("file_size", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            mime_type=data.get("mime_type", ""),
            validation_status=data.get("validation_status", "pending"),
            processing_status=data.get("processing_status", "pending"),
            error=data.get("error", ""),
            metadata=data.get("metadata", {}),
            duplicate_of=data.get("duplicate_of"),
            similar_to=data.get("similar_to", []),
            processing_status_detail=data.get("processing_status_detail", "pending"),
            error_message=data.get("error_message", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


@dataclass
class ImageCollection:
    """A collection of images for storytelling."""
    collection_id: str = field(default_factory=lambda: f"col_{uuid.uuid4().hex[:8]}")
    images: list[ImageRecord] = field(default_factory=list)
    ordering_mode: str = "auto"  # "upload_order", "filename", "timestamp", "auto", "unordered"
    total_images: int = 0
    valid_images: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        self.total_images = len(self.images)
        self.valid_images = sum(1 for img in self.images if img.validation_status == "valid")
    
    def add_image(self, image: ImageRecord) -> None:
        image.sequence_index = len(self.images)
        self.images.append(image)
        self.total_images = len(self.images)
        self.valid_images = sum(1 for img in self.images if img.validation_status == "valid")
        self.updated_at = datetime.now().isoformat()
    
    def add_images(self, images: list[ImageRecord]) -> None:
        for img in images:
            self.add_image(img)
    
    def get_valid_images(self) -> list[ImageRecord]:
        return [img for img in self.images if img.validation_status == "valid"]
    
    def get_images_by_status(self, status: str) -> list[ImageRecord]:
        return [img for img in self.images if img.processing_status == status]
    
    def get_valid_sorted(self, ordering: str = "auto") -> list[ImageRecord]:
        """Get valid images sorted by the specified ordering mode."""
        valid = self.get_valid_images()
        if ordering == "upload_order" or ordering == "auto":
            return sorted(valid, key=lambda x: x.sequence_index)
        elif ordering == "filename":
            return sorted(valid, key=lambda x: x.path)
        elif ordering == "timestamp":
            # Would need creation time metadata
            return sorted(valid, key=lambda x: x.created_at)
        else:
            return valid
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "images": [img.to_dict() for img in self.images],
            "ordering_mode": self.ordering_mode,
            "total_images": self.total_images,
            "valid_images": self.valid_images,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageCollection":
        images = [ImageRecord.from_dict(img) for img in data.get("images", [])]
        return cls(
            collection_id=data.get("collection_id", f"col_{uuid.uuid4().hex[:8]}"),
            images=images,
            ordering_mode=data.get("ordering_mode", "auto"),
            total_images=data.get("total_images", len(images)),
            valid_images=data.get("valid_images", sum(1 for img in images if img.validation_status == "valid")),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )




@dataclass
class ProcessingJob:
    """A processing job for a single image or batch."""
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:8]}")
    collection_id: str = ""
    image_ids: list[str] = field(default_factory=list)
    status: str = "queued"  # "queued", "processing", "completed", "failed", "skipped", "cancelled"
    stage: str = "queued"  # current stage: validation, hashing, deduplication, vision, etc.
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    retry_count: int = 0
    max_retries: int = 3
    progress: float = 0.0  # 0.0 to 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "collection_id": self.collection_id,
            "image_ids": self.image_ids,
            "status": self.status,
            "stage": self.stage,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingJob":
        return cls(
            job_id=data.get("job_id", f"job_{uuid.uuid4().hex[:8]}"),
            collection_id=data.get("collection_id", ""),
            image_ids=data.get("image_ids", []),
            status=data.get("status", "queued"),
            stage=data.get("stage", "queued"),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            error=data.get("error", ""),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            progress=data.get("progress", 0.0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )




@dataclass
class StorySession:
    """A story generation session for a collection of images."""
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:8]}")
    collection_id: str = ""
    config_id: str = ""  # reference to PipelineConfig
    status: str = "pending"  # "pending", "queued", "processing", "completed", "partial_success", "failed", "cancelled"
    pipeline_mode: str = "standard"
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    artifacts_ref: str = ""  # reference to stored artifacts
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    error: str = ""
    processing_summary: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "collection_id": self.collection_id,
            "config_id": self.config_id,
            "status": self.status,
            "pipeline_mode": self.pipeline_mode,
            "config_snapshot": self.config_snapshot,
            "artifacts_ref": self.artifacts_ref,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "processing_summary": self.processing_summary,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StorySession":
        return cls(
            session_id=data.get("session_id", f"sess_{uuid.uuid4().hex[:8]}"),
            collection_id=data.get("collection_id", ""),
            config_id=data.get("config_id", ""),
            status=data.get("status", "pending"),
            pipeline_mode=data.get("pipeline_mode", "standard"),
            config_snapshot=data.get("config_snapshot", {}),
            artifacts_ref=data.get("artifacts_ref", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at", ""),
            error=data.get("error", ""),
            processing_summary=data.get("processing_summary", {}),
        )


class CacheEntry:
    """Cache entry for vision results."""
    cache_key: str  # content_hash + model_version
    content_hash: str
    model_name: str
    model_version: str
    result: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "content_hash": self.content_hash,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "result": self.result,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            cache_key=data.get("cache_key", ""),
            content_hash=data.get("content_hash", ""),
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", ""),
            result=data.get("result", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            accessed_at=data.get("accessed_at", datetime.now().isoformat()),
            access_count=data.get("access_count", 0),
        )


@dataclass
class ProcessingConfig:
    """Configuration for collection processing."""
    resume: bool = True
    retry_failed: bool = True
    force_reprocess: bool = False
    skip_validation: bool = False
    skip_deduplication: bool = False
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class CollectionPipelineConfig:
    """Configuration for the collection pipeline."""
    mode: str = "standard"
    ordering_mode: str = "auto"
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    cache_dir: str = "cache"
    artifacts_dir: str = "artifacts"
    max_images: int = 1000
    skip_validation: bool = False
    skip_deduplication: bool = False
    max_images_per_batch: int = 10

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["processing"] = asdict(self.processing)
        return result


@dataclass
class StoryDraft:
    """Generated story with metadata."""
    text: str
    story_plan: StoryPlan | None = None
    word_count: int = 0
    generation_time_s: float = 0.0
    model_used: str = "qwen2.5-0.5b-instruct"
    prompt_used: str = ""
    
    def __post_init__(self):
        self.word_count = len(self.text.split())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "story_plan": self.story_plan.to_dict() if self.story_plan else None,
            "word_count": self.word_count,
            "generation_time_s": self.generation_time_s,
            "model_used": self.model_used,
            "prompt_used": self.prompt_used,
        }


@dataclass
class StoryClaim:
    """A claim extracted from the generated story."""
    id: str = field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:8]}")
    subject: str = ""
    relation: str = ""
    object: str = ""
    original_sentence: str = ""
    claim_type: InformationClass = InformationClass.HARD_FACT
    claim_classification: str = "inferred"  # "observed", "inferred", "creative"
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_natural_language(self) -> str:
        return f"The {self.subject} {self.relation} the {self.object}."
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "original_sentence": self.original_sentence,
            "claim_type": self.claim_type.value,
            "claim_classification": self.claim_classification,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
            "natural_language": self.to_natural_language(),
        }


@dataclass
class VerificationResult:
    """Result of verifying a claim against visual evidence."""
    claim_id: str
    claim: StoryClaim
    status: Literal["supported", "unsupported", "contradicted"]
    supporting_evidence: list[EvidenceRecord] = field(default_factory=list)
    contradicting_evidence: list[EvidenceRecord] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim.to_dict(),
            "status": self.status,
            "supporting_evidence_count": len(self.supporting_evidence),
            "contradicting_evidence_count": len(self.contradicting_evidence),
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""
    grounding_score: float = 0.0
    clip_image_story_mean: float = 0.0
    clip_image_story_min: float = 0.0
    clip_image_caption: float = 0.0
    nli_contra_mean: float = 0.0
    nli_contra_max: float = 0.0
    attribute_conflict: list[str] = field(default_factory=list)
    repetition_rate: float = 0.0
    length_valid: bool = False
    grounding_pass: bool = False
    
    # V2 metrics
    claim_support_rate: float = 0.0
    claim_grounding_score: float = 0.0
    continuity_score: float = 0.0
    narrative_coherence: float = 0.0
    contradiction_count: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    
    # Runtime
    eval_clip_s: float = 0.0
    eval_nli_s: float = 0.0
    eval_rules_s: float = 0.0
    eval_verification_s: float = 0.0
    eval_total_s: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "grounding_score": self.grounding_score,
            "clip_image_story_mean": self.clip_image_story_mean,
            "clip_image_story_min": self.clip_image_story_min,
            "clip_image_caption": self.clip_image_caption,
            "nli_contra_mean": self.nli_contra_mean,
            "nli_contra_max": self.nli_contra_max,
            "attribute_conflict": self.attribute_conflict,
            "repetition_rate": self.repetition_rate,
            "length_valid": self.length_valid,
            "grounding_pass": self.grounding_pass,
            "claim_support_rate": self.claim_support_rate,
            "claim_grounding_score": self.claim_grounding_score,
            "continuity_score": self.continuity_score,
            "narrative_coherence": self.narrative_coherence,
            "contradiction_count": self.contradiction_count,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "contradicted_claims": self.contradicted_claims,
            "eval_clip_s": self.eval_clip_s,
            "eval_nli_s": self.eval_nli_s,
            "eval_rules_s": self.eval_rules_s,
            "eval_verification_s": self.eval_verification_s,
            "eval_total_s": self.eval_total_s,
        }


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    mode: Literal["fast", "standard", "full", "baseline"] = "standard"
    use_grounding_dino: bool = True
    use_ocr: bool = False
    use_faiss: bool = True
    use_creative_planner: bool = True
    use_verification: bool = True
    max_evidence_per_frame: int = 50
    faiss_top_k: int = 10
    context_max_words: int = 500
    # V2.3: Hierarchical memory configuration
    scene_similarity_threshold: float = 0.65
    max_scene_size: int = 20
    top_k_scenes: int = 5
    top_k_evidence_per_scene: int = 10
    scene_retrieval_threshold: float = 0.5
    evidence_retrieval_threshold: float = 0.4
    max_scenes_in_context: int = 5
    target_story_words: int = 250
    creativity_config: dict[str, float] = field(default_factory=lambda: {
        "creativity": 0.7,
        "surprise": 0.6,
        "humor": 0.5,
        "mystery": 0.4,
        "emotion": 0.5,
        "dialogue": 0.4,
        "metaphor": 0.3,
    })
    # V2.2: Risk-aware creative budget
    creative_budget: dict[str, Any] = field(default_factory=lambda: {
        "safe_creative": {"max": 8, "used": 0},      # personality, humor, dialogue, metaphor
        "risky_inferred": {"max": 3, "used": 0},    # motivations, uncertain actions
        "forbidden_visual": {"max": 0, "used": 0},  # new objects, colors, materials, people
    })
    genre: str = "whimsical"
    tone: str = "comedic"
    seed: int = 0
    device: str = "cpu"
    
    @classmethod
    def from_mode(cls, mode: str) -> PipelineConfig:
        """Create config from preset mode."""
        if mode == "fast":
            return cls(
                mode="fast",
                use_grounding_dino=False,
                use_ocr=False,
                use_faiss=False,
                use_creative_planner=False,
                use_verification=False,
                target_story_words=100,
                scene_similarity_threshold=0.65,
                max_scene_size=20,
                top_k_scenes=3,
                top_k_evidence_per_scene=5,
                scene_retrieval_threshold=0.5,
                evidence_retrieval_threshold=0.4,
                max_scenes_in_context=3,
                creative_budget={
                    "safe_creative": {"max": 2, "used": 0},
                    "risky_inferred": {"max": 1, "used": 0},
                    "forbidden_visual": {"max": 0, "used": 0},
                },
            )
        elif mode == "standard":
            return cls(
                mode="standard",
                use_grounding_dino=True,
                use_ocr=False,
                use_faiss=True,
                use_creative_planner=True,
                use_verification=True,
                target_story_words=250,
                scene_similarity_threshold=0.65,
                max_scene_size=20,
                top_k_scenes=5,
                top_k_evidence_per_scene=10,
                scene_retrieval_threshold=0.5,
                evidence_retrieval_threshold=0.4,
                max_scenes_in_context=5,
                creative_budget={
                    "safe_creative": {"max": 8, "used": 0},
                    "risky_inferred": {"max": 3, "used": 0},
                    "forbidden_visual": {"max": 0, "used": 0},
                },
            )
        elif mode == "full":
            return cls(
                mode="full",
                use_grounding_dino=True,
                use_ocr=True,
                use_faiss=True,
                use_creative_planner=True,
                use_verification=True,
                target_story_words=300,
                creativity_config={
                    "creativity": 0.8,
                    "surprise": 0.7,
                    "humor": 0.6,
                    "mystery": 0.5,
                    "emotion": 0.6,
                    "dialogue": 0.5,
                    "metaphor": 0.4,
                },
                scene_similarity_threshold=0.65,
                max_scene_size=20,
                top_k_scenes=7,
                top_k_evidence_per_scene=15,
                scene_retrieval_threshold=0.5,
                evidence_retrieval_threshold=0.4,
                max_scenes_in_context=7,
                creative_budget={
                    "safe_creative": {"max": 10, "used": 0},
                    "risky_inferred": {"max": 4, "used": 0},
                    "forbidden_visual": {"max": 0, "used": 0},
                },
            )
        elif mode == "baseline":
            return cls(
                mode="baseline",
                use_grounding_dino=False,
                use_ocr=False,
                use_faiss=False,
                use_creative_planner=False,
                use_verification=False,
                target_story_words=100,
                creativity_config={
                    "creativity": 0.3,
                    "surprise": 0.2,
                    "humor": 0.2,
                    "mystery": 0.1,
                    "emotion": 0.2,
                    "dialogue": 0.2,
                    "metaphor": 0.1,
                },
                scene_similarity_threshold=0.65,
                max_scene_size=20,
                top_k_scenes=3,
                top_k_evidence_per_scene=5,
                scene_retrieval_threshold=0.5,
                evidence_retrieval_threshold=0.4,
                max_scenes_in_context=3,
                creative_budget={
                    "safe_creative": {"max": 2, "used": 0},
                    "risky_inferred": {"max": 1, "used": 0},
                    "forbidden_visual": {"max": 0, "used": 0},
                },
            )
        elif mode == "collection":
            return cls(
                mode="collection",
                use_grounding_dino=True,
                use_ocr=False,
                use_faiss=True,
                use_creative_planner=True,
                use_verification=True,
                target_story_words=300,
                scene_similarity_threshold=0.65,
                max_scene_size=20,
                top_k_scenes=10,
                top_k_evidence_per_scene=20,
                scene_retrieval_threshold=0.5,
                evidence_retrieval_threshold=0.4,
                max_scenes_in_context=10,
                creative_budget={
                    "safe_creative": {"max": 10, "used": 0},
                    "risky_inferred": {"max": 4, "used": 0},
                    "forbidden_visual": {"max": 0, "used": 0},
                },
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "use_grounding_dino": self.use_grounding_dino,
            "use_ocr": self.use_ocr,
            "use_faiss": self.use_faiss,
            "use_creative_planner": self.use_creative_planner,
            "use_verification": self.use_verification,
            "max_evidence_per_frame": self.max_evidence_per_frame,
            "faiss_top_k": self.faiss_top_k,
            "context_max_words": self.context_max_words,
            "scene_similarity_threshold": self.scene_similarity_threshold,
            "max_scene_size": self.max_scene_size,
            "top_k_scenes": self.top_k_scenes,
            "top_k_evidence_per_scene": self.top_k_evidence_per_scene,
            "scene_retrieval_threshold": self.scene_retrieval_threshold,
            "evidence_retrieval_threshold": self.evidence_retrieval_threshold,
            "max_scenes_in_context": self.max_scenes_in_context,
            "target_story_words": self.target_story_words,
            "creativity_config": self.creativity_config,
            "creative_budget": self.creative_budget,
            "genre": self.genre,
            "tone": self.tone,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass
class ExperimentManifest:
    """Manifest for experiment reproducibility."""
    git_commit: str = ""
    vision_model: str = ""
    language_model: str = ""
    embedding_model: str = ""
    dataset: str = ""
    dataset_hash: str = ""
    seed: int = 0
    device: str = "cpu"
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "vision_model": self.vision_model,
            "language_model": self.language_model,
            "embedding_model": self.embedding_model,
            "dataset": self.dataset,
            "dataset_hash": self.dataset_hash,
            "seed": self.seed,
            "device": self.device,
            "config": self.config,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


@dataclass
class PipelineArtifacts:
    """Artifacts produced by a pipeline run."""
    run_id: str
    manifest: ExperimentManifest
    observations: list[VisualObservations] = field(default_factory=list)
    world_state: WorldState | None = None
    retrieved_evidence: list[RetrievedEvidence] = field(default_factory=list)
    ranked_evidence: list[RetrievedEvidence] = field(default_factory=list)
    context: str = ""
    creative_plan: CreativePlan | None = None
    story_plan: StoryPlan | None = None
    story_draft: StoryDraft | None = None
    claims: list[StoryClaim] = field(default_factory=list)
    verification_results: list[VerificationResult] = field(default_factory=list)
    evaluation: EvaluationResult | None = None
    runtime: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    # V2.3: Collection memory artifacts
    collection_memory: "CollectionMemory | None" = field(default=None, repr=False)
    retrieval_result: "RetrievalResult | None" = field(default=None, repr=False)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest": self.manifest.to_dict(),
            "observations": [obs.to_dict() for obs in self.observations],
            "world_state": self.world_state.to_dict() if self.world_state else None,
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "ranked_evidence": [e.to_dict() for e in self.ranked_evidence],
            "context": self.context,
            "creative_plan": self.creative_plan.to_dict() if self.creative_plan else None,
            "story_plan": self.story_plan.to_dict() if self.story_plan else None,
            "story_draft": self.story_draft.to_dict() if self.story_draft else None,
            "claims": [c.to_dict() for c in self.claims],
            "verification_results": [v.to_dict() for v in self.verification_results],
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "runtime": self.runtime,
            "logs": self.logs,
            "collection_memory": self.collection_memory.to_dict() if self.collection_memory else None,
            "retrieval_result": self.retrieval_result.to_dict() if self.retrieval_result else None,
        }