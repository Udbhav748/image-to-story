"""Image collection and ingestion models for V2.3A."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
from datetime import datetime
import uuid
import hashlib

from .enums import (
    ImageOrderingMode,
    ImageValidationStatus,
    ImageProcessingStatus,
    ProcessingJobStatus,
    CollectionStatus,
    SessionStatus,
    ImageOrderingMode,
    ImageValidationStatus,
    ImageProcessingStatus,
    ProcessingJobStatus,
    CollectionStatus,
)
from .enums import PipelineMode


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
    validation_status: str = "pending"  # ImageValidationStatus
    processing_status: str = "pending"  # ImageProcessingStatus
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
    ordering_mode: str = "auto"  # ImageOrderingMode
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
    status: str = "queued"  # ProcessingJobStatus
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
    status: str = "pending"  # SessionStatus
    pipeline_mode: str = "standard"  # PipelineMode
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


@dataclass
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