"""Main ingestion pipeline for image collections."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from datetime import datetime
import uuid

from ..domain.schemas import (
    ImageRecord,
    ImageCollection,
    ProcessingJob,
    ProcessingConfig,
    ImageCollection,
    ImageRecord,
    ProcessingConfig,
    StorySession,
)
from .validation import ImageValidator
from .hashing import ImageHasher, Deduplicator
from ..domain.schemas import ProcessingConfig as PipelineProcessingConfig


class IngestionPipeline:
    """Handles ingestion of image collections: validation, hashing, deduplication."""
    
    def __init__(
        self,
        validator: "ImageValidator" | None = None,
        hasher: "ImageHasher" | None = None,
        deduplicator: "Deduplicator" | None = None,
        processing_config: "PipelineProcessingConfig | None" = None,
    ):
        from .validation import ImageValidator
        from .hashing import ImageHasher, Deduplicator
        
        self.validator = validator or ImageValidator()
        self.hasher = hasher or ImageHasher()
        self.deduplicator = deduplicator or Deduplicator()
        self.processing_config = processing_config or ProcessingConfig()
    
    def validate_images(self, collection: "ImageCollection") -> tuple[int, int, int]:
        """
        Validate all images in a collection.
        Returns (valid_count, invalid_count, skipped_count).
        """
        valid = 0
        invalid = 0
        skipped = 0
        
        for image in collection.images:
            if image.processing_status == "completed" and not self.processing_config.force_reprocess:
                if image.validation_status == "valid":
                    valid += 1
                else:
                    skipped += 1
                continue
            
            status, error = self.validator.validate(image)
            image.validation_status = status
            image.error = error
            image.processing_status = "completed" if status == "valid" else "failed"
            image.updated_at = datetime.now().isoformat()
            
            if status == "valid":
                valid += 1
            elif status == "skipped":
                skipped += 1
            else:
                invalid += 1
        
        collection.valid_images = sum(1 for img in collection.images if img.validation_status == "valid")
        return valid, invalid, skipped
    
    def compute_hashes(self, collection: "ImageCollection") -> tuple[int, int]:
        """Compute content and perceptual hashes for all valid images."""
        computed = 0
        skipped = 0
        
        for image in collection.images:
            if image.validation_status != "valid":
                skipped += 1
                continue
            
            if image.content_hash and image.perceptual_hash and not self.processing_config.force_reprocess:
                skipped += 1
                continue
            
            content_hash, perceptual_hash = self.hasher.compute_hashes(image)
            image.content_hash = content_hash
            image.perceptual_hash = perceptual_hash
            image.updated_at = datetime.now().isoformat()
            computed += 1
        
        return computed, skipped
    
    def deduplicate(self, collection: "ImageCollection") -> tuple[int, int]:
        """Find and mark duplicates in the collection."""
        from ..domain.schemas import ImageRecord
        
        exact_duplicates = 0
        near_duplicates = 0
        
        for image in collection.images:
            if image.validation_status != "valid":
                continue
            
            # Check exact duplicate
            duplicate_of = self.deduplicator.check_exact_duplicate(image.content_hash)
            if duplicate_of and duplicate_of != image.image_id:
                image.validation_status = "duplicate"
                image.duplicate_of = duplicate_of
                image.processing_status = "completed"
                image.updated_at = datetime.now().isoformat()
                image.metadata["duplicate_of"] = duplicate_of
                # Register the image anyway for near-duplicate detection
                self.deduplicator.register_image(image.image_id, image.content_hash, image.perceptual_hash)
                continue
            
            # Check near duplicates
            similar = self.deduplicator.check_near_duplicates(
                image.perceptual_hash, exclude_id=image.image_id
            )
            if similar:
                image.similar_to = [img_id for img_id, _ in similar]
                near_duplicates += 1
            
            # Register this image
            self.deduplicator.register_image(image.image_id, image.content_hash, image.perceptual_hash)
        
        return exact_duplicates, near_duplicates
    
    def process_collection(
        self,
        collection: "ImageCollection",
        processing_config: "PipelineProcessingConfig" | None = None,
    ) -> dict[str, Any]:
        """
        Run full ingestion pipeline on a collection.
        Returns processing summary.
        """
        if processing_config:
            self.processing_config = processing_config
        
        start_time = datetime.now()
        
        # Stage 1: Validation
        print(f"Validating {len(collection.images)} images...")
        valid, invalid, skipped = self.validate_images(collection)
        
        # Stage 2: Hashing
        print(f"Computing hashes for {collection.valid_images} valid images...")
        computed, skipped = self.compute_hashes(collection)
        
        # Stage 3: Deduplication
        print("Checking for duplicates...")
        exact_dupes, near_dupes = self.deduplicate(collection)
        
        end_time = datetime.now()
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "total_images": len(collection.images),
            "valid": collection.valid_images,
            "invalid": len(collection.images) - collection.valid_images,
            "duplicates_exact": sum(1 for img in collection.images if img.validation_status == "duplicate"),
            "duplicates_near": sum(1 for img in collection.images if img.similar_to),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "images": [img.to_dict() for img in collection.images],
        }
    
    def create_collection_from_paths(
        self,
        paths: list[str],
        ordering_mode: str = "auto",
    ) -> "ImageCollection":
        """Create an ImageCollection from a list of file paths."""
        from ..domain.schemas import ImageCollection, ImageRecord
        
        collection = ImageCollection(ordering_mode="auto")
        
        for idx, path in enumerate(paths):
            record = ImageRecord(
                path=path,
                sequence_index=idx,
            )
            collection.add_image(record)
        
        # Apply ordering
        if ordering_mode != "auto" and ordering_mode != "upload_order":
            collection.images.sort(key=lambda x: x.path if ordering_mode == "filename" else x.created_at)
            for idx, img in enumerate(collection.images):
                img.sequence_index = idx
        
        return collection
    
    def process_collection_with_resume(
        self,
        collection: "ImageCollection",
        session: "StorySession",
    ) -> dict[str, Any]:
        """
        Process a collection with resume capability.
        Skips already completed images unless force_reprocess is True.
        """
        # If resume is enabled, skip already completed images
        if self.processing_config.resume:
            for image in collection.images:
                if image.processing_status == "completed" and image.validation_status == "valid":
                    image.processing_status = "skipped"
        
        return self.process_collection(collection)