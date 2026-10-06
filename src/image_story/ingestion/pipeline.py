"""Ingestion pipeline for image collections: validation, hashing, deduplication."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..domain.schemas import (
    ImageCollection,
    ImageRecord,
    ProcessingConfig,
    StorySession,
)
from ..observability.logging import get_logger
from .dedup import Deduplicator
from .hashing import ImageHasher
from .validator import ImageValidator


class IngestionPipeline:
    """Handles ingestion of image collections: validation, hashing, deduplication."""

    def __init__(
        self,
        validator: ImageValidator | None = None,
        hasher: ImageHasher | None = None,
        deduplicator: Deduplicator | None = None,
        processing_config: ProcessingConfig | None = None,
    ):
        self.validator = validator or ImageValidator()
        self.hasher = hasher or ImageHasher()
        self.deduplicator = deduplicator or Deduplicator()
        self.processing_config = processing_config or ProcessingConfig()
        self.logger = get_logger("image_story.ingestion")

    def log(self, message: str) -> None:
        self.logger.info(message)

    def validate_images(self, collection: ImageCollection) -> tuple[int, int, int]:
        """
        Validate all images in a collection.
        Returns (valid_count, invalid_count, skipped_count).

        Images already processed are left alone unless `force_reprocess` is set.
        "Skipped" counts as already processed: `process_collection_with_resume`
        marks completed images as skipped, and this loop used to re-validate them
        and reset the status, which defeated resume entirely.
        """
        valid = 0
        invalid = 0
        skipped = 0
        already_done = {"completed", "skipped"}

        for image in collection.images:
            if image.processing_status in already_done and not self.processing_config.force_reprocess:
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

    def compute_hashes(self, collection: ImageCollection) -> tuple[int, int]:
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

    def deduplicate(self, collection: ImageCollection) -> tuple[int, int]:
        """Find and mark duplicates in the collection.

        Returns (exact_duplicates, near_duplicates). Registration and duplicate
        detection are a single step in `Deduplicator.register_image`, so an exact
        duplicate is still registered and can act as a near-duplicate reference
        for later images.
        """
        exact_duplicates = 0
        near_duplicates = 0

        for image in collection.images:
            if image.validation_status != "valid":
                continue

            duplicate_of, similar = self.deduplicator.register_image(
                image.image_id, image.content_hash, image.perceptual_hash
            )
            image.updated_at = datetime.now().isoformat()

            if duplicate_of and duplicate_of != image.image_id:
                image.validation_status = "duplicate"
                image.duplicate_of = duplicate_of
                image.processing_status = "completed"
                image.metadata["duplicate_of"] = duplicate_of
                exact_duplicates += 1
                continue

            if similar:
                image.similar_to = [img_id for img_id, _ in similar]
                near_duplicates += 1

        return exact_duplicates, near_duplicates

    def process_collection(
        self,
        collection: ImageCollection,
        processing_config: ProcessingConfig | None = None,
    ) -> dict[str, Any]:
        """
        Run full ingestion pipeline on a collection.
        Returns processing summary.
        """
        if processing_config:
            self.processing_config = processing_config

        start_time = datetime.now()

        # Stage 1: Validation
        self.log(f"Validating {len(collection.images)} images...")
        self.validate_images(collection)

        # Stage 2: Hashing
        self.log(f"Computing hashes for {collection.valid_images} valid images...")
        self.compute_hashes(collection)

        # Stage 3: Deduplication
        self.log("Checking for duplicates...")
        self.deduplicate(collection)

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
    ) -> ImageCollection:
        """Create an ImageCollection from a list of file paths."""
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
        collection: ImageCollection,
        session: StorySession,
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
