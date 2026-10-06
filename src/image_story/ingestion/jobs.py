"""Collection processing jobs: ingest an image collection, generate a story, evaluate it."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..domain.schemas import (
    CollectionPipelineConfig,
    ImageCollection,
    PipelineConfig,
    StorySession,
)
from ..experiments.artifacts import save_artifacts
from ..observability.logging import get_logger
from .pipeline import IngestionPipeline


class CollectionPipeline:
    """High-level pipeline for processing image collections and generating stories."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        collection_config: CollectionPipelineConfig | None = None,
    ):
        self.config = config or PipelineConfig()
        self.collection_config = collection_config or CollectionPipelineConfig()
        self.ingestion = IngestionPipeline(
            processing_config=self.collection_config.processing
        )
        self.logger = get_logger("image_story.ingestion.jobs")
        self._orchestrator = None

    def _get_orchestrator(self):
        """Get or create the pipeline orchestrator."""
        if self._orchestrator is None:
            from ..pipeline.orchestrator import PipelineOrchestrator

            self._orchestrator = PipelineOrchestrator(self.config)
        return self._orchestrator

    def create_collection_from_paths(
        self,
        paths: list[str],
        ordering_mode: str = "auto",
    ) -> ImageCollection:
        """Create an ImageCollection from a list of image paths."""
        return self.ingestion.create_collection_from_paths(paths, ordering_mode)

    def process_collection(
        self,
        collection: ImageCollection,
        config: PipelineConfig | None = None,
        evaluate: bool = True,
        save_dir: str | None = None,
        use_collection_memory: bool = False,
    ) -> dict[str, Any]:
        """
        Process an image collection through the full pipeline:
        1. Ingestion pipeline (validation, hashing, deduplication)
        2. Story generation via PipelineOrchestrator
        3. Evaluation
        """
        config = config or self.config
        orchestrator = self._get_orchestrator()

        start_time = datetime.now()
        self.ingestion.process_collection(
            collection,
            processing_config=self.collection_config.processing,
        )
        ingestion_time = (datetime.now() - start_time).total_seconds()

        valid_images = collection.get_valid_sorted(collection.ordering_mode)
        image_paths = [img.path for img in valid_images]

        if not image_paths:
            return {
                "success": False,
                "error": "No valid images in collection",
                "collection_id": collection.collection_id,
            }

        pipeline_start = datetime.now()
        if use_collection_memory:
            artifacts = orchestrator.run_collection(image_paths, evaluate=evaluate)
        else:
            artifacts = orchestrator.run_multi_image(image_paths, evaluate=evaluate)
        generation_time = (datetime.now() - pipeline_start).total_seconds()
        total_time = ingestion_time + generation_time

        session = StorySession(
            collection_id=collection.collection_id,
            pipeline_mode=config.mode,
            config_snapshot=config.to_dict(),
        )

        if save_dir:
            save_path = Path(save_dir) / f"session_{session.session_id}"
            save_artifacts(artifacts, str(save_path))
            session.artifacts_ref = str(save_path)

        return {
            "success": True,
            "session_id": session.session_id,
            "collection_id": collection.collection_id,
            "story": artifacts.story_draft.text if artifacts.story_draft else "",
            "word_count": artifacts.story_draft.word_count if artifacts.story_draft else 0,
            "evaluation": artifacts.evaluation.to_dict() if artifacts.evaluation else None,
            "ingestion_time": ingestion_time,
            "generation_time": generation_time,
            "total_time": total_time,
        }

    def process_collection_with_resume(
        self,
        collection: ImageCollection,
        session: StorySession | None = None,
        evaluate: bool = True,
        save_dir: str | None = None,
        use_collection_memory: bool = False,
    ) -> dict[str, Any]:
        """Process a collection with resume capability."""
        if session is None:
            session = StorySession(
                collection_id=collection.collection_id,
                pipeline_mode=self.config.mode,
                config_snapshot=self.config.to_dict(),
            )

        session.status = "processing"
        session.updated_at = datetime.now().isoformat()

        try:
            result = self.process_collection(
                collection,
                evaluate=evaluate,
                save_dir=save_dir,
                use_collection_memory=use_collection_memory,
            )
        except Exception as e:
            session.status = "failed"
            session.error = str(e)
            session.updated_at = datetime.now().isoformat()
            self.logger.error("Collection processing failed: %s", e)
            return {
                "success": False,
                "session": session.to_dict(),
                "error": str(e),
            }

        session.status = "completed" if result["success"] else "failed"
        session.completed_at = datetime.now().isoformat()
        session.processing_summary = result
        session.updated_at = datetime.now().isoformat()

        return {
            "success": result["success"],
            "session": session.to_dict(),
            "result": result,
            **({"error": result["error"]} if not result["success"] else {}),
        }

    def retry_failed_images(
        self,
        collection: ImageCollection,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Retry processing for failed images in a collection."""
        max_retries = max_retries or self.collection_config.processing.max_retries
        retried = 0

        for image in collection.images:
            if image.processing_status == "failed" and image.error:
                retry_count = image.metadata.get("retry_count", 0)
                if retry_count < max_retries:
                    image.processing_status = "pending"
                    image.validation_status = "pending"
                    image.error = ""
                    image.error_message = ""
                    image.metadata["retry_count"] = retry_count + 1
                    image.updated_at = datetime.now().isoformat()
                    retried += 1

        if retried == 0:
            return {"retried": 0, "message": "No images to retry"}

        # Report how many images were retried alongside the pipeline result; the
        # count used to be discarded, so a caller could not tell a retry from a
        # first run.
        result = self.process_collection(collection)
        result["retried"] = retried
        return result


def create_story_session(
    image_paths: list[str],
    config: PipelineConfig | None = None,
    output_dir: str = "artifacts",
) -> dict[str, Any]:
    """Run a complete story session from image paths."""
    from ..config.settings import get_pipeline_config

    config = config or get_pipeline_config("standard")
    pipeline = CollectionPipeline(config=config)

    collection = pipeline.create_collection_from_paths(image_paths)
    session = StorySession(
        collection_id=collection.collection_id,
        pipeline_mode=config.mode,
        config_snapshot=config.to_dict(),
    )

    return pipeline.process_collection_with_resume(collection, session, save_dir=output_dir)
