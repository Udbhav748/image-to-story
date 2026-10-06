"""Collection pipeline for running story generation on image collections."""
import os
import json
from pathlib import Path
from typing import Any
from datetime import datetime
import uuid

from ..domain.schemas import (
    PipelineConfig,
    PipelineArtifacts,
    ExperimentManifest,
    ImageCollection,
    ImageRecord,
    StorySession,
    PipelineConfig,
    ProcessingConfig,
    CollectionPipelineConfig,
)
from .pipeline import IngestionPipeline
from ..pipeline.orchestrator import PipelineOrchestrator
from ..domain.schemas import PipelineArtifacts


class CollectionPipeline:
    """High-level pipeline for processing image collections and generating stories."""
    
    def __init__(
        self,
        config: PipelineConfig | None = None,
        collection_config: "CollectionPipelineConfig | None" = None,
    ):
        from ..config.settings import get_pipeline_config
        from .pipeline import IngestionPipeline
        
        self.config = config or PipelineConfig()
        self.collection_config = collection_config or CollectionPipelineConfig()
        self.ingestion = IngestionPipeline(
            processing_config=self.collection_config.processing
        )
        
        # Initialize orchestrator lazily
        self._orchestrator: "PipelineOrchestrator | None" = None
    
    def _get_orchestrator(self) -> "PipelineOrchestrator":
        """Get or create the pipeline orchestrator."""
        if self._orchestrator is None:
            from ..pipeline.orchestrator import PipelineOrchestrator
            self._orchestrator = PipelineOrchestrator(self.config)
        return self._orchestrator
    
    def create_collection_from_paths(
        self,
        paths: list[str],
        ordering_mode: str = "auto",
    ) -> "ImageCollection":
        """Create an ImageCollection from a list of image paths."""
        return self.ingestion.create_collection_from_paths(paths, ordering_mode)
    
    def process_collection(
        self,
        collection: "ImageCollection",
        config: PipelineConfig | None = None,
        evaluate: bool = True,
        save_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Process an image collection through the full pipeline:
        1. Ingestion (validation, hashing, deduplication)
        2. Story generation via PipelineOrchestrator
        3. Evaluation
        """
        config = self.config if config is None else config
        orchestrator = self._get_orchestrator()
        
        # Run ingestion pipeline
        start_time = datetime.now()
        ingestion_results = self.ingestion.process_collection(
            collection, 
            processing_config=self.collection_config.processing,
        )
        ingestion_time = (datetime.now() - start_time).total_seconds()
        
        # Get valid images in correct order
        valid_images = collection.get_valid_sorted(collection.ordering_mode)
        image_paths = [img.path for img in valid_images]
        
        if not image_paths:
            return {
                "success": False,
                "error": "No valid images in collection",
                "collection_id": collection.collection_id,
            }
        
        # Run story generation pipeline
        pipeline_start = datetime.now()
        artifacts = self._orchestrator.run_multi_image(
            image_paths, 
            evaluate=True,
        )
        generation_time = (datetime.now() - pipeline_start).total_seconds()
        
        # Create session record
        session = StorySession(
            collection_id=collection.collection_id,
            config_id=f"config_{hash(str(self.config.to_dict()))}",
            pipeline_mode=self.config.mode,
            config_snapshot=self.config.to_dict(),
        )
        
        # Save artifacts
        if self.collection_config.artifacts_dir:
            from main_v2 import save_artifacts
            save_dir = Path(self.collection_config.artifacts_dir) / f"session_{session.session_id}"
            save_artifacts(artifacts, str(save_dir))
            session.artifacts_ref = str(save_dir)
        
        return {
            "success": True,
            "session_id": session.session_id,
            "collection_id": collection.collection_id,
            "story": artifacts.story_draft.text if artifacts.story_draft else "",
            "word_count": artifacts.story_draft.word_count if artifacts.story_draft else 0,
            "evaluation": artifacts.evaluation.to_dict() if artifacts.evaluation else None,
            "ingestion_time": (datetime.now() - datetime.now()).total_seconds(),  # Will be overwritten
            "generation_time": generation_time,
            "total_time": 0,
        }
    
    def process_collection_with_resume(
        self,
        collection: "ImageCollection",
        session: "StorySession" | None = None,
        evaluate: bool = True,
        save_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a collection with resume capability.
        If session is provided, resumes from where it left off.
        """
        # Create or resume session
        if session is None:
            session = StorySession(
                collection_id=collection.collection_id,
                pipeline_mode=self.config.mode,
                config_snapshot=self.config.to_dict(),
            )
        
        # Update session status
        session.status = "processing"
        session.updated_at = datetime.now().isoformat()
        
        try:
            result = self.process_collection(collection, evaluate=True, save_dir=save_dir)
            
            session.status = "completed" if result["success"] else "failed"
            session.completed_at = datetime.now().isoformat()
            session.processing_summary = result
            session.updated_at = datetime.now().isoformat()
            
            return {
                "success": True,
                "session": session.to_dict(),
                "result": result,
            }
        except Exception as e:
            session.status = "failed"
            session.error = str(e)
            session.updated_at = datetime.now().isoformat()
            return {
                "success": False,
                "session": session.to_dict(),
                "error": str(e),
            }
    
    def retry_failed_images(
        self,
        collection: "ImageCollection",
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Retry processing for failed images in a collection."""
        from ..domain.schemas import ImageRecord
        
        max_retries = max_retries or self.collection_config.processing.max_retries
        retried = 0
        
        for image in collection.images:
            if image.processing_status == "failed" and image.error:
                # Check retry count from metadata
                retry_count = image.metadata.get("retry_count", 0)
                if retry_count < (max_retries or 3):
                    # Reset for retry
                    image.processing_status = "pending"
                    image.validation_status = "pending"
                    image.error = ""
                    image.error_message = ""
                    image.metadata["retry_count"] = retry_count + 1
                    image.updated_at = datetime.now().isoformat()
                    retried += 1
        
        if retried > 0:
            return self.process_collection(collection)
        
        return {"retried": 0, "message": "No images to retry"}


def create_story_session(
    image_paths: list[str],
    config: PipelineConfig | None = None,
    output_dir: str = "artifacts",
) -> dict[str, Any]:
    """
    Convenience function to run a complete story session from image paths.
    
    Args:
        image_paths: List of image file paths
        config: Pipeline configuration (uses standard if None)
        output_dir: Directory to save artifacts
    
    Returns:
        Dictionary with session results
    """
    from ..config.settings import get_pipeline_config
    
    config = config or get_pipeline_config("standard")
    pipeline = CollectionPipeline(config=config)
    
    # Create collection
    collection = pipeline.create_collection_from_paths(image_paths)
    
    # Create session
    session = StorySession(
        collection_id=collection.collection_id,
        pipeline_mode=config.mode,
        config_snapshot=config.to_dict(),
    )
    
    # Process with resume
    result = pipeline.process_collection_with_resume(collection, session, save_dir=output_dir)
    
    return result