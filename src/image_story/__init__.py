"""Image -> Story: grounded creative multimodal storytelling.

Canonical production architecture (V2). Flow:

    IMAGE(S)
      -> INGESTION / SESSION        image_story.ingestion
      -> VISUAL PERCEPTION         image_story.vision
      -> VISUAL EVIDENCE           image_story.evidence
      -> WORLD MODEL               image_story.world
      -> HIERARCHICAL MEMORY       image_story.memory
      -> RETRIEVAL                 image_story.retrieval
      -> CAUSAL REASONING          (V3 - not implemented)
      -> NARRATIVE PLANNING        image_story.narrative
      -> STORY GENERATION          image_story.generation
      -> VERIFICATION              image_story.verification
      -> EVALUATION                image_story.evaluation
      -> EVIDENCE + METRICS        image_story.experiments

Run the pipeline with `python -m image_story <paths>`.

The Challenge V1 implementation is preserved separately under
`challenges/image-story-v1/` as a historical baseline. It is not imported here.
"""
from .config.settings import Settings, get_pipeline_config, settings
from .domain.enums import (
    BeatType,
    ClaimClassification,
    ConflictType,
    EvidenceConfidence,
    EvidenceType,
    HumorStyle,
    InformationClass,
    PipelineMode,
    StoryGenre,
    StoryTone,
)
from .domain.schemas import (
    CollectionPipelineConfig,
    CreativePlan,
    EvaluationResult,
    EvidenceRecord,
    ExperimentManifest,
    ImageCollection,
    ImageRecord,
    PipelineArtifacts,
    PipelineConfig,
    ProcessingConfig,
    ProcessingJob,
    StoryBeat,
    StoryClaim,
    StoryDraft,
    StoryPlan,
    StorySession,
    VerificationResult,
    VisualObservations,
    WorldEntity,
    WorldState,
)
from .evidence.provenance import IndexedEvidenceRecord
from .experiments.runners import ExperimentSummary, run_experiment
from .memory.hierarchical import (
    CollectionMemory,
    EntityMemory,
    NarrativeElement,
    RetrievalResult,
    SceneSummary,
    StateTransition,
)
from .pipeline.orchestrator import PipelineOrchestrator, create_orchestrator
from .pipeline.stages import Stage

__version__ = "2.3.0"
__author__ = "Image -> Story Team"

__all__ = [
    # Domain contracts
    "EvidenceRecord",
    "VisualObservations",
    "WorldState",
    "WorldEntity",
    "SceneSummary",
    "CollectionMemory",
    "EntityMemory",
    "StateTransition",
    "NarrativeElement",
    "IndexedEvidenceRecord",
    "RetrievalResult",
    "CreativePlan",
    "StoryPlan",
    "StoryBeat",
    "StoryDraft",
    "StoryClaim",
    "VerificationResult",
    "EvaluationResult",
    "PipelineConfig",
    "ProcessingConfig",
    "CollectionPipelineConfig",
    "ImageRecord",
    "ImageCollection",
    "ProcessingJob",
    "StorySession",
    "ExperimentManifest",
    "PipelineArtifacts",
    # Enums
    "PipelineMode",
    "EvidenceType",
    "EvidenceConfidence",
    "InformationClass",
    "ClaimClassification",
    "StoryGenre",
    "StoryTone",
    "BeatType",
    "ConflictType",
    "HumorStyle",
    # Pipeline
    "PipelineOrchestrator",
    "create_orchestrator",
    "Stage",
    # Experiments
    "run_experiment",
    "ExperimentSummary",
    # Config
    "Settings",
    "settings",
    "get_pipeline_config",
]
