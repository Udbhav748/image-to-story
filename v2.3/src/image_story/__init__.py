"""Image → Story V2: Grounded Creative Multimodal Storytelling System.

A professional, modular, reproducible multimodal ML system that converts
visual input into creative stories while maintaining strict visual grounding.

Architecture:
  IMAGE(S) → VISUAL PERCEPTION → GROUNDED EVIDENCE → WORLD/ENTITY STATE
  → FAISS RETRIEVAL → EVIDENCE RANKING → RELIABILITY-AWARE CONTEXT
  → CREATIVE STORY ENGINE → STORY BEAT PLANNER → QWEN WRITER
  → CLAIM EXTRACTION → CLAIM/VISUAL VERIFICATION
  → CONTINUITY + CONTRADICTION + QUALITY EVALUATION
  → FINAL GROUNDED CREATIVE STORY
"""

__version__ = "2.0.0"
__author__ = "Image → Story Team"

from .domain.schemas import (
    EvidenceRecord,
    VisualObservations,
    WorldState,
    WorldEntity,
    CreativePlan,
    StoryPlan,
    StoryBeat,
    StoryDraft,
    StoryClaim,
    VerificationResult,
    EvaluationResult,
    PipelineConfig,
    ExperimentManifest,
    PipelineArtifacts,
)
from .domain.enums import (
    PipelineMode,
    EvidenceType,
    InformationClass,
    StoryGenre,
    StoryTone,
    BeatType,
    ConflictType,
    HumorStyle,
)
from .pipeline.orchestrator import PipelineOrchestrator, create_orchestrator
from .config.settings import Settings, settings, get_pipeline_config

__all__ = [
    # Domain
    "EvidenceRecord",
    "VisualObservations",
    "WorldState",
    "WorldEntity",
    "CreativePlan",
    "StoryPlan",
    "StoryBeat",
    "StoryDraft",
    "StoryClaim",
    "VerificationResult",
    "EvaluationResult",
    "PipelineConfig",
    "ExperimentManifest",
    "PipelineArtifacts",
    # Enums
    "PipelineMode",
    "EvidenceType",
    "InformationClass",
    "StoryGenre",
    "StoryTone",
    "BeatType",
    "ConflictType",
    "HumorStyle",
    # Pipeline
    "PipelineOrchestrator",
    "create_orchestrator",
    # Config
    "Settings",
    "settings",
    "get_pipeline_config",
]