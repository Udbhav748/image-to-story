"""Custom exceptions for the Image → Story V2 system."""


class ImageStoryError(Exception):
    """Base exception for Image → Story system."""
    pass


class VisionError(ImageStoryError):
    """Vision model related errors."""
    pass


class ModelLoadError(VisionError):
    """Failed to load a vision model."""
    pass


class InferenceError(VisionError):
    """Failed to run inference on a vision model."""
    pass


class GroundingDINOError(VisionError):
    """GroundingDINO specific errors."""
    pass


class OCRError(VisionError):
    """OCR specific errors."""
    pass


class EvidenceError(ImageStoryError):
    """Evidence store related errors."""
    pass


class EvidenceNotFoundError(EvidenceError):
    """Requested evidence not found."""
    pass


class InvalidEvidenceError(EvidenceError):
    """Evidence record is invalid."""
    pass


class WorldStateError(ImageStoryError):
    """World state tracking errors."""
    pass


class EntityNotFoundError(WorldStateError):
    """Entity not found in world state."""
    pass


class MemoryError(ImageStoryError):
    """Memory/FAISS related errors."""
    pass


class FAISSIndexError(MemoryError):
    """FAISS index operation failed."""
    pass


class EmbeddingError(MemoryError):
    """Embedding generation failed."""
    pass


class RetrievalError(MemoryError):
    """Evidence retrieval failed."""
    pass


class ContextError(ImageStoryError):
    """Context building errors."""
    pass


class ContextBuildError(ContextError):
    """Failed to build context."""
    pass


class NarrativeError(ImageStoryError):
    """Narrative/creative engine errors."""
    pass


class CreativePlanningError(NarrativeError):
    """Creative planning failed."""
    pass


class StoryBeatError(NarrativeError):
    """Story beat planning failed."""
    pass


class GenerationError(ImageStoryError):
    """Story generation errors."""
    pass


class ModelGenerationError(GenerationError):
    """Language model generation failed."""
    pass


class PromptError(GenerationError):
    """Prompt construction failed."""
    pass


class VerificationError(ImageStoryError):
    """Claim verification errors."""
    pass


class ClaimExtractionError(VerificationError):
    """Failed to extract claims from story."""
    pass


class ClaimVerificationError(VerificationError):
    """Failed to verify claim against evidence."""
    pass


class EvaluationError(ImageStoryError):
    """Evaluation errors."""
    pass


class MetricComputationError(EvaluationError):
    """Failed to compute evaluation metric."""
    pass


class ConfigurationError(ImageStoryError):
    """Configuration errors."""
    pass


class InvalidConfigError(ConfigurationError):
    """Configuration is invalid."""
    pass


class MissingDependencyError(ConfigurationError):
    """Required dependency not available."""
    pass


class PipelineError(ImageStoryError):
    """Pipeline orchestration errors."""
    pass


class StageExecutionError(PipelineError):
    """Pipeline stage execution failed."""
    pass


class ModeNotSupportedError(PipelineError):
    """Requested mode not supported."""
    pass