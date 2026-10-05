"""Memory module initialization."""
from .embeddings import EmbeddingModelInterface, SentenceTransformerEmbedding, create_embedding_model
from .faiss_store import FAISSVectorStore
from .retrieval import EvidenceRetriever
from .hierarchical import (
    SceneSummary,
    CollectionMemory,
    EntityMemory,
    StateTransition,
    NarrativeElement,
    RetrievalResult,
    MemoryLevel,
    IndexedEvidenceRecord,
    RetrievalFilter,
    create_retrieval_filter,
    apply_retrieval_filter,
)
from .scene_grouper import SceneGrouper, SceneGroupConfig, create_scene_grouper
from .entity_memory import EntityMemoryTracker, create_entity_memory_tracker
from .state_transitions import StateTransitionDetector, create_state_transition_detector
from .narrative_memory import NarrativeMemory, NarrativeMemoryRetriever, create_narrative_memory
from .hierarchical_retriever import HierarchicalRetriever, HierarchicalRetrievalConfig, create_hierarchical_retriever
from .collection_builder import CollectionMemoryBuilder, create_collection_memory_builder

__all__ = [
    "EmbeddingModelInterface",
    "SentenceTransformerEmbedding",
    "create_embedding_model",
    "FAISSVectorStore",
    "EvidenceRetriever",
    # Hierarchical memory
    "SceneSummary",
    "CollectionMemory",
    "EntityMemory",
    "StateTransition",
    "NarrativeElement",
    "RetrievalResult",
    "MemoryLevel",
    "IndexedEvidenceRecord",
    "RetrievalFilter",
    "create_retrieval_filter",
    "apply_retrieval_filter",
    # Scene grouping
    "SceneGrouper",
    "SceneGroupConfig",
    "create_scene_grouper",
    # Entity memory
    "EntityMemoryTracker",
    "create_entity_memory_tracker",
    # State transitions
    "StateTransitionDetector",
    "create_state_transition_detector",
    # Narrative memory
    "NarrativeMemory",
    "NarrativeMemoryRetriever",
    "create_narrative_memory",
    # Hierarchical retrieval
    "HierarchicalRetriever",
    "HierarchicalRetrievalConfig",
    "create_hierarchical_retriever",
    # Collection builder
    "CollectionMemoryBuilder",
    "create_collection_memory_builder",
]