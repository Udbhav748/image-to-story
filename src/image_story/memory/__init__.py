"""Memory layer: embeddings, vector store and hierarchical memory construction.

Canonical locations for the memory contracts are `memory/hierarchical.py`
(`SceneSummary`, `CollectionMemory`, `EntityMemory`, `StateTransition`,
`NarrativeElement`, `RetrievalResult`, `RetrievalFilter`) and
`evidence/provenance.py` (`IndexedEvidenceRecord`, `SceneEvidenceLink`).
"""
from .collection import CollectionMemoryBuilder, create_collection_memory_builder
from .embeddings import (
    EmbeddingModelInterface,
    SentenceTransformerEmbedding,
    create_embedding_model,
)
from .entities import EntityMemoryTracker, create_entity_memory_tracker
from .faiss_store import FAISSVectorStore
from .hierarchical import (
    CollectionMemory,
    EntityMemory,
    IndexedEvidenceRecord,
    MemoryLevel,
    NarrativeElement,
    RetrievalFilter,
    RetrievalResult,
    SceneEvidenceLink,
    SceneSummary,
    StateTransition,
    apply_retrieval_filter,
    create_retrieval_filter,
)
from .narrative import NarrativeMemory, NarrativeMemoryRetriever, create_narrative_memory
from .scenes import SceneGroupConfig, SceneGrouper, create_scene_grouper
from .transitions import StateTransitionDetector, create_state_transition_detector

__all__ = [
    # Embeddings and vector store
    "EmbeddingModelInterface",
    "SentenceTransformerEmbedding",
    "create_embedding_model",
    "FAISSVectorStore",
    # Hierarchical memory contracts
    "SceneSummary",
    "CollectionMemory",
    "EntityMemory",
    "StateTransition",
    "NarrativeElement",
    "SceneEvidenceLink",
    "RetrievalResult",
    "MemoryLevel",
    "IndexedEvidenceRecord",
    "RetrievalFilter",
    "create_retrieval_filter",
    "apply_retrieval_filter",
    # Builders
    "SceneGrouper",
    "SceneGroupConfig",
    "create_scene_grouper",
    "EntityMemoryTracker",
    "create_entity_memory_tracker",
    "StateTransitionDetector",
    "create_state_transition_detector",
    "NarrativeMemory",
    "NarrativeMemoryRetriever",
    "create_narrative_memory",
    "CollectionMemoryBuilder",
    "create_collection_memory_builder",
]
