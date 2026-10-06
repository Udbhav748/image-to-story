"""Retrieval layer: ranking, hierarchical retrieval and context compression.

Context builders live here because their single responsibility is turning
retrieved evidence into bounded text. `ContextBuilder` is the base class;
`SequenceContextBuilder` and `CollectionContextBuilder` extend it for the
sequence and collection paths.
"""
from .collection_context import (
    CollectionContextBuilder,
    CollectionContextConfig,
    create_collection_context_builder,
)
from .compression import ContextBuilder
from .flat import EvidenceRetriever
from .hierarchical import (
    HierarchicalRetrievalConfig,
    HierarchicalRetriever,
    create_hierarchical_retriever,
)
from .ranking import DEFAULT_WEIGHTS, EvidenceRanker, RankingWeights
from .sequence import SequenceContextBuilder

__all__ = [
    "EvidenceRanker",
    "RankingWeights",
    "DEFAULT_WEIGHTS",
    "HierarchicalRetriever",
    "HierarchicalRetrievalConfig",
    "create_hierarchical_retriever",
    "EvidenceRetriever",
    "ContextBuilder",
    "SequenceContextBuilder",
    "CollectionContextBuilder",
    "CollectionContextConfig",
    "create_collection_context_builder",
]
