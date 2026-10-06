"""High-level retrieval interface combining embedding and FAISS."""
from typing import Any, Callable
import numpy as np

from ..domain.schemas import EvidenceRecord, RetrievedEvidence, VisualObservations
from .embeddings import EmbeddingModelInterface, create_embedding_model
from .faiss_store import FAISSVectorStore


class EvidenceRetriever:
    """High-level interface for evidence retrieval."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModelInterface | None = None,
        vector_store: FAISSVectorStore | None = None,
        device: str = "cpu",
    ):
        self._embedding_model = embedding_model or create_embedding_model(device=device)
        self._vector_store = vector_store or FAISSVectorStore(self._embedding_model, device=device)
        self._device = device
    
    @property
    def embedding_model(self) -> EmbeddingModelInterface:
        return self._embedding_model
    
    @property
    def vector_store(self) -> FAISSVectorStore:
        return self._vector_store
    
    def initialize(self) -> None:
        self._embedding_model.load()
        self._vector_store.initialize()
    
    def index_observations(self, observations: list[VisualObservations]) -> int:
        """Index all evidence records from observations."""
        evidence_records = []
        texts = []
        
        for obs in observations:
            for evidence in obs.evidence_records:
                text = self._evidence_to_text(evidence)
                evidence_records.append(evidence)
                texts.append(text)
        
        if evidence_records:
            self._vector_store.add_batch(evidence_records, texts)
        
        return len(evidence_records)
    
    def _evidence_to_text(self, evidence: EvidenceRecord) -> str:
        """Convert evidence to text for embedding."""
        parts = []
        if evidence.entity:
            parts.append(f"Entity: {evidence.entity}")
        if evidence.type:
            parts.append(f"Type: {evidence.type.value}")
        if evidence.action:
            parts.append(f"Action: {evidence.action}")
        if evidence.relationship:
            parts.append(f"Relation: {evidence.relationship}")
        if evidence.evidence_text:
            parts.append(evidence.evidence_text)
        if evidence.frame_id >= 0:
            parts.append(f"Frame: {evidence.frame_id}")
        
        return " | ".join(parts)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Callable[[EvidenceRecord], bool] | None = None,
    ) -> list[RetrievedEvidence]:
        """Retrieve evidence relevant to a query."""
        return self._vector_store.search(query, top_k, filter_fn)
    
    def retrieve_by_entity(
        self,
        entity: str,
        top_k: int = 10,
    ) -> list[RetrievedEvidence]:
        """Retrieve evidence for a specific entity."""
        return self._vector_store.search(f"Entity: {entity}", top_k)
    
    def retrieve_by_frame(
        self,
        frame_id: int,
        top_k: int = 10,
    ) -> list[RetrievedEvidence]:
        """Retrieve evidence from a specific frame."""
        def frame_filter(ev: EvidenceRecord) -> bool:
            return ev.frame_id == frame_id
        return self._vector_store.search(f"Frame: {frame_id}", top_k, frame_filter)
    
    def retrieve_by_type(
        self,
        evidence_type: str,
        top_k: int = 10,
    ) -> list[RetrievedEvidence]:
        """Retrieve evidence of a specific type."""
        def type_filter(ev: EvidenceRecord) -> bool:
            return ev.type.value == evidence_type
        return self._vector_store.search(f"Type: {evidence_type}", top_k, type_filter)
    
    def retrieve_for_story_planning(
        self,
        world_state_summary: str,
        current_frame: int,
        top_k: int = 15,
    ) -> list[RetrievedEvidence]:
        """Retrieve evidence relevant for story planning.
        
        Combines:
        - Current frame evidence
        - Recurring entities
        - Open loops
        - Semantic similarity to world state
        """
        results = []
        
        current_frame_evidence = self.retrieve_by_frame(current_frame, top_k=top_k)
        results.extend(current_frame_evidence)
        
        if world_state_summary:
            semantic_results = self.retrieve(world_state_summary, top_k=top_k)
            for r in semantic_results:
                if not any(r.record.id == existing.record.id for existing in results):
                    results.append(r)
        
        seen = set()
        unique_results = []
        for r in results:
            if r.record.id not in seen:
                seen.add(r.record.id)
                unique_results.append(r)
        
        return unique_results[:top_k]
    
    def get_all_evidence(self) -> list[EvidenceRecord]:
        """Get all indexed evidence records."""
        return list(self._vector_store._evidence_map.values())
    
    def save(self, path: str) -> None:
        self._vector_store.save(path)
    
    def load(self, path: str) -> None:
        self._vector_store.load(path)
    
    def get_stats(self) -> dict[str, Any]:
        return {
            "embedding_model": self._embedding_model.model_id,
            "vector_store": self._vector_store.get_stats(),
        }