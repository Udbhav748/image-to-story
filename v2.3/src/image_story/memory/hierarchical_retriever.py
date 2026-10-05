"""Hierarchical retrieval for scene-aware evidence access."""
from __future__ import annotations
from typing import Any, Callable
import time
import numpy as np
from dataclasses import dataclass

from ..domain.schemas import EvidenceRecord, RetrievedEvidence, VisualObservations, WorldState
from ..memory.embeddings import EmbeddingModelInterface
from ..memory.faiss_store import FAISSVectorStore
from ..memory.hierarchical import (
    SceneSummary,
    CollectionMemory,
    EntityMemory,
    StateTransition,
    NarrativeElement,
    RetrievalResult,
    RetrievalFilter,
    create_retrieval_filter,
)
from ..memory.scene_grouper import SceneGrouper, SceneGroupConfig


@dataclass
class HierarchicalRetrievalConfig:
    """Configuration for hierarchical retrieval."""
    top_k_scenes: int = 5
    top_k_evidence_per_scene: int = 10
    max_total_evidence: int = 50
    scene_similarity_threshold: float = 0.5
    evidence_similarity_threshold: float = 0.4
    include_transitions: bool = True
    include_narrative_elements: bool = True
    max_context_tokens: int = 2000


class HierarchicalRetriever:
    """Multi-level retrieval: scene -> evidence -> entity -> narrative."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModelInterface,
        vector_store: FAISSVectorStore,
        config: HierarchicalRetrievalConfig | None = None,
    ):
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._config = config or HierarchicalRetrievalConfig()
        
        # Scene grouper for query-time scene grouping if needed
        self._scene_grouper = SceneGrouper(embedding_model)
        
        # Cached collection memory
        self._collection_memory: CollectionMemory | None = None
    
    @property
    def config(self) -> HierarchicalRetrievalConfig:
        return self._config
    
    def set_collection_memory(self, collection: CollectionMemory) -> None:
        """Set the collection memory for retrieval."""
        self._collection_memory = collection
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_fn: Callable[[EvidenceRecord], bool] | None = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[RetrievedEvidence]:
        """
        Main retrieval entry point.
        
        For small collections: direct evidence search.
        For large collections: hierarchical scene->evidence search.
        """
        top_k = top_k or self._config.max_total_evidence
        
        if self._collection_memory and len(self._collection_memory.scene_summaries) > 1:
            return self._hierarchical_retrieve(query, top_k, retrieval_filter)
        else:
            return self._vector_store.search(query, top_k, filter_fn, retrieval_filter)
    
    def _hierarchical_retrieve(
        self,
        query: str,
        top_k: int,
        retrieval_filter: RetrievalFilter | None,
    ) -> list[RetrievedEvidence]:
        """Hierarchical retrieval: scenes first, then evidence within scenes."""
        # Step 1: Retrieve relevant scenes
        query_embedding = self._embedding_model.encode_single(query)
        relevant_scenes = self._retrieve_scenes_by_embedding(query_embedding)
        
        if not relevant_scenes:
            # Fallback to direct search
            return self._vector_store.search(query, top_k, retrieval_filter=retrieval_filter)
        
        # Step 2: Retrieve evidence from each relevant scene
        all_evidence = []
        evidence_per_scene = max(1, top_k // len(relevant_scenes))
        
        for scene in relevant_scenes:
            scene_evidence = self._retrieve_evidence_from_scene(
                scene, query, evidence_per_scene, retrieval_filter
            )
            all_evidence.extend(scene_evidence)
        
        # Step 3: Deduplicate and rank
        seen = set()
        unique_evidence = []
        for ev in all_evidence:
            if ev.record.id not in seen:
                seen.add(ev.record.id)
                unique_evidence.append(ev)
        
        # Sort by semantic similarity
        unique_evidence.sort(key=lambda x: x.semantic_similarity, reverse=True)
        
        return unique_evidence[:top_k]
    
    def _retrieve_scenes_by_embedding(
        self,
        query_embedding: np.ndarray,
    ) -> list[SceneSummary]:
        """Retrieve relevant scenes using embedding similarity."""
        if not self._collection_memory:
            return []
        
        scenes = self._collection_memory.scene_summaries
        if not scenes:
            return []
        
        # Compute similarities
        scene_similarities = []
        for scene in scenes:
            if scene.embedding is not None:
                sim = float(np.dot(query_embedding, scene.embedding) / 
                           (np.linalg.norm(query_embedding) * np.linalg.norm(scene.embedding) + 1e-8))
            else:
                sim = 0.0
            scene_similarities.append((scene, sim))
        
        # Filter by threshold and sort
        filtered = [(s, sim) for s, sim in scene_similarities if sim >= self._config.scene_similarity_threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return [s for s, _ in filtered[:self._config.top_k_scenes]]
    
    def _retrieve_scenes_by_embedding_with_fallback(
        self,
        query_embedding: np.ndarray,
    ) -> list[SceneSummary]:
        """Retrieve scenes with fallback to top-k if none pass threshold."""
        scenes = self._retrieve_scenes_by_embedding(query_embedding)
        if not scenes and self._collection_memory:
            # Fallback: return top scenes by similarity
            scene_similarities = []
            for scene in self._collection_memory.scene_summaries:
                if scene.embedding is not None:
                    sim = float(np.dot(query_embedding, scene.embedding) / 
                               (np.linalg.norm(query_embedding) * np.linalg.norm(scene.embedding) + 1e-8))
                else:
                    sim = 0.0
                scene_similarities.append((scene, sim))
            scene_similarities.sort(key=lambda x: x[1], reverse=True)
            scenes = [s for s, _ in scene_similarities[:self._config.top_k_scenes]]
        return scenes
    
    def _retrieve_evidence_from_scene(
        self,
        scene: SceneSummary,
        query: str,
        top_k: int,
        retrieval_filter: RetrievalFilter | None,
    ) -> list[RetrievedEvidence]:
        """Retrieve evidence from a specific scene."""
        # Create filter for this scene
        scene_filter = create_retrieval_filter(scene_ids=[scene.scene_id])
        if retrieval_filter:
            # Merge filters
            for k, v in retrieval_filter.items():
                if k not in scene_filter:
                    scene_filter[k] = v
        
        return self._vector_store.search_by_scene(scene.scene_id, query, top_k)
    
    def retrieve_scenes(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SceneSummary]:
        """Retrieve relevant scenes for a query."""
        if not self._collection_memory:
            return []
        
        top_k = top_k or self._config.top_k_scenes
        query_embedding = self._embedding_model.encode_single(query)
        return self._retrieve_scenes_by_embedding(query_embedding)[:top_k]
    
    def retrieve_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_type: str | None = None,
    ) -> list[EntityMemory]:
        """Retrieve relevant entities for a query."""
        if not self._collection_memory:
            return []
        
        query_lower = query.lower()
        entities = list(self._collection_memory.global_entities.values())
        
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        
        # Score by relevance to query
        scored = []
        for entity in entities:
            score = 0.0
            if entity.normalized_label in query_lower:
                score += 1.0
            for alias in entity.aliases:
                if alias in query_lower:
                    score += 0.5
            # Boost recurring entities
            if len(entity.scene_ids) > 1:
                score += 0.3
            if score > 0:
                scored.append((entity, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]
    
    def retrieve_transitions(
        self,
        entity_label: str | None = None,
        transition_type: str | None = None,
        scene_id: str | None = None,
    ) -> list[StateTransition]:
        """Retrieve state transitions, optionally filtered."""
        if not self._collection_memory:
            return []
        
        transitions = self._collection_memory.state_transitions
        
        if entity_label:
            transitions = [t for t in transitions if t.entity_label == entity_label]
        if transition_type:
            transitions = [t for t in transitions if t.transition_type == transition_type]
        if scene_id:
            transitions = [t for t in transitions if t.from_scene == scene_id or t.to_scene == scene_id]
        
        return sorted(transitions, key=lambda t: t.timestamp)
    
    def retrieve_narrative_elements(
        self,
        query: str,
        top_k: int = 10,
        element_type: str | None = None,
        min_importance: float = 0.0,
    ) -> list[NarrativeElement]:
        """Retrieve narrative elements for callback/foreshadowing."""
        if not self._collection_memory:
            return []
        
        elements = self._collection_memory.narrative_elements
        
        if element_type:
            elements = [e for e in elements if e.element_type == element_type]
        
        elements = [e for e in elements if e.narrative_importance >= min_importance]
        
        query_lower = query.lower()
        scored = []
        for elem in elements:
            score = elem.narrative_importance
            if elem.label.lower() in query_lower:
                score += 1.0
            if elem.description.lower() in query_lower:
                score += 0.5
            for eid in elem.evidence_ids:
                if eid in query_lower:
                    score += 0.3
            scored.append((elem, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]
    
    def retrieve_callback_candidates(
        self,
        current_scene: str,
        query: str = "",
        top_k: int = 10,
    ) -> list[NarrativeElement]:
        """
        Retrieve narrative elements that could serve as callbacks.
        
        Prioritizes:
        - Recurring elements from earlier scenes
        - Elements with high narrative importance
        - Open loops that haven't been resolved
        """
        if not self._collection_memory:
            return []
        
        # Get all elements from scenes before current
        current_idx = -1
        for i, scene in enumerate(self._collection_memory.scene_summaries):
            if scene.scene_id == current_scene:
                current_idx = i
                break
        
        if current_idx <= 0:
            return []
        
        candidate_elements = []
        for elem in self._collection_memory.narrative_elements:
            # Only consider elements from earlier scenes
            elem_scene_idx = -1
            for i, scene in enumerate(self._collection_memory.scene_summaries):
                if scene.scene_id == elem.first_scene:
                    elem_scene_idx = i
                    break
            
            if elem_scene_idx >= 0 and elem_scene_idx < current_idx:
                candidate_elements.append(elem)
        
        # Score for callback potential
        scored = []
        for elem in candidate_elements:
            score = elem.narrative_importance * 2.0  # Base importance
            
            # Boost for recurrence
            score += elem.recurrence_count * 0.3
            
            # Boost for open loops
            if elem.element_type == "open_loop" and elem.callback_payoff_scene is None:
                score += 1.0
            
            # Boost for potential callback flag
            if elem.potential_callback:
                score += 1.5
            
            # Query relevance
            if query and (elem.label.lower() in query.lower() or 
                          elem.description.lower() in query.lower()):
                score += 1.0
            
            # Recency penalty (older elements are better for callbacks)
            age = current_idx - elem_scene_idx
            if age > 1:
                score += min(age * 0.1, 0.5)
            
            scored.append((elem, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]
    
    def retrieve_full(
        self,
        query: str,
        top_k_scenes: int | None = None,
        top_k_evidence_per_scene: int | None = None,
        include_transitions: bool | None = None,
        include_narrative: bool | None = None,
    ) -> RetrievalResult:
        """
        Full hierarchical retrieval returning all levels.
        
        Returns RetrievalResult with scenes, evidence, entities, transitions, and narrative elements.
        """
        start_time = time.perf_counter()
        
        top_k_scenes = top_k_scenes or self._config.top_k_scenes
        top_k_evidence = top_k_evidence_per_scene or self._config.top_k_evidence_per_scene
        include_trans = include_transitions if include_transitions is not None else self._config.include_transitions
        include_narr = include_narrative if include_narrative is not None else self._config.include_narrative_elements
        
        # Retrieve scenes with fallback for low-similarity cases
        query_embedding = self._embedding_model.encode_single(query)
        scenes = self._retrieve_scenes_by_embedding_with_fallback(query_embedding)[:top_k_scenes]
        
        # Retrieve evidence from scenes
        all_evidence = []
        for scene in scenes:
            scene_evidence = self._retrieve_evidence_from_scene(scene, query, top_k_evidence, None)
            all_evidence.extend(scene_evidence)
        
        # Deduplicate
        seen = set()
        unique_evidence = []
        for ev in all_evidence:
            if ev.record.id not in seen:
                seen.add(ev.record.id)
                unique_evidence.append(ev)
        
        unique_evidence.sort(key=lambda x: x.semantic_similarity, reverse=True)
        unique_evidence = unique_evidence[:self._config.max_total_evidence]
        
        # Retrieve entities
        entities = self.retrieve_entities(query, top_k=10)
        
        # Retrieve transitions
        transitions = []
        if include_trans:
            transitions = self.retrieve_transitions()
        
        # Retrieve narrative elements
        narrative_elements = []
        if include_narr:
            narrative_elements = self.retrieve_narrative_elements(query, top_k=10)
        
        result = RetrievalResult(
            scenes=scenes,
            evidence=[e.record for e in unique_evidence],
            entities=entities,
            transitions=transitions,
            narrative_elements=narrative_elements,
            query=query,
            retrieval_time_ms=(time.perf_counter() - start_time) * 1000,
        )
        
        return result


def create_hierarchical_retriever(
    embedding_model: EmbeddingModelInterface,
    vector_store: FAISSVectorStore,
    top_k_scenes: int = 5,
    top_k_evidence_per_scene: int = 10,
    max_total_evidence: int = 50,
) -> HierarchicalRetriever:
    """Factory function to create hierarchical retriever."""
    config = HierarchicalRetrievalConfig(
        top_k_scenes=top_k_scenes,
        top_k_evidence_per_scene=top_k_evidence_per_scene,
        max_total_evidence=max_total_evidence,
    )
    return HierarchicalRetriever(embedding_model, vector_store, config)