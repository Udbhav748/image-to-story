"""High-level collection memory builder orchestrating all components."""
from __future__ import annotations
from typing import Any
import uuid
import time

from ..domain.schemas import (
    VisualObservations,
    WorldState,
    EvidenceRecord,
    PipelineConfig,
)
from ..memory.embeddings import EmbeddingModelInterface, create_embedding_model
from ..memory.faiss_store import FAISSVectorStore
from ..memory.hierarchical import (
    SceneSummary,
    CollectionMemory,
    EntityMemory,
    StateTransition,
    NarrativeElement,
    IndexedEvidenceRecord,
)
from ..memory.scene_grouper import SceneGrouper, SceneGroupConfig
from ..memory.entity_memory import EntityMemoryTracker
from ..memory.state_transitions import StateTransitionDetector
from ..memory.narrative_memory import NarrativeMemory, NarrativeMemoryRetriever
from ..memory.hierarchical_retriever import HierarchicalRetriever, HierarchicalRetrievalConfig
from ..context.world_state import WorldStateBuilder


class CollectionMemoryBuilder:
    """Build complete hierarchical memory from observations."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModelInterface | None = None,
        vector_store: FAISSVectorStore | None = None,
        device: str = "cpu",
        scene_config: SceneGroupConfig | None = None,
        retrieval_config: HierarchicalRetrievalConfig | None = None,
    ):
        self._device = device
        self._embedding_model = embedding_model or create_embedding_model(device=device)
        self._vector_store = vector_store or FAISSVectorStore(self._embedding_model, device=device)
        
        # Components
        self._scene_grouper = SceneGrouper(self._embedding_model, scene_config or SceneGroupConfig())
        self._entity_tracker = EntityMemoryTracker(self._embedding_model)
        self._transition_detector = StateTransitionDetector()
        self._narrative_memory = NarrativeMemory()
        self._world_builder = WorldStateBuilder()
        self._retriever = HierarchicalRetriever(
            self._embedding_model, self._vector_store, retrieval_config or HierarchicalRetrievalConfig()
        )
        
        self._collection_id = f"coll_{uuid.uuid4().hex[:8]}"
    
    def build_from_observations(
        self,
        observations: list[VisualObservations],
        config: PipelineConfig | None = None,
    ) -> CollectionMemory:
        """
        Build complete collection memory from observations.
        
        Pipeline:
        1. Build world state
        2. Group into scenes
        3. Index evidence with provenance
        4. Build entity memories
        5. Detect state transitions
        6. Build narrative elements
        7. Create collection memory
        """
        # 1. World state
        world_state = self._world_builder.add_observations_batch(observations)
        
        # 2. Scene grouping
        scenes = self._scene_grouper.group_observations(observations, world_state)
        
        # 3. Index evidence with provenance
        self._index_evidence_with_provenance(observations, scenes)
        
        # 4. Entity memories
        entity_memories = self._entity_tracker.build_entity_memory(scenes, observations, world_state)
        entity_memories = self._entity_tracker.update_entity_state(entity_memories, scenes)
        
        # 5. State transitions
        transitions = self._transition_detector.detect_transitions(entity_memories, scenes, observations)
        
        # 6. Narrative elements
        narrative_elements = self._narrative_memory.build_narrative_elements(
            entity_memories, scenes, transitions, observations, world_state
        )
        
        # Convert entity_memories to use normalized_label as key for global_entities
        global_entities = {}
        for entity_mem in entity_memories.values():
            global_entities[entity_mem.normalized_label] = entity_mem
        
        # 7. Create collection memory
        collection = CollectionMemory(
            collection_id=self._collection_id,
            scene_summaries=scenes,
            global_entities=global_entities,
            state_transitions=transitions,
            narrative_elements=narrative_elements,
            total_images=len(observations),
            total_evidence=sum(len(obs.evidence_records) for obs in observations),
            created_at=time.time(),
            updated_at=time.time(),
        )
        
        # Set collection in retriever
        self._retriever.set_collection_memory(collection)
        
        return collection
    
    def _index_evidence_with_provenance(
        self,
        observations: list[VisualObservations],
        scenes: list[SceneSummary],
    ) -> None:
        """Index all evidence with scene/image/frame provenance."""
        # Build frame->scene and image->scene mappings
        frame_to_scene = {}
        image_to_scene = {}
        
        for scene in scenes:
            for frame_idx in scene.frame_indices:
                frame_to_scene[frame_idx] = scene.scene_id
            for img_id in scene.image_ids:
                image_to_scene[img_id] = scene.scene_id
        
        # Prepare batch data
        all_evidence = []
        all_texts = []
        all_image_ids = []
        all_scene_ids = []
        all_frame_ids = []
        
        for obs in observations:
            scene_id = image_to_scene.get(obs.image_id) or frame_to_scene.get(obs.frame_id, "unknown")
            
            for evidence in obs.evidence_records:
                text = self._evidence_to_text(evidence)
                all_evidence.append(evidence)
                all_texts.append(text)
                all_image_ids.append(obs.image_id)
                all_scene_ids.append(scene_id)
                all_frame_ids.append(obs.frame_id)
        
        if all_evidence:
            self._vector_store.add_batch_with_provenance(
                all_evidence, all_texts, all_image_ids, all_scene_ids, all_frame_ids, self._collection_id
            )
    
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
    
    def get_retriever(self) -> HierarchicalRetriever:
        """Get the hierarchical retriever for querying."""
        return self._retriever
    
    def get_collection_id(self) -> str:
        return self._collection_id


def create_collection_memory_builder(
    device: str = "cpu",
    similarity_threshold: float = 0.99,
    top_k_scenes: int = 5,
    top_k_evidence_per_scene: int = 10,
) -> CollectionMemoryBuilder:
    """Factory function to create collection memory builder."""
    embedding_model = create_embedding_model(device=device)
    vector_store = FAISSVectorStore(embedding_model, device=device)
    
    scene_config = SceneGroupConfig(similarity_threshold=similarity_threshold)
    retrieval_config = HierarchicalRetrievalConfig(
        top_k_scenes=top_k_scenes,
        top_k_evidence_per_scene=top_k_evidence_per_scene,
    )
    
    return CollectionMemoryBuilder(
        embedding_model=embedding_model,
        vector_store=vector_store,
        device=device,
        scene_config=scene_config,
        retrieval_config=retrieval_config,
    )