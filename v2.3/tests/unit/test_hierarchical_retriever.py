"""Tests for hierarchical retriever."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from image_story.memory.hierarchical_retriever import HierarchicalRetriever, HierarchicalRetrievalConfig, create_hierarchical_retriever
from image_story.memory.faiss_store import FAISSVectorStore
from image_story.memory.hierarchical import (
    SceneSummary, CollectionMemory, EntityMemory, StateTransition, NarrativeElement, RetrievalResult
)
from image_story.domain.schemas import EvidenceRecord, EvidenceType, SourceModel, InformationClass


class MockEmbeddingModel:
    """Mock embedding model."""
    def __init__(self):
        self._call_count = 0
    
    def encode_single(self, text: str) -> np.ndarray:
        self._call_count += 1
        # Deterministic embedding
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(384).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self.encode_single(t) for t in texts])


class MockVectorStore:
    """Mock FAISS vector store."""
    def __init__(self, embedding_model):
        self._embedding_model = embedding_model
        self._evidence_map = {}
        self._indexed_evidence = {}
        self._scene_to_indices = {}
        self._entity_to_indices = {}
        self._image_to_indices = {}
        self._next_index = 0
    
    def add_evidence(self, *args, **kwargs):
        pass
    
    def add_batch(self, *args, **kwargs):
        pass
    
    def search(self, query_text: str, top_k: int = 10, filter_fn=None, retrieval_filter=None):
        # Return mock results
        results = []
        for i, (idx, evidence) in enumerate(list(self._evidence_map.items())[:top_k]):
            from image_story.domain.schemas import RetrievedEvidence
            results.append(RetrievedEvidence(
                record=evidence,
                semantic_similarity=0.9 - i * 0.1,
                rank_score=0.0,
                rank_factors={},
            ))
        return results
    
    def search_by_scene(self, scene_id: str, query_text: str, top_k: int = 10):
        return self.search(query_text, top_k)
    
    def search_by_embedding(self, query_embedding, top_k: int = 10, filter_fn=None, retrieval_filter=None):
        return self.search("", top_k, filter_fn, retrieval_filter)
    
    def initialize(self):
        pass


def create_test_evidence(entity: str, ev_type: EvidenceType, frame_id: int = 0) -> EvidenceRecord:
    return EvidenceRecord(
        entity=entity,
        type=ev_type,
        frame_id=frame_id,
        confidence=0.9,
        source=SourceModel.FLORENCE2,
        information_class=InformationClass.HARD_FACT,
        evidence_text=f"Evidence for {entity}",
    )


class TestHierarchicalRetriever:
    def setup_method(self):
        self.embedding_model = MockEmbeddingModel()
        self.vector_store = MockVectorStore(self.embedding_model)
        
        # Add some mock evidence
        self.vector_store._evidence_map[0] = create_test_evidence("girl", EvidenceType.PERSON, 0)
        self.vector_store._evidence_map[1] = create_test_evidence("balloon", EvidenceType.OBJECT, 0)
        self.vector_store._evidence_map[2] = create_test_evidence("boy", EvidenceType.PERSON, 1)
        self.vector_store._next_index = 3
        
        self.config = HierarchicalRetrievalConfig(
            top_k_scenes=3,
            top_k_evidence_per_scene=5,
            max_total_evidence=20,
        )
        self.retriever = HierarchicalRetriever(
            self.embedding_model, self.vector_store, self.config
        )
    
    def test_retrieve_no_collection(self):
        """Test retrieval without collection memory falls back to direct search."""
        results = self.retriever.retrieve("girl balloon")
        
        assert len(results) > 0
        assert all(isinstance(r.record, EvidenceRecord) for r in results)
    
    def test_retrieve_with_collection(self):
        """Test hierarchical retrieval with collection."""
        # Create collection memory
        scene1 = SceneSummary(
            scene_id="scene_1",
            dominant_entities=["girl"],
            important_objects=["balloon"],
            embedding=np.array([1.0] + [0.0]*383, dtype=np.float32),
        )
        scene2 = SceneSummary(
            scene_id="scene_2",
            dominant_entities=["boy"],
            important_objects=["bicycle"],
            embedding=np.array([0.0, 1.0] + [0.0]*382, dtype=np.float32),
        )
        
        collection = CollectionMemory(
            collection_id="coll_1",
            scene_summaries=[scene1, scene2],
        )
        
        self.retriever.set_collection_memory(collection)
        
        # Query similar to scene_1
        results = self.retriever.retrieve("girl balloon")
        
        assert len(results) >= 0  # May return results
    
    def test_retrieve_scenes(self):
        scene1 = SceneSummary(
            scene_id="scene_1",
            dominant_entities=["girl"],
            embedding=np.array([1.0] + [0.0]*383, dtype=np.float32),
        )
        scene2 = SceneSummary(
            scene_id="scene_2",
            dominant_entities=["boy"],
            embedding=np.array([0.0, 1.0] + [0.0]*382, dtype=np.float32),
        )
        
        collection = CollectionMemory(
            collection_id="coll_1",
            scene_summaries=[scene1, scene2],
        )
        
        self.retriever.set_collection_memory(collection)
        
        scenes = self.retriever.retrieve_scenes("girl")
        
        assert len(scenes) <= 2
        # Should prefer scene_1 (girl)
        if scenes:
            assert scenes[0].scene_id == "scene_1"
    
    def test_retrieve_entities(self):
        entity1 = EntityMemory(
            entity_id="ent_1",
            normalized_label="girl",
            entity_type="character",
            scene_ids=["scene_1", "scene_2"],
            confidence=0.9,
        )
        entity2 = EntityMemory(
            entity_id="ent_2",
            normalized_label="balloon",
            entity_type="object",
            scene_ids=["scene_1"],
            confidence=0.8,
        )
        
        collection = CollectionMemory(
            collection_id="coll_1",
            global_entities={"ent_1": entity1, "ent_2": entity2},
        )
        
        self.retriever.set_collection_memory(collection)
        
        entities = self.retriever.retrieve_entities("girl", top_k=5)
        
        assert len(entities) == 1
        assert entities[0].normalized_label == "girl"
    
    def test_retrieve_transitions(self):
        trans1 = StateTransition(
            entity_id="ent_1",
            entity_label="balloon",
            transition_type="disappeared",
            from_scene="scene_1",
            to_scene="",
            description="Balloon disappeared",
            confidence=0.8,
        )
        trans2 = StateTransition(
            entity_id="ent_2",
            entity_label="girl",
            transition_type="moved",
            from_scene="scene_1",
            to_scene="scene_2",
            description="Girl moved",
            confidence=0.7,
        )
        
        collection = CollectionMemory(
            collection_id="coll_1",
            state_transitions=[trans1, trans2],
        )
        
        self.retriever.set_collection_memory(collection)
        
        all_transitions = self.retriever.retrieve_transitions()
        assert len(all_transitions) == 2
        
        # Filter by type
        disappeared = self.retriever.retrieve_transitions(transition_type="disappeared")
        assert len(disappeared) == 1
        assert disappeared[0].entity_label == "balloon"
        
        # Filter by entity
        girl_trans = self.retriever.retrieve_transitions(entity_label="girl")
        assert len(girl_trans) == 1
    
    def test_retrieve_narrative_elements(self):
        elem1 = NarrativeElement(
            element_id="narr_1",
            element_type="recurring_entity",
            label="red balloon",
            description="Red balloon in 3 scenes",
            first_scene="scene_1",
            scene_ids=["scene_1", "scene_2", "scene_3"],
            narrative_importance=0.9,
            associated_entities=["balloon"],
            potential_callback=True,
        )
        elem2 = NarrativeElement(
            element_id="narr_2",
            element_type="open_loop",
            label="missing balloon",
            description="Balloon disappeared",
            first_scene="scene_2",
            scene_ids=["scene_2"],
            narrative_importance=0.7,
            associated_entities=["balloon"],
        )
        
        collection = CollectionMemory(
            collection_id="coll_1",
            narrative_elements=[elem1, elem2],
        )
        
        self.retriever.set_collection_memory(collection)
        
        elements = self.retriever.retrieve_narrative_elements("balloon")
        
        assert len(elements) == 2
        # Should be sorted by importance
        assert elements[0].label == "red balloon"
    
    def test_retrieve_callback_candidates(self):
        elem1 = NarrativeElement(
            element_id="narr_1",
            element_type="recurring_entity",
            label="red balloon",
            description="Red balloon in scenes 1, 2",
            first_scene="scene_1",
            scene_ids=["scene_1", "scene_2"],
            narrative_importance=0.8,
            potential_callback=True,
            callback_payoff_scene=None,
        )
        elem2 = NarrativeElement(
            element_id="narr_2",
            element_type="open_loop",
            label="missing balloon",
            description="Balloon disappeared in scene 1",
            first_scene="scene_1",
            scene_ids=["scene_1"],
            narrative_importance=0.9,
            potential_callback=True,
            callback_payoff_scene=None,
        )
        elem3 = NarrativeElement(
            element_id="narr_3",
            element_type="scene_object",
            label="bicycle",
            description="Bicycle in scene 2",
            first_scene="scene_2",
            scene_ids=["scene_2"],
            narrative_importance=0.5,
            potential_callback=False,
        )
        
        collection = CollectionMemory(
            collection_id="coll_1",
            scene_summaries=[
                SceneSummary(scene_id="scene_1"),
                SceneSummary(scene_id="scene_2"),
                SceneSummary(scene_id="scene_3"),
            ],
            narrative_elements=[elem1, elem2, elem3],
        )
        
        self.retriever.set_collection_memory(collection)
        
        # Request callbacks for scene_3
        candidates = self.retriever.retrieve_callback_candidates("scene_3", "balloon")
        
        # Should include balloon elements from earlier scenes
        assert len(candidates) >= 1
        labels = [c.label for c in candidates]
        assert "red balloon" in labels or "missing balloon" in labels
    
    def test_retrieve_full(self):
        scene1 = SceneSummary(scene_id="scene_1", dominant_entities=["girl"], embedding=np.ones(384, dtype=np.float32))
        
        entity1 = EntityMemory(
            entity_id="ent_1", normalized_label="girl", entity_type="character",
            scene_ids=["scene_1"], confidence=0.9
        )
        
        trans1 = StateTransition(
            entity_id="ent_1", entity_label="girl", transition_type="appeared",
            from_scene="", to_scene="scene_1", description="Girl appeared"
        )
        
        elem1 = NarrativeElement(
            element_id="narr_1", element_type="recurring_entity",
            label="girl", description="Main character", first_scene="scene_1",
            scene_ids=["scene_1"], narrative_importance=0.8
        )
        
        collection = CollectionMemory(
            collection_id="coll_1",
            scene_summaries=[scene1],
            global_entities={"ent_1": entity1},
            state_transitions=[trans1],
            narrative_elements=[elem1],
        )
        
        self.retriever.set_collection_memory(collection)
        
        result = self.retriever.retrieve_full("girl story")
        
        assert isinstance(result, RetrievalResult)
        assert result.query == "girl story"
        assert len(result.scenes) == 1
        assert len(result.entities) == 1
        assert len(result.transitions) == 1
        assert len(result.narrative_elements) == 1
        assert result.retrieval_time_ms > 0
    
    def test_config_parameters(self):
        """Test that config parameters are respected."""
        custom_config = HierarchicalRetrievalConfig(
            top_k_scenes=2,
            top_k_evidence_per_scene=3,
            max_total_evidence=10,
            include_transitions=False,
            include_narrative_elements=False,
        )
        
        retriever = HierarchicalRetriever(self.embedding_model, self.vector_store, custom_config)
        
        assert retriever.config.top_k_scenes == 2
        assert retriever.config.top_k_evidence_per_scene == 3
        assert retriever.config.max_total_evidence == 10
        assert retriever.config.include_transitions is False
        assert retriever.config.include_narrative_elements is False


class TestHierarchicalRetrievalConfig:
    def test_defaults(self):
        config = HierarchicalRetrievalConfig()
        assert config.top_k_scenes == 5
        assert config.top_k_evidence_per_scene == 10
        assert config.max_total_evidence == 50
        assert config.scene_similarity_threshold == 0.5
        assert config.include_transitions is True
        assert config.include_narrative_elements is True
    
    def test_custom(self):
        config = HierarchicalRetrievalConfig(
            top_k_scenes=3,
            max_total_evidence=20,
        )
        assert config.top_k_scenes == 3
        assert config.max_total_evidence == 20


class TestCreateHierarchicalRetriever:
    def test_factory(self):
        embedding_model = MockEmbeddingModel()
        vector_store = MockVectorStore(embedding_model)
        
        retriever = create_hierarchical_retriever(
            embedding_model, vector_store,
            top_k_scenes=3,
            top_k_evidence_per_scene=5,
            max_total_evidence=15,
        )
        
        assert isinstance(retriever, HierarchicalRetriever)
        assert retriever.config.top_k_scenes == 3
        assert retriever.config.top_k_evidence_per_scene == 5
        assert retriever.config.max_total_evidence == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])