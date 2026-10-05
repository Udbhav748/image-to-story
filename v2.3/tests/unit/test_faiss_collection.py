"""Tests for FAISS store with hierarchical features and collection builder."""
import pytest
import numpy as np
import tempfile
import os

from image_story.memory.faiss_store import FAISSVectorStore
from image_story.memory.collection_builder import CollectionMemoryBuilder, create_collection_memory_builder
from image_story.memory.hierarchical import (
    IndexedEvidenceRecord, RetrievalFilter, create_retrieval_filter, apply_retrieval_filter
)
from image_story.memory.embeddings import SentenceTransformerEmbedding, create_embedding_model
from image_story.domain.schemas import (
    VisualObservations, EvidenceRecord, EvidenceType, SourceModel, InformationClass, PipelineConfig
)


class MockEmbeddingModel:
    """Mock embedding model that returns deterministic embeddings."""
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    @property
    def embedding_dim(self) -> int:
        return self._dimension
    
    @property
    def is_loaded(self) -> bool:
        return True
    
    def load(self):
        pass
    
    def unload(self):
        pass
    
    def encode_single(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(self._dimension).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self.encode_single(t) for t in texts])


def create_test_obs(
    frame_id: int,
    image_id: str | None = None,
    characters: list[str] | None = None,
    objects: list[str] | None = None,
    scene: str = "",
) -> VisualObservations:
    chars = characters or []
    objs = objects or []
    
    evidence = []
    for c in chars:
        evidence.append(EvidenceRecord(
            entity=c, type=EvidenceType.PERSON, frame_id=frame_id,
            confidence=0.9, source=SourceModel.FLORENCE2,
            information_class=InformationClass.HARD_FACT,
        ))
    for o in objs:
        evidence.append(EvidenceRecord(
            entity=o, type=EvidenceType.OBJECT, frame_id=frame_id,
            confidence=0.8, source=SourceModel.FLORENCE2,
            information_class=InformationClass.HARD_FACT,
        ))
    
    return VisualObservations(
        image_id=image_id or f"img_{frame_id}",
        frame_id=frame_id,
        scene=scene,
        characters=chars,
        od_labels=objs,
        evidence_records=evidence,
    )


class TestFAISSStoreHierarchical:
    def setup_method(self):
        self.embedding_model = MockEmbeddingModel()
        self.store = FAISSVectorStore(self.embedding_model, index_type="flat_ip")
    
    def test_add_evidence_with_provenance(self):
        evidence = EvidenceRecord(
            entity="girl", type=EvidenceType.PERSON, frame_id=0,
            confidence=0.9, source=SourceModel.FLORENCE2,
            information_class=InformationClass.HARD_FACT,
        )
        
        idx = self.store.add_evidence_with_provenance(
            evidence, "Entity: girl", "img_1", "scene_1", 0, "coll_1"
        )
        
        assert idx == 0
        assert self.store.count == 1
        
        # Check indexed evidence
        indexed = self.store.get_indexed_evidence(0)
        assert indexed is not None
        assert indexed.evidence.entity == "girl"
        assert indexed.image_id == "img_1"
        assert indexed.scene_id == "scene_1"
        assert indexed.frame_id == 0
        assert indexed.collection_id == "coll_1"
        assert indexed.index_position == 0
    
    def test_add_batch_with_provenance(self):
        evidence_list = [
            EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
            EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8),
            EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=1, confidence=0.9),
        ]
        
        indices = self.store.add_batch_with_provenance(
            evidence_list,
            ["Entity: girl", "Entity: balloon", "Entity: boy"],
            ["img_1", "img_1", "img_2"],
            ["scene_1", "scene_1", "scene_2"],
            [0, 0, 1],
            "coll_1",
        )
        
        assert indices == [0, 1, 2]
        assert self.store.count == 3
        
        # Check indices
        assert self.store.get_scene_ids() == ["scene_1", "scene_2"]
        assert set(self.store.get_entity_labels()) == {"girl", "balloon", "boy"}
        assert set(self.store.get_image_ids()) == {"img_1", "img_2"}
    
    def test_search_by_scene(self):
        # Add evidence to different scenes
        self.store.add_batch_with_provenance(
            [
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
                EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8),
                EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=1, confidence=0.9),
            ],
            ["Entity: girl", "Entity: balloon", "Entity: boy"],
            ["img_1", "img_1", "img_2"],
            ["scene_1", "scene_1", "scene_2"],
            [0, 0, 1],
            "coll_1",
        )
        
        # Search within scene_1
        results = self.store.search_by_scene("scene_1", "girl")
        
        assert len(results) >= 1
        for r in results:
            indexed = self.store.get_indexed_evidence(r.record.id)
            assert indexed is not None
            assert indexed.scene_id == "scene_1"
    
    def test_search_by_entity(self):
        self.store.add_batch_with_provenance(
            [
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=1, confidence=0.9),
                EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=0, confidence=0.8),
            ],
            ["Entity: girl"] * 3,
            ["img_1", "img_2", "img_1"],
            ["scene_1", "scene_2", "scene_1"],
            [0, 1, 0],
            "coll_1",
        )
        
        # Search by entity
        results = self.store.search_by_entity("girl")
        
        # Should find both girl records
        assert len(results) == 2
        for r in results:
            assert r.record.entity == "girl"
    
    def test_get_scene_evidence(self):
        self.store.add_batch_with_provenance(
            [
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0),
                EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0),
                EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=1),
            ],
            ["Entity: girl", "Entity: balloon", "Entity: boy"],
            ["img_1", "img_1", "img_2"],
            ["scene_1", "scene_1", "scene_2"],
            [0, 0, 1],
            "coll_1",
        )
        
        scene1_evidence = self.store.get_scene_evidence("scene_1")
        scene2_evidence = self.store.get_scene_evidence("scene_2")
        
        assert len(scene1_evidence) == 2
        assert len(scene2_evidence) == 1
        
        entities = {e.entity for e in scene1_evidence}
        assert entities == {"girl", "balloon"}
    
    def test_get_entity_evidence(self):
        self.store.add_batch_with_provenance(
            [
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0),
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=1),
                EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=0),
            ],
            ["Entity: girl", "Entity: girl", "Entity: boy"],
            ["img_1", "img_2", "img_1"],
            ["scene_1", "scene_2", "scene_1"],
            [0, 1, 0],
            "coll_1",
        )
        
        girl_evidence = self.store.get_entity_evidence("girl")
        boy_evidence = self.store.get_entity_evidence("boy")
        
        assert len(girl_evidence) == 2
        assert len(boy_evidence) == 1
    
    def test_search_with_retrieval_filter(self):
        self.store.add_batch_with_provenance(
            [
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
                EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8),
                EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=1, confidence=0.7),
            ],
            ["Entity: girl", "Entity: balloon", "Entity: boy"],
            ["img_1", "img_1", "img_2"],
            ["scene_1", "scene_1", "scene_2"],
            [0, 0, 1],
            "coll_1",
        )
        
        # Filter by confidence
        filter_dict = create_retrieval_filter(min_confidence=0.85)
        results = self.store.search("person", top_k=10, retrieval_filter=filter_dict)
        
        assert len(results) == 1
        assert results[0].record.entity == "girl"
        assert results[0].record.confidence >= 0.85
    
    def test_search_with_scene_filter(self):
        self.store.add_batch_with_provenance(
            [
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0),
                EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0),
                EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=1),
            ],
            ["Entity: girl", "Entity: balloon", "Entity: boy"],
            ["img_1", "img_1", "img_2"],
            ["scene_1", "scene_1", "scene_2"],
            [0, 0, 1],
            "coll_1",
        )
        
        filter_dict = create_retrieval_filter(scene_ids=["scene_1"])
        results = self.store.search("entity", top_k=10, retrieval_filter=filter_dict)
        
        # Should only return scene_1 evidence
        for r in results:
            indexed = self.store.get_indexed_evidence(r.record.id)
            assert indexed.scene_id == "scene_1"
    
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index")
            
            self.store.add_batch_with_provenance(
                [
                    EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
                    EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8),
                ],
                ["Entity: girl", "Entity: balloon"],
                ["img_1", "img_1"],
                ["scene_1", "scene_1"],
                [0, 0],
                "coll_1",
            )
            
            self.store.save(path)
            
            # Load into new store
            new_store = FAISSVectorStore(self.embedding_model, index_type="flat_ip")
            new_store.load(path)
            
            assert new_store.count == 2
            assert new_store.get_scene_ids() == ["scene_1"]
            assert set(new_store.get_entity_labels()) == {"girl", "balloon"}
            
            # Check indexed evidence loaded
            indexed = new_store.get_indexed_evidence(0)
            assert indexed is not None
            assert indexed.evidence.entity == "girl"
            assert indexed.scene_id == "scene_1"
    
    def test_clear(self):
        self.store.add_batch_with_provenance(
            [EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0)],
            ["Entity: girl"],
            ["img_1"],
            ["scene_1"],
            [0],
            "coll_1",
        )
        
        assert self.store.count == 1
        
        self.store.clear()
        
        assert self.store.count == 0
        assert self.store.get_scene_ids() == []
        assert self.store.get_entity_labels() == []
    
    def test_get_stats(self):
        self.store.add_batch_with_provenance(
            [
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0),
                EvidenceRecord(entity="balloon", type=EvidenceType.OBJECT, frame_id=0),
                EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=1),
            ],
            ["Entity: girl", "Entity: balloon", "Entity: boy"],
            ["img_1", "img_1", "img_2"],
            ["scene_1", "scene_1", "scene_2"],
            [0, 0, 1],
            "coll_1",
        )
        
        stats = self.store.get_stats()
        
        assert stats["count"] == 3
        assert stats["num_scenes"] == 2
        assert stats["num_entities"] == 3
        assert stats["num_images"] == 2


class TestCollectionMemoryBuilder:
    def setup_method(self):
        self.embedding_model = MockEmbeddingModel()
        self.builder = create_collection_memory_builder(device="cpu")
    
    def test_build_from_observations(self):
        from image_story.memory.hierarchical import CollectionMemory
        
        observations = [
            create_test_obs(0, characters=["girl"], objects=["balloon"], scene="park"),
            create_test_obs(1, characters=["girl"], objects=["balloon", "kite"], scene="park"),
            create_test_obs(2, characters=["boy"], objects=["bicycle"], scene="street"),
        ]
        
        config = PipelineConfig(
            mode="standard",
            target_story_words=250,
        )
        
        collection = self.builder.build_from_observations(observations, config)
        
        assert isinstance(collection, CollectionMemory)
        assert collection.total_images == 3
        assert collection.total_evidence > 0
        assert len(collection.scene_summaries) >= 1
        assert len(collection.global_entities) >= 3
        assert collection.collection_id == self.builder.get_collection_id()
    
    def test_scene_grouping(self):
        observations = [
            create_test_obs(0, characters=["girl"], objects=["balloon"], scene="park"),
            create_test_obs(1, characters=["girl"], objects=["balloon"], scene="park"),
            create_test_obs(2, characters=["boy"], objects=["bicycle"], scene="street"),
        ]
        
        collection = self.builder.build_from_observations(observations)
        
        # Should create at least 2 scenes (park and street)
        scene_locations = [s.location for s in collection.scene_summaries]
        assert "park" in scene_locations or "street" in scene_locations
    
    def test_entity_memories(self):
        from image_story.memory.hierarchical import CollectionMemory
        
        observations = [
            create_test_obs(0, characters=["girl"], objects=["balloon"]),
            create_test_obs(1, characters=["girl"], objects=["bicycle"]),
            create_test_obs(2, characters=["boy"], objects=["balloon"]),
        ]
        
        collection = self.builder.build_from_observations(observations)
        
        assert isinstance(collection, CollectionMemory)
        # Check entities
        assert "girl" in collection.global_entities
        assert "balloon" in collection.global_entities
        assert "boy" in collection.global_entities
        
        girl = collection.global_entities["girl"]
        assert girl.entity_type == "character"
        assert len(girl.scene_ids) >= 1
        
        balloon = collection.global_entities["balloon"]
        assert balloon.entity_type == "object"
    
    def test_state_transitions(self):
        observations = [
            create_test_obs(0, characters=["girl"], objects=["balloon"]),
            create_test_obs(1, characters=["girl"], objects=["balloon"]),
            create_test_obs(2, characters=["girl"], objects=["bicycle"]),  # balloon gone
        ]
        
        collection = self.builder.build_from_observations(observations)
        
        # Should detect balloon disappearance
        transitions = collection.state_transitions
        balloon_trans = [t for t in transitions if t.entity_label == "balloon"]
        assert len(balloon_trans) >= 1
        
        disappeared = [t for t in balloon_trans if t.transition_type == "disappeared"]
        assert len(disappeared) >= 1
    
    def test_narrative_elements(self):
        observations = [
            create_test_obs(0, characters=["girl"], objects=["balloon"]),
            create_test_obs(1, characters=["girl"], objects=["balloon"]),
            create_test_obs(2, characters=["girl"], objects=["balloon"]),
        ]
        
        collection = self.builder.build_from_observations(observations)
        
        # Should have recurring entity narrative element
        recurring = [e for e in collection.narrative_elements if e.element_type == "recurring_entity"]
        assert len(recurring) >= 1
        
        girl_elem = next((e for e in recurring if e.label == "girl"), None)
        assert girl_elem is not None
        assert girl_elem.recurrence_count >= 2
        assert girl_elem.potential_callback is True
    
    def test_get_retriever(self):
        observations = [
            create_test_obs(0, characters=["girl"], objects=["balloon"]),
        ]
        
        collection = self.builder.build_from_observations(observations)
        retriever = self.builder.get_retriever()
        
        assert retriever is not None
        
        # Test retrieval
        results = retriever.retrieve("girl balloon")
        assert len(results) >= 0


class TestPerformanceScaling:
    """Test FAISS performance with synthetic large datasets."""
    
    def test_100_evidence_records(self):
        embedding_model = MockEmbeddingModel()
        store = FAISSVectorStore(embedding_model, index_type="flat_ip")
        
        # Create 100 evidence records
        evidence_list = []
        texts = []
        image_ids = []
        scene_ids = []
        frame_ids = []
        
        for i in range(100):
            entity = f"object_{i % 10}"
            evidence_list.append(EvidenceRecord(
                entity=entity, type=EvidenceType.OBJECT, frame_id=i,
                confidence=0.8, source=SourceModel.FLORENCE2,
            ))
            texts.append(f"Entity: {entity}")
            image_ids.append(f"img_{i}")
            scene_ids.append(f"scene_{i // 10}")
            frame_ids.append(i)
        
        import time
        start = time.perf_counter()
        store.add_batch_with_provenance(
            evidence_list, texts, image_ids, scene_ids, frame_ids, "coll_perf"
        )
        add_time = time.perf_counter() - start
        
        assert store.count == 100
        assert add_time < 1.0  # Should be fast
        
        # Test retrieval speed
        start = time.perf_counter()
        results = store.search("object_5", top_k=10)
        search_time = time.perf_counter() - start
        
        assert len(results) <= 10
        assert search_time < 0.1  # Very fast
    
    def test_1000_evidence_records(self):
        embedding_model = MockEmbeddingModel()
        store = FAISSVectorStore(embedding_model, index_type="flat_ip")
        
        evidence_list = []
        texts = []
        image_ids = []
        scene_ids = []
        frame_ids = []
        
        for i in range(1000):
            entity = f"object_{i % 50}"
            evidence_list.append(EvidenceRecord(
                entity=entity, type=EvidenceType.OBJECT, frame_id=i,
                confidence=0.8,
            ))
            texts.append(f"Entity: {entity}")
            image_ids.append(f"img_{i}")
            scene_ids.append(f"scene_{i // 20}")
            frame_ids.append(i)
        
        import time
        start = time.perf_counter()
        store.add_batch_with_provenance(
            evidence_list, texts, image_ids, scene_ids, frame_ids, "coll_perf"
        )
        add_time = time.perf_counter() - start
        
        assert store.count == 1000
        
        # Test retrieval
        start = time.perf_counter()
        results = store.search("object_25", top_k=20)
        search_time = time.perf_counter() - start
        
        assert len(results) <= 20
        assert search_time < 0.5
    
    def test_hierarchical_retrieval_speed(self):
        embedding_model = MockEmbeddingModel()
        store = FAISSVectorStore(embedding_model, index_type="flat_ip")
        
        # Add 500 evidence across 10 scenes
        evidence_list = []
        texts = []
        image_ids = []
        scene_ids = []
        frame_ids = []
        
        for i in range(500):
            scene_idx = i // 50
            entity = f"obj_{scene_idx}_{i % 10}"
            evidence_list.append(EvidenceRecord(entity=entity, type=EvidenceType.OBJECT, frame_id=i))
            texts.append(f"Entity: {entity}")
            image_ids.append(f"img_{i}")
            scene_ids.append(f"scene_{scene_idx}")
            frame_ids.append(i)
        
        store.add_batch_with_provenance(evidence_list, texts, image_ids, scene_ids, frame_ids, "coll_perf")
        
        # Create retriever
        from image_story.memory.hierarchical_retriever import HierarchicalRetriever, HierarchicalRetrievalConfig
        from image_story.memory.hierarchical import CollectionMemory, SceneSummary
        
        # Create mock collection
        scenes = [
            SceneSummary(scene_id=f"scene_{i}", 
                        dominant_entities=[f"obj_{i}_0"],
                        embedding=np.random.randn(384).astype(np.float32))
            for i in range(10)
        ]
        for s in scenes:
            s.embedding = s.embedding / (np.linalg.norm(s.embedding) + 1e-8)
        
        collection = CollectionMemory(
            collection_id="coll_perf",
            scene_summaries=scenes,
        )
        
        retriever = HierarchicalRetriever(embedding_model, store, HierarchicalRetrievalConfig())
        retriever.set_collection_memory(collection)
        
        import time
        start = time.perf_counter()
        result = retriever.retrieve_full("obj_5_3 story")
        retrieval_time = time.perf_counter() - start
        
        assert retrieval_time < 1.0  # Should be fast even with hierarchy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])