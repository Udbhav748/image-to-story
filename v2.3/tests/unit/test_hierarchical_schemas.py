"""Tests for hierarchical memory schemas."""
import pytest
import numpy as np
from image_story.memory.hierarchical import (
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
from image_story.domain.schemas import EvidenceRecord, EvidenceType, SourceModel, InformationClass, BoundingBox


class TestSceneSummary:
    def test_creation(self):
        scene = SceneSummary(
            scene_id="scene_1",
            image_ids=["img1", "img2"],
            frame_indices=[0, 1],
            dominant_entities=["girl"],
            important_objects=["balloon"],
            actions=["holding"],
            location="park",
            environment="peaceful",
            recurring_entities=["girl"],
            key_evidence_ids=["ev1", "ev2"],
            summary_text="Girl holding balloon in park",
            evidence_count=2,
            start_frame=0,
            end_frame=1,
        )
        
        assert scene.scene_id == "scene_1"
        assert len(scene.image_ids) == 2
        assert "girl" in scene.dominant_entities
    
    def test_to_dict_roundtrip(self):
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        scene = SceneSummary(
            scene_id="scene_1",
            embedding=embedding,
            summary_text="Test scene",
        )
        
        d = scene.to_dict()
        restored = SceneSummary.from_dict(d)
        
        assert restored.scene_id == scene.scene_id
        assert restored.summary_text == scene.summary_text
        assert np.allclose(restored.embedding, embedding)
    
    def test_from_dict_without_embedding(self):
        scene = SceneSummary.from_dict({"scene_id": "scene_1", "summary_text": "Test"})
        assert scene.scene_id == "scene_1"
        assert scene.embedding is None


class TestCollectionMemory:
    def test_creation(self):
        collection = CollectionMemory(
            collection_id="coll_1",
            total_images=10,
            total_evidence=50,
        )
        
        assert collection.collection_id == "coll_1"
        assert collection.total_images == 10
        assert collection.total_evidence == 50
    
    def test_get_scene(self):
        scene = SceneSummary(scene_id="scene_1", summary_text="Scene 1")
        collection = CollectionMemory(
            collection_id="coll_1",
            scene_summaries=[scene],
        )
        
        found = collection.get_scene("scene_1")
        assert found is not None
        assert found.scene_id == "scene_1"
        
        not_found = collection.get_scene("scene_999")
        assert not_found is None
    
    def test_get_scenes_by_entity(self):
        scene1 = SceneSummary(scene_id="s1", dominant_entities=["girl"], recurring_entities=["balloon"])
        scene2 = SceneSummary(scene_id="s2", dominant_entities=["boy"], recurring_entities=["balloon"])
        scene3 = SceneSummary(scene_id="s3", dominant_entities=["dog"])
        
        collection = CollectionMemory(
            collection_id="coll_1",
            scene_summaries=[scene1, scene2, scene3],
        )
        
        balloon_scenes = collection.get_scenes_by_entity("balloon")
        assert len(balloon_scenes) == 2
        
        dog_scenes = collection.get_scenes_by_entity("dog")
        assert len(dog_scenes) == 1
    
    def test_to_dict(self):
        collection = CollectionMemory(collection_id="coll_1", total_images=5)
        d = collection.to_dict()
        
        assert d["collection_id"] == "coll_1"
        assert d["total_images"] == 5
        assert "scene_summaries" in d


class TestEntityMemory:
    def test_creation(self):
        entity = EntityMemory(
            entity_id="ent_1",
            normalized_label="girl",
            aliases=["girl", "young girl"],
            entity_type="character",
            first_seen_scene="scene_1",
            last_seen_scene="scene_3",
            scene_ids=["scene_1", "scene_2", "scene_3"],
            image_ids=["img1", "img2", "img3"],
            evidence_ids=["ev1", "ev2"],
            confidence=0.9,
            state="present",
        )
        
        assert entity.entity_id == "ent_1"
        assert entity.normalized_label == "girl"
        assert entity.entity_type == "character"
        assert len(entity.scene_ids) == 3
    
    def test_to_dict_roundtrip(self):
        entity = EntityMemory(
            entity_id="ent_1",
            normalized_label="balloon",
            entity_type="object",
            scene_ids=["s1", "s2"],
            confidence=0.8,
        )
        
        d = entity.to_dict()
        restored = EntityMemory.from_dict(d)
        
        assert restored.entity_id == entity.entity_id
        assert restored.normalized_label == entity.normalized_label
        assert restored.scene_ids == entity.scene_ids


class TestStateTransition:
    def test_creation(self):
        trans = StateTransition(
            entity_id="ent_1",
            entity_label="balloon",
            transition_type="disappeared",
            from_scene="scene_1",
            to_scene="",
            description="Balloon disappeared after scene 1",
            confidence=0.8,
            evidence_ids=["ev1"],
        )
        
        assert trans.transition_type == "disappeared"
        assert trans.entity_label == "balloon"
        assert trans.confidence == 0.8
    
    def test_to_dict(self):
        trans = StateTransition(
            transition_id="trans_1",
            entity_id="ent_1",
            entity_label="balloon",
            transition_type="appeared",
            from_scene="",
            to_scene="scene_1",
            description="Balloon appeared",
            confidence=0.9,
        )
        
        d = trans.to_dict()
        assert d["transition_id"] == "trans_1"
        assert d["transition_type"] == "appeared"


class TestNarrativeElement:
    def test_creation(self):
        elem = NarrativeElement(
            element_id="narr_1",
            element_type="recurring_entity",
            label="red balloon",
            description="Red balloon appears in 3 scenes",
            first_scene="scene_1",
            last_scene="scene_3",
            scene_ids=["scene_1", "scene_2", "scene_3"],
            evidence_ids=["ev1", "ev2", "ev3"],
            recurrence_count=3,
            narrative_importance=0.8,
            associated_entities=["balloon"],
            potential_callback=True,
        )
        
        assert elem.element_type == "recurring_entity"
        assert elem.potential_callback is True
        assert elem.narrative_importance == 0.8
    
    def test_to_dict(self):
        elem = NarrativeElement(
            element_id="narr_1",
            element_type="open_loop",
            label="missing balloon",
            description="Where did the balloon go?",
            first_scene="scene_2",
            scene_ids=["scene_2"],
            potential_callback=True,
            callback_payoff_scene=None,
        )
        
        d = elem.to_dict()
        assert d["element_type"] == "open_loop"
        assert d["potential_callback"] is True
        assert d["callback_payoff_scene"] is None


class TestRetrievalFilter:
    def test_create_retrieval_filter(self):
        filter_dict = create_retrieval_filter(
            scene_ids=["scene_1", "scene_2"],
            entity_labels=["girl", "balloon"],
            evidence_types=["object", "person"],
            information_classes=["hard_fact"],
            min_confidence=0.7,
            frame_range=(0, 10),
        )
        
        assert filter_dict["scene_ids"] == ["scene_1", "scene_2"]
        assert filter_dict["entity_labels"] == ["girl", "balloon"]
        assert filter_dict["evidence_types"] == ["object", "person"]
        assert filter_dict["information_classes"] == ["hard_fact"]
        assert filter_dict["min_confidence"] == 0.7
        assert filter_dict["frame_range"] == (0, 10)
    
    def test_create_retrieval_filter_partial(self):
        filter_dict = create_retrieval_filter(entity_labels=["girl"])
        assert filter_dict["entity_labels"] == ["girl"]
        assert "scene_ids" not in filter_dict
    
    def test_apply_retrieval_filter_match(self):
        evidence = EvidenceRecord(
            entity="girl",
            type=EvidenceType.PERSON,
            frame_id=5,
            confidence=0.9,
            source=SourceModel.FLORENCE2,
            information_class=InformationClass.HARD_FACT,
        )
        
        filter_dict = create_retrieval_filter(
            entity_labels=["girl"],
            evidence_types=["person"],
            min_confidence=0.8,
            frame_range=(0, 10),
        )
        
        assert apply_retrieval_filter(evidence, filter_dict) is True
    
    def test_apply_retrieval_filter_no_match_entity(self):
        evidence = EvidenceRecord(
            entity="boy",
            type=EvidenceType.PERSON,
            frame_id=5,
            confidence=0.9,
        )
        
        filter_dict = create_retrieval_filter(entity_labels=["girl"])
        assert apply_retrieval_filter(evidence, filter_dict) is False
    
    def test_apply_retrieval_filter_no_match_confidence(self):
        evidence = EvidenceRecord(
            entity="girl",
            type=EvidenceType.PERSON,
            frame_id=5,
            confidence=0.5,
        )
        
        filter_dict = create_retrieval_filter(min_confidence=0.7)
        assert apply_retrieval_filter(evidence, filter_dict) is False
    
    def test_apply_retrieval_filter_no_match_frame(self):
        evidence = EvidenceRecord(
            entity="girl",
            type=EvidenceType.PERSON,
            frame_id=20,
            confidence=0.9,
        )
        
        filter_dict = create_retrieval_filter(frame_range=(0, 10))
        assert apply_retrieval_filter(evidence, filter_dict) is False
    
    def test_apply_retrieval_filter_empty(self):
        evidence = EvidenceRecord(entity="girl", frame_id=5)
        assert apply_retrieval_filter(evidence, {}) is True
        assert apply_retrieval_filter(evidence, None) is True


class TestMemoryLevel:
    def test_values(self):
        assert MemoryLevel.EVIDENCE == "evidence"
        assert MemoryLevel.ENTITY == "entity"
        assert MemoryLevel.SCENE == "scene"
        assert MemoryLevel.EVENT == "event"
        assert MemoryLevel.NARRATIVE == "narrative"
        assert MemoryLevel.COLLECTION == "collection"


class TestIndexedEvidenceRecord:
    def test_creation(self):
        evidence = EvidenceRecord(
            entity="balloon",
            type=EvidenceType.OBJECT,
            frame_id=0,
            confidence=0.9,
        )
        
        indexed = IndexedEvidenceRecord(
            evidence=evidence,
            image_id="img_1",
            scene_id="scene_1",
            frame_id=0,
            collection_id="coll_1",
            index_position=42,
        )
        
        assert indexed.evidence.entity == "balloon"
        assert indexed.image_id == "img_1"
        assert indexed.scene_id == "scene_1"
        assert indexed.index_position == 42
    
    def test_to_dict_roundtrip(self):
        evidence = EvidenceRecord(
            entity="balloon",
            type=EvidenceType.OBJECT,
            frame_id=0,
            confidence=0.9,
        )
        
        indexed = IndexedEvidenceRecord(
            evidence=evidence,
            image_id="img_1",
            scene_id="scene_1",
            frame_id=0,
            collection_id="coll_1",
            index_position=42,
        )
        
        d = indexed.to_dict()
        restored = IndexedEvidenceRecord.from_dict(d)
        
        assert restored.evidence.entity == "balloon"
        assert restored.image_id == "img_1"
        assert restored.scene_id == "scene_1"
        assert restored.index_position == 42


class TestRetrievalResult:
    def test_creation(self):
        result = RetrievalResult(
            query="girl balloon",
            retrieval_time_ms=15.5,
        )
        
        assert result.query == "girl balloon"
        assert result.retrieval_time_ms == 15.5
        assert result.scenes == []
        assert result.evidence == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])