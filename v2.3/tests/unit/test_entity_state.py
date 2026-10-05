"""Tests for entity memory tracker and state transitions."""
import pytest
import numpy as np
from unittest.mock import Mock

from image_story.memory.entity_memory import EntityMemoryTracker, create_entity_memory_tracker
from image_story.memory.state_transitions import StateTransitionDetector, create_state_transition_detector
from image_story.memory.hierarchical import SceneSummary, EntityMemory, StateTransition
from image_story.domain.schemas import (
    VisualObservations, EvidenceRecord, EvidenceType, SourceModel, InformationClass, WorldState, WorldEntity, BoundingBox
)
from image_story.context.world_state import WorldStateBuilder


class MockEmbeddingModel:
    """Mock embedding model."""
    def encode_single(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(384).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)


def create_test_obs(
    frame_id: int,
    image_id: str | None = None,
    characters: list[str] | None = None,
    objects: list[str] | None = None,
    actions: list[str] | None = None,
    scene: str = "",
) -> VisualObservations:
    """Create test observation."""
    chars = characters or []
    objs = objects or []
    acts = actions or []
    
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
    for a in acts:
        evidence.append(EvidenceRecord(
            entity=a, type=EvidenceType.ACTION, frame_id=frame_id,
            confidence=0.7, source=SourceModel.FLORENCE2,
            information_class=InformationClass.SOFT_INFERENCE,
        ))
    
    return VisualObservations(
        image_id=image_id or f"img_{frame_id}",
        frame_id=frame_id,
        scene=scene,
        characters=chars,
        od_labels=objs,
        actions=acts,
        evidence_records=evidence,
    )


class TestEntityMemoryTracker:
    def setup_method(self):
        self.embedding_model = MockEmbeddingModel()
        self.tracker = EntityMemoryTracker(self.embedding_model, similarity_threshold=0.8)
    
    def test_build_entity_memory_single_scene(self):
        scenes = [
            SceneSummary(
                scene_id="scene_1",
                frame_indices=[0, 1],
                image_ids=["img_0", "img_1"],
                dominant_entities=["girl"],
                important_objects=["balloon"],
            )
        ]
        
        observations = [
            create_test_obs(0, characters=["girl"], objects=["balloon"]),
            create_test_obs(1, characters=["girl"], objects=["balloon", "kite"]),
        ]
        
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1, frames_present=[0, 1], is_recurring=True),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="balloon", entity_type="object", first_frame=0, last_frame=1, frames_present=[0, 1], is_recurring=True),
        ]
        
        entity_memories = self.tracker.build_entity_memory(scenes, observations, world_state)
        
        assert len(entity_memories) >= 2
        girl_ent = next((e for e in entity_memories.values() if e.normalized_label == "girl"), None)
        assert girl_ent is not None
        assert girl_ent.entity_type == "character"
        assert "scene_1" in girl_ent.scene_ids
        assert len(girl_ent.image_ids) == 2
    
    def test_build_entity_memory_multiple_scenes(self):
        scenes = [
            SceneSummary(scene_id="scene_1", frame_indices=[0], image_ids=["img_0"], dominant_entities=["girl"], important_objects=["balloon"]),
            SceneSummary(scene_id="scene_2", frame_indices=[1], image_ids=["img_1"], dominant_entities=["girl"], important_objects=["bicycle"]),
            SceneSummary(scene_id="scene_3", frame_indices=[2], image_ids=["img_2"], dominant_entities=["boy"], important_objects=["balloon"]),
        ]
        
        observations = [
            create_test_obs(0, image_id="img_0", characters=["girl"], objects=["balloon"]),
            create_test_obs(1, image_id="img_1", characters=["girl"], objects=["bicycle"]),
            create_test_obs(2, image_id="img_2", characters=["boy"], objects=["balloon"]),
        ]
        
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1, frames_present=[0, 1], is_recurring=True),
            WorldEntity(id="c2", label="boy", entity_type="character", first_frame=2, last_frame=2, frames_present=[2]),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="balloon", entity_type="object", first_frame=0, last_frame=2, frames_present=[0, 1, 2], is_recurring=True),
            WorldEntity(id="o2", label="bicycle", entity_type="object", first_frame=1, last_frame=1, frames_present=[1]),
        ]
        
        entity_memories = self.tracker.build_entity_memory(scenes, observations, world_state)
        
        girl_ent = next((e for e in entity_memories.values() if e.normalized_label == "girl"), None)
        assert girl_ent is not None
        assert "scene_1" in girl_ent.scene_ids
        assert "scene_2" in girl_ent.scene_ids
        assert "scene_3" not in girl_ent.scene_ids
        
        balloon_ent = next((e for e in entity_memories.values() if e.normalized_label == "balloon"), None)
        assert balloon_ent is not None
        assert len(balloon_ent.scene_ids) == 3
    
    def test_entity_merging(self):
        scenes = [
            SceneSummary(scene_id="scene_1", frame_indices=[0], image_ids=["img_0"]),
            SceneSummary(scene_id="scene_2", frame_indices=[1], image_ids=["img_1"]),
        ]
        
        observations = [
            create_test_obs(0, image_id="img_0", characters=["girl"], objects=["red balloon"]),
            create_test_obs(1, image_id="img_1", characters=["girl"], objects=["red balloon"]),
        ]
        
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1, frames_present=[0, 1]),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="red balloon", entity_type="object", first_frame=0, last_frame=1, frames_present=[0, 1]),
        ]
        
        entity_memories = self.tracker.build_entity_memory(scenes, observations, world_state)
        
        girl_entities = [e for e in entity_memories.values() if e.normalized_label == "girl"]
        balloon_entities = [e for e in entity_memories.values() if e.normalized_label == "red_balloon"]
        
        assert len(girl_entities) <= 1
        assert len(balloon_entities) <= 1
    
    def test_update_entity_state_disappeared(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0, frame_indices=[0]),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1, frame_indices=[1]),
            SceneSummary(scene_id="scene_3", start_frame=2, end_frame=2, frame_indices=[2]),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="balloon",
                entity_type="object",
                scene_ids=["scene_1"],
                confidence=0.9,
                state="present",
            )
        }
        
        updated = self.tracker.update_entity_state(entity_memories, scenes)
        
        balloon = updated["ent_1"]
        assert balloon.state == "disappeared"
    
    def test_update_entity_state_persistent(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0, frame_indices=[0]),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1, frame_indices=[1]),
            SceneSummary(scene_id="scene_3", start_frame=2, end_frame=2, frame_indices=[2]),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="girl",
                entity_type="character",
                scene_ids=["scene_1", "scene_2", "scene_3"],
                confidence=0.9,
                state="present",
            )
        }
        
        updated = self.tracker.update_entity_state(entity_memories, scenes)
        
        girl = updated["ent_1"]
        assert girl.state == "present"


class TestStateTransitionDetector:
    def setup_method(self):
        self.detector = StateTransitionDetector(position_threshold=0.3)
    
    def test_detect_appearance(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="balloon",
                entity_type="object",
                scene_ids=["scene_1", "scene_2"],
                evidence_ids=["ev1", "ev2"],
            )
        }
        
        observations = [
            create_test_obs(0, objects=["balloon"]),
            create_test_obs(1, objects=["balloon"]),
        ]
        
        transitions = self.detector.detect_transitions(entity_memories, scenes, observations)
        
        appeared = [t for t in transitions if t.transition_type == "appeared"]
        assert len(appeared) == 1
        assert appeared[0].entity_label == "balloon"
        assert appeared[0].to_scene == "scene_1"
    
    def test_detect_disappearance(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1),
            SceneSummary(scene_id="scene_3", start_frame=2, end_frame=2),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="balloon",
                entity_type="object",
                scene_ids=["scene_1"],
                evidence_ids=["ev1"],
            )
        }
        
        observations = [
            create_test_obs(0, objects=["balloon"]),
            create_test_obs(1, objects=["bicycle"]),
            create_test_obs(2, objects=["kite"]),
        ]
        
        transitions = self.detector.detect_transitions(entity_memories, scenes, observations)
        
        disappeared = [t for t in transitions if t.transition_type == "disappeared"]
        assert len(disappeared) == 1
        assert disappeared[0].entity_label == "balloon"
        assert disappeared[0].from_scene == "scene_1"
    
    def test_detect_movement(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="girl",
                entity_type="character",
                scene_ids=["scene_1", "scene_2"],
                evidence_ids=["ev1", "ev2"],
                bounding_boxes={
                    "scene_1": [100, 100, 200, 200],
                    "scene_2": [300, 300, 400, 400],
                },
            )
        }
        
        observations = [
            create_test_obs(0, characters=["girl"]),
            create_test_obs(1, characters=["girl"]),
        ]
        
        transitions = self.detector.detect_transitions(entity_memories, scenes, observations)
        
        moved = [t for t in transitions if t.transition_type == "moved"]
        assert len(moved) == 1
        assert moved[0].entity_label == "girl"
        assert "moved significantly" in moved[0].description
    
    def test_detect_no_movement(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="girl",
                entity_type="character",
                scene_ids=["scene_1", "scene_2"],
                bounding_boxes={
                    "scene_1": [100, 100, 200, 200],
                    "scene_2": [105, 105, 205, 205],
                },
            )
        }
        
        observations = [
            create_test_obs(0, characters=["girl"]),
            create_test_obs(1, characters=["girl"]),
        ]
        
        transitions = self.detector.detect_transitions(entity_memories, scenes, observations)
        
        moved = [t for t in transitions if t.transition_type == "moved"]
        assert len(moved) == 0
    
    def test_detect_relationship_change(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="girl",
                entity_type="character",
                scene_ids=["scene_1", "scene_2"],
            )
        }
        
        obs1 = create_test_obs(0, characters=["girl"], objects=["balloon"])
        obs1.evidence_records[0].relationship = "holding balloon"
        
        obs2 = create_test_obs(1, characters=["girl"])
        
        observations = [obs1, obs2]
        
        transitions = self.detector.detect_transitions(entity_memories, scenes, observations)
        
        rel_changed = [t for t in transitions if t.transition_type == "changed_relationship"]
        assert len(rel_changed) == 1
        assert "lost: holding balloon" in rel_changed[0].description
    
    def test_detect_state_change(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="girl",
                entity_type="character",
                scene_ids=["scene_1", "scene_2"],
            )
        }
        
        obs1 = create_test_obs(0, characters=["girl"], actions=["walking"])
        obs2 = create_test_obs(1, characters=["girl"], actions=["sitting"])
        
        observations = [obs1, obs2]
        
        transitions = self.detector.detect_transitions(entity_memories, scenes, observations)
        
        state_changed = [t for t in transitions if t.transition_type == "changed_state"]
        assert len(state_changed) == 1
        assert "stopped: walking" in state_changed[0].description
        assert "started: sitting" in state_changed[0].description
    
    def test_detect_reappearance(self):
        scenes = [
            SceneSummary(scene_id="scene_1", start_frame=0, end_frame=0),
            SceneSummary(scene_id="scene_2", start_frame=1, end_frame=1),
            SceneSummary(scene_id="scene_3", start_frame=2, end_frame=2),
        ]
        
        entity_memories = {
            "ent_1": EntityMemory(
                entity_id="ent_1",
                normalized_label="balloon",
                entity_type="object",
                scene_ids=["scene_1", "scene_3"],
                state="disappeared",
            )
        }
        
        observations = [
            create_test_obs(0, objects=["balloon"]),
            create_test_obs(1, objects=["bicycle"]),
            create_test_obs(2, objects=["balloon"]),
        ]
        
        transitions = self.detector.detect_transitions(entity_memories, scenes, observations)
        
        reappeared = [t for t in transitions if t.transition_type == "reappeared"]
        assert len(reappeared) == 1
        assert reappeared[0].entity_label == "balloon"
        assert "reappears" in reappeared[0].description


class TestCreateFunctions:
    def test_create_entity_memory_tracker(self):
        tracker = create_entity_memory_tracker(similarity_threshold=0.7)
        assert isinstance(tracker, EntityMemoryTracker)
    
    def test_create_state_transition_detector(self):
        detector = create_state_transition_detector(position_threshold=0.5)
        assert isinstance(detector, StateTransitionDetector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])