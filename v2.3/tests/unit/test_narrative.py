"""Unit tests for narrative module."""
import pytest
from image_story.domain.schemas import (
    WorldState,
    WorldEntity,
    CreativePlan,
    StoryPlan,
    StoryBeat,
)
from image_story.narrative.character import CharacterSystem, CharacterProfile
from image_story.narrative.conflict import ConflictEngine, ConflictType
from image_story.narrative.humor import HumorEngine, HumorStyle
from image_story.narrative.surprise import SurpriseEngine
from image_story.narrative.planner import CreativePlanner
from image_story.domain.enums import StoryGenre, StoryTone


class TestCharacterSystem:
    def test_generate_profiles(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0),
            WorldEntity(id="c2", label="boy", entity_type="character", first_frame=0, last_frame=0),
        ]
        
        system = CharacterSystem(seed=42)
        profiles = system.generate_profiles(world_state, StoryGenre.WHIMSICAL, StoryTone.COMEDIC)
        
        assert len(profiles) == 2
        for p in profiles:
            assert "label" in p
            assert "archetype" in p
            assert "personality" in p
            assert "quirk" in p
            assert "motivation" in p
            assert "goal" in p
    
    def test_deterministic_with_seed(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0),
        ]
        
        system1 = CharacterSystem(seed=123)
        profiles1 = system1.generate_profiles(world_state)
        
        system2 = CharacterSystem(seed=123)
        profiles2 = system2.generate_profiles(world_state)
        
        assert profiles1[0]["archetype"] == profiles2[0]["archetype"]
        assert profiles1[0]["quirk"] == profiles2[0]["quirk"]


class TestConflictEngine:
    def test_generate_conflict(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1, frames_present=[0, 1]),
            WorldEntity(id="c2", label="boy", entity_type="character", first_frame=1, last_frame=1, frames_present=[1]),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="backpack", entity_type="object", first_frame=0, last_frame=0, frames_present=[0]),
        ]
        
        engine = ConflictEngine(seed=42)
        conflict = engine.generate_conflict(world_state, "girl backpack street", "whimsical")
        
        assert "type" in conflict
        assert "description" in conflict
        assert "involved_entities" in conflict
        assert conflict["grounded_in_evidence"] is True
        assert conflict["type"] in [ct.value for ct in ConflictType]
    
    def test_conflict_uses_open_loops(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="backpack", entity_type="object", first_frame=0, last_frame=0, disappearance_frame=0),
        ]
        world_state.open_loops = [{
            "type": "disappearance",
            "entity_label": "backpack",
            "description": "backpack disappeared after frame 0",
        }]
        
        engine = ConflictEngine(seed=42)
        conflict = engine.generate_conflict(world_state, "girl backpack missing", "mystery")
        
        # Should pick MISSING_OBJECT due to disappearance open loop
        assert conflict["type"] == ConflictType.MISSING_OBJECT.value


class TestHumorEngine:
    def test_generate_humor_moments(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="statue", entity_type="object", first_frame=0, last_frame=0),
        ]
        
        engine = HumorEngine(seed=42, humor_level=1.0)  # Force humor
        moments = engine.generate_humor_moments(world_state, {}, HumorStyle.SITUATIONAL, count=2)
        
        assert len(moments) <= 2
        assert all(isinstance(m, str) for m in moments)
        assert all(len(m) > 0 for m in moments)
    
    def test_humor_level_zero(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0),
        ]
        
        engine = HumorEngine(seed=42, humor_level=0.0)  # No humor
        moments = engine.generate_humor_moments(world_state, {}, HumorStyle.SITUATIONAL, count=2)
        
        assert len(moments) == 0


class TestSurpriseEngine:
    def test_generate_surprise(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="red backpack", entity_type="object", first_frame=0, last_frame=0, disappearance_frame=0),
        ]
        
        foreshadowing = [
            {"detail": "red backpack", "frame": 0, "type": "object"},
        ]
        
        engine = SurpriseEngine(seed=42, surprise_level=1.0)
        surprise = engine.generate_surprise(world_state, "girl backpack street", foreshadowing)
        
        assert surprise is not None
        assert "type" in surprise
        assert "description" in surprise
        assert "pattern" in surprise
    
    def test_callback_payoff(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1),
        ]
        
        foreshadowing = [
            {"detail": "red umbrella", "frame": 0, "type": "object"},
        ]
        
        engine = SurpriseEngine(seed=42, surprise_level=1.0)
        surprise = engine.generate_surprise(world_state, "", foreshadowing)
        
        # Might generate callback_payoff
        if surprise and surprise["type"] == "callback_payoff":
            assert "umbrella" in surprise["description"]


class TestCreativePlanner:
    def test_create_creative_plan(self):
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="red backpack", entity_type="object", first_frame=0, last_frame=0),
        ]
        world_state.open_loops = [{
            "description": "backpack disappeared",
        }]
        
        from image_story.domain.schemas import RetrievedEvidence, EvidenceRecord, EvidenceType, SourceModel, InformationClass
        
        evidence = [
            EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
            EvidenceRecord(entity="red backpack", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8),
        ]
        retrieved = [
            RetrievedEvidence(record=e, semantic_similarity=0.8) for e in evidence
        ]
        
        planner = CreativePlanner(
            seed=42,
            creativity_config={"creativity": 0.7, "surprise": 0.6, "humor": 0.5},
            genre="whimsical",
            tone="comedic",
        )
        
        creative_plan = planner.create_creative_plan(
            world_state, [], retrieved
        )
        
        assert creative_plan.genre == "whimsical"
        assert creative_plan.tone == "comedic"
        assert len(creative_plan.characters) == 1
        assert creative_plan.central_conflict is not None
        assert len(creative_plan.open_loops) > 0
        assert len(creative_plan.foreshadowing_elements) > 0
    
    def test_create_story_plan(self):
        creative_plan = CreativePlan(
            genre="whimsical",
            tone="comedic",
            characters=[{"label": "girl", "goal": "find backpack"}],
            central_conflict={"description": "backpack missing", "involved_entities": ["girl"]},
            open_loops=["backpack disappeared"],
        )
        
        planner = CreativePlanner(seed=42)
        story_plan = planner.create_story_plan(creative_plan, target_words=250)
        
        assert len(story_plan.beats) == 7
        beat_types = [b.beat_type for b in story_plan.beats]
        assert beat_types == ["setup", "goal", "conflict", "escalation", "surprise", "resolution", "callback"]
        assert story_plan.target_total_words == 250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])