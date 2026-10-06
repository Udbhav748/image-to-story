"""Unit tests for context module."""
import pytest
from image_story.domain.schemas import (
    VisualObservations,
    WorldState,
    WorldEntity,
    EvidenceRecord,
    EvidenceType,
    InformationClass,
    SourceModel,
    BoundingBox,
    RetrievedEvidence,
)
from image_story.context.builder import ContextBuilder
from image_story.context.ranker import EvidenceRanker, RankingWeights
from image_story.context.world_state import EntityTracker, WorldStateBuilder


class TestContextBuilder:
    def setup_method(self):
        self.builder = ContextBuilder(max_words=200)
    
    def test_build_single_image_context(self):
        obs = VisualObservations(
            image_id="test_001",
            frame_id=0,
            scene="A park scene",
            characters=["girl", "boy"],
            od_labels=["tree", "bench"],
            actions=["sitting", "talking"],
            spatial_relations=["next to"],
            style_or_mood="peaceful",
        )
        
        context = self.builder.build_single_image_context(obs)
        
        assert "Scene: A park scene" in context
        assert "Characters: girl, boy" in context
        assert "Objects: tree, bench" in context
        assert "Actions: sitting, talking" in context
        assert "Spatial: next to" in context
        assert "Style: peaceful" in context
    
    def test_build_context_with_facts_inferences_creative(self):
        obs = VisualObservations(
            image_id="test_001",
            frame_id=0,
            scene="A street",
            characters=["woman"],
            od_labels=["red backpack", "doorway"],
            actions=["waiting"],
        )
        
        # Create evidence records
        evidence = [
            EvidenceRecord(
                entity="woman",
                type=EvidenceType.PERSON,
                frame_id=0,
                confidence=0.9,
                source=SourceModel.FLORENCE2,
                evidence_text="Person detected: woman",
                information_class=InformationClass.HARD_FACT,
            ),
            EvidenceRecord(
                entity="red backpack",
                type=EvidenceType.OBJECT,
                frame_id=0,
                confidence=0.85,
                source=SourceModel.GROUNDING_DINO,
                evidence_text="Grounded detection: red backpack",
                information_class=InformationClass.HARD_FACT,
            ),
            EvidenceRecord(
                entity="waiting",
                type=EvidenceType.ACTION,
                frame_id=0,
                confidence=0.7,
                source=SourceModel.FLORENCE2,
                evidence_text="Action detected: waiting",
                information_class=InformationClass.SOFT_INFERENCE,
            ),
        ]
        
        retrieved = [
            RetrievedEvidence(record=e, semantic_similarity=0.9) for e in evidence
        ]
        
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="woman", entity_type="character", first_frame=0, last_frame=0)
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="red backpack", entity_type="object", first_frame=0, last_frame=0)
        ]
        
        context = self.builder.build_context(
            [obs], world_state, retrieved, None
        )
        
        assert "HARD FACTS" in context
        assert "SOFT INFERENCES" in context
        assert "CREATIVE SPACE" in context
        assert "woman" in context
        assert "red backpack" in context
        assert "waiting" in context


class TestSequenceContextBuilder:
    def test_build_sequence_context(self):
        from image_story.context.sequence import SequenceContextBuilder
        
        builder = SequenceContextBuilder()
        
        obs1 = VisualObservations(
            image_id="frame_0",
            frame_id=0,
            scene="Street",
            characters=["girl"],
            od_labels=["car"],
            actions=["walking"],
        )
        obs2 = VisualObservations(
            image_id="frame_1",
            frame_id=1,
            scene="Park",
            characters=["girl"],
            od_labels=["tree"],
            actions=["sitting"],
        )
        
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1, frames_present=[0, 1], is_recurring=True)
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="car", entity_type="object", first_frame=0, last_frame=0),
            WorldEntity(id="o2", label="tree", entity_type="object", first_frame=1, last_frame=1),
        ]
        
        context = builder.build_sequence_context([obs1, obs2], world_state, [], None)
        
        assert "SEQUENCE OF IMAGES" in context
        assert "IMAGE 1" in context
        assert "IMAGE 2" in context
        assert "CONTINUITY NOTES" in context
        assert "Recurring characters" in context
        assert "girl" in context


class TestEvidenceRanker:
    def setup_method(self):
        self.ranker = EvidenceRanker()
    
    def test_rank_evidence(self):
        evidence = [
            EvidenceRecord(entity="woman", type=EvidenceType.PERSON, frame_id=0, confidence=0.9, source=SourceModel.FLORENCE2),
            EvidenceRecord(entity="car", type=EvidenceType.OBJECT, frame_id=0, confidence=0.5, source=SourceModel.FLORENCE2),
            EvidenceRecord(entity="tree", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8, source=SourceModel.GROUNDING_DINO),
        ]
        
        retrieved = [
            RetrievedEvidence(record=e, semantic_similarity=0.8) for e in evidence
        ]
        
        ranked = self.ranker.rank(retrieved, "woman street", current_frame=0, entity_recurrence={"woman": 2})
        
        # Higher confidence + recurrence should rank higher
        assert ranked[0].record.entity == "woman"
        assert ranked[0].rank_score > ranked[1].rank_score
    
    def test_select_for_context(self):
        evidence = [
            EvidenceRecord(entity=f"obj_{i}", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8, evidence_text="x " * 50)
            for i in range(10)
        ]
        retrieved = [RetrievedEvidence(record=e, semantic_similarity=0.8) for e in evidence]
        
        ranked = self.ranker.rank(retrieved, "", current_frame=0, entity_recurrence={})
        selected = self.ranker.select_for_context(ranked, max_tokens=200)
        
        assert len(selected) < len(retrieved)


class TestEntityTracker:
    def test_track_recurring_entity(self):
        tracker = EntityTracker()
        world_state = WorldState()
        
        obs1 = VisualObservations(
            image_id="f0", frame_id=0,
            characters=["girl"], od_labels=["car"],
            evidence_records=[
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
                EvidenceRecord(entity="car", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8),
            ]
        )
        
        tracker.update_from_observations(world_state, obs1)
        assert len(world_state.characters) == 1
        assert world_state.characters[0].label == "girl"
        
        obs2 = VisualObservations(
            image_id="f1", frame_id=1,
            characters=["girl"], od_labels=["tree"],
            evidence_records=[
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=1, confidence=0.9),
                EvidenceRecord(entity="tree", type=EvidenceType.OBJECT, frame_id=1, confidence=0.8),
            ]
        )
        
        tracker.update_from_observations(world_state, obs2)
        
        girl = world_state.get_entity_by_label("girl")
        assert girl.is_recurring is True
        assert girl.frames_present == [0, 1]
    
    def test_detect_disappearance(self):
        tracker = EntityTracker()
        world_state = WorldState()
        
        # Frame 0: backpack present
        obs1 = VisualObservations(
            image_id="f0", frame_id=0,
            characters=["girl"], od_labels=["backpack"],
            evidence_records=[
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
                EvidenceRecord(entity="backpack", type=EvidenceType.OBJECT, frame_id=0, confidence=0.9),
            ]
        )
        tracker.update_from_observations(world_state, obs1)
        
        # Frame 1: girl still there
        obs2 = VisualObservations(
            image_id="f1", frame_id=1,
            characters=["girl"], od_labels=[],
            evidence_records=[
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=1, confidence=0.9),
            ]
        )
        tracker.update_from_observations(world_state, obs2)
        
        # Frame 2: backpack gone (disappearance detected)
        obs3 = VisualObservations(
            image_id="f2", frame_id=2,
            characters=["girl"], od_labels=[],
            evidence_records=[
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=2, confidence=0.9),
            ]
        )
        tracker.update_from_observations(world_state, obs3)
        
        backpack = world_state.get_entity_by_label("backpack")
        assert backpack is not None
        assert backpack.disappearance_frame == 0
        
        # Open loop should be created
        assert len(world_state.open_loops) > 0
        loop = world_state.open_loops[0]
        assert loop["type"] == "disappearance"
        assert "backpack" in loop["description"]


class TestWorldStateBuilder:
    def test_build_from_observations(self):
        builder = WorldStateBuilder()
        
        obs1 = VisualObservations(
            image_id="f0", frame_id=0,
            characters=["girl"], od_labels=["car"],
            evidence_records=[
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=0, confidence=0.9),
                EvidenceRecord(entity="car", type=EvidenceType.OBJECT, frame_id=0, confidence=0.8),
            ]
        )
        
        state = builder.add_observations(obs1)
        assert len(state.characters) == 1
        assert len(state.objects) == 1
        
        obs2 = VisualObservations(
            image_id="f1", frame_id=1,
            characters=["girl", "boy"], od_labels=["tree"],
            evidence_records=[
                EvidenceRecord(entity="girl", type=EvidenceType.PERSON, frame_id=1, confidence=0.9),
                EvidenceRecord(entity="boy", type=EvidenceType.PERSON, frame_id=1, confidence=0.8),
                EvidenceRecord(entity="tree", type=EvidenceType.OBJECT, frame_id=1, confidence=0.8),
            ]
        )
        
        state = builder.add_observations(obs2)
        assert len(state.characters) == 2
        assert len(state.objects) == 2
        
        girl = state.get_entity_by_label("girl")
        assert girl.is_recurring is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])