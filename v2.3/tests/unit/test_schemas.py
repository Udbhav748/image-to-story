"""Unit tests for domain schemas."""
import pytest
from image_story.domain.schemas import (
    EvidenceRecord,
    EvidenceType,
    EvidenceConfidence,
    InformationClass,
    SourceModel,
    BoundingBox,
    VisualObservations,
    WorldEntity,
    WorldState,
    CreativePlan,
    StoryPlan,
    StoryBeat,
    StoryDraft,
    StoryClaim,
    VerificationResult,
    EvaluationResult,
    PipelineConfig,
    ExperimentManifest,
    PipelineArtifacts,
)


class TestBoundingBox:
    def test_creation(self):
        bbox = BoundingBox(10, 20, 100, 200)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 200
    
    def test_to_list(self):
        bbox = BoundingBox(10, 20, 100, 200)
        assert bbox.to_list() == [10, 20, 100, 200]
    
    def test_from_list(self):
        bbox = BoundingBox.from_list([10, 20, 100, 200])
        assert bbox.x1 == 10
        assert bbox.x2 == 100


class TestEvidenceRecord:
    def test_creation_minimal(self):
        evidence = EvidenceRecord(entity="test", frame_id=0)
        assert evidence.entity == "test"
        assert evidence.frame_id == 0
        assert evidence.id.startswith("obs_")
    
    def test_confidence_classification(self):
        high = EvidenceRecord(entity="test", confidence=0.9, frame_id=0)
        assert high.confidence_class == EvidenceConfidence.HIGH
        
        medium = EvidenceRecord(entity="test", confidence=0.6, frame_id=0)
        assert medium.confidence_class == EvidenceConfidence.MEDIUM
        
        low = EvidenceRecord(entity="test", confidence=0.3, frame_id=0)
        assert low.confidence_class == EvidenceConfidence.LOW
    
    def test_to_dict_roundtrip(self):
        original = EvidenceRecord(
            entity="red backpack",
            type=EvidenceType.OBJECT,
            frame_id=3,
            confidence=0.94,
            source=SourceModel.GROUNDING_DINO,
            bbox=BoundingBox(120, 88, 300, 420),
        )
        d = original.to_dict()
        restored = EvidenceRecord.from_dict(d)
        
        assert restored.entity == original.entity
        assert restored.type == original.type
        assert restored.frame_id == original.frame_id
        assert restored.confidence == original.confidence
        assert restored.source == original.source
        assert restored.bbox.to_list() == original.bbox.to_list()


class TestVisualObservations:
    def test_creation(self):
        obs = VisualObservations(
            image_id="test_001",
            frame_id=0,
            detailed_caption="A test image",
            objects=["car", "tree"],
            characters=["person"],
        )
        assert obs.image_id == "test_001"
        assert obs.frame_id == 0
        assert "car" in obs.objects
        assert "person" in obs.characters


class TestWorldEntity:
    def test_creation(self):
        entity = WorldEntity(
            id="char_1",
            label="girl",
            entity_type="character",
            first_frame=0,
            last_frame=2,
        )
        assert entity.label == "girl"
        assert entity.entity_type == "character"
        assert entity.is_recurring is False
    
    def test_to_dict(self):
        entity = WorldEntity(
            id="char_1",
            label="girl",
            entity_type="character",
            first_frame=0,
            last_frame=2,
            frames_present=[0, 1, 2],
        )
        d = entity.to_dict()
        assert d["label"] == "girl"
        assert d["frames_present"] == [0, 1, 2]


class TestWorldState:
    def test_get_all_entities(self):
        state = WorldState()
        state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0)
        ]
        state.objects = [
            WorldEntity(id="o1", label="car", entity_type="object", first_frame=0, last_frame=0)
        ]
        state.locations = [
            WorldEntity(id="l1", label="street", entity_type="location", first_frame=0, last_frame=0)
        ]
        
        all_entities = state.get_all_entities()
        assert len(all_entities) == 3
    
    def test_get_entity_by_label(self):
        state = WorldState()
        entity = WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0)
        state.characters = [entity]
        
        found = state.get_entity_by_label("girl")
        assert found is not None
        assert found.id == "c1"
        
        not_found = state.get_entity_by_label("boy")
        assert not_found is None


class TestCreativePlan:
    def test_creation(self):
        plan = CreativePlan(genre="mystery", tone="suspenseful")
        assert plan.genre == "mystery"
        assert plan.tone == "suspenseful"
    
    def test_to_dict(self):
        plan = CreativePlan(
            genre="comedy",
            tone="comedic",
            characters=[{"label": "hero", "goal": "save the day"}],
        )
        d = plan.to_dict()
        assert d["genre"] == "comedy"
        assert len(d["characters"]) == 1


class TestStoryBeat:
    def test_creation(self):
        beat = StoryBeat(
            beat_number=1,
            beat_type="setup",
            description="Introduction",
            target_words=50,
        )
        assert beat.beat_number == 1
        assert beat.beat_type == "setup"
        assert beat.target_words == 50


class TestStoryPlan:
    def test_creation(self):
        beats = [
            StoryBeat(beat_number=1, beat_type="setup", description="Start", target_words=50),
            StoryBeat(beat_number=2, beat_type="conflict", description="Problem", target_words=50),
        ]
        plan = StoryPlan(beats=beats, target_total_words=100)
        assert len(plan.beats) == 2
        assert plan.target_total_words == 100


class TestPipelineConfig:
    def test_from_mode_fast(self):
        config = PipelineConfig.from_mode("fast")
        assert config.mode == "fast"
        assert config.use_grounding_dino is False
        assert config.use_faiss is False
    
    def test_from_mode_standard(self):
        config = PipelineConfig.from_mode("standard")
        assert config.mode == "standard"
        assert config.use_grounding_dino is True
        assert config.use_faiss is True
    
    def test_from_mode_full(self):
        config = PipelineConfig.from_mode("full")
        assert config.mode == "full"
        assert config.use_ocr is True
        assert config.target_story_words == 300
    
    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            PipelineConfig.from_mode("invalid")


class TestExperimentManifest:
    def test_creation(self):
        manifest = ExperimentManifest(
            git_commit="abc123",
            vision_model="florence2",
            language_model="qwen2.5-0.5b",
        )
        assert manifest.git_commit == "abc123"
        assert manifest.run_id.startswith("run_")


class TestPipelineArtifacts:
    def test_to_dict(self):
        artifacts = PipelineArtifacts(
            run_id="test_run",
            manifest=ExperimentManifest(),
        )
        d = artifacts.to_dict()
        assert d["run_id"] == "test_run"
        assert "manifest" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])