"""Tests for scene grouper."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from image_story.memory.scene_grouper import SceneGrouper, SceneGroupConfig, create_scene_grouper
from image_story.memory.hierarchical import SceneSummary
from image_story.domain.schemas import VisualObservations, EvidenceRecord, EvidenceType, SourceModel, InformationClass
from image_story.context.world_state import WorldState, WorldEntity


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, embeddings_map: dict[str, np.ndarray] | None = None):
        self._embeddings_map = embeddings_map or {}
        self._call_count = 0
    
    def encode_single(self, text: str) -> np.ndarray:
        self._call_count += 1
        # Check for exact match first
        if text in self._embeddings_map:
            return self._embeddings_map[text]
        # Check for partial matches (for scene grouping tests)
        for key, emb in self._embeddings_map.items():
            if key in text:
                return emb
        # For clustering tests: return same embedding for texts with same key entities
        if "park" in text and "girl" in text:
            if "park_girl" not in self._embeddings_map:
                np.random.seed(42)
                self._embeddings_map["park_girl"] = np.random.randn(384).astype(np.float32)
                self._embeddings_map["park_girl"] /= np.linalg.norm(self._embeddings_map["park_girl"]) + 1e-8
            return self._embeddings_map["park_girl"]
        if "street" in text and "girl" in text:
            if "street_girl" not in self._embeddings_map:
                np.random.seed(43)
                self._embeddings_map["street_girl"] = np.random.randn(384).astype(np.float32)
                self._embeddings_map["street_girl"] /= np.linalg.norm(self._embeddings_map["street_girl"]) + 1e-8
            return self._embeddings_map["street_girl"]
        # Return deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(384).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self.encode_single(t) for t in texts])


def create_test_observation(
    frame_id: int,
    scene: str = "",
    characters: list[str] | None = None,
    objects: list[str] | None = None,
    actions: list[str] | None = None,
    style: str = "",
) -> VisualObservations:
    """Create a test visual observation."""
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
        image_id=f"img_{frame_id}",
        frame_id=frame_id,
        scene=scene,
        detailed_caption=f"Frame {frame_id} in {scene}",
        characters=chars,
        od_labels=objs,
        actions=acts,
        style_or_mood=style,
        evidence_records=evidence,
    )


class TestSceneGrouper:
    def setup_method(self):
        self.embedding_model = MockEmbeddingModel()
        self.config = SceneGroupConfig(
            similarity_threshold=0.5,
            min_scene_size=1,
            max_scene_size=10,
            use_sequence_fallback=True,
        )
        self.grouper = SceneGrouper(self.embedding_model, self.config)
    
    def test_single_observation(self):
        obs = create_test_observation(0, scene="park", characters=["girl"], objects=["balloon"])
        
        scenes = self.grouper.group_observations([obs])
        
        assert len(scenes) == 1
        assert scenes[0].scene_id.startswith("scene_0_")
        assert scenes[0].location == "park"
        assert "girl" in scenes[0].dominant_entities
        assert "balloon" in scenes[0].important_objects
        assert scenes[0].evidence_count == 2
    
    def test_multiple_observations_different_scenes(self):
        obs1 = create_test_observation(0, scene="park", characters=["girl"], objects=["balloon"])
        obs2 = create_test_observation(1, scene="street", characters=["boy"], objects=["bicycle"])
        
        scenes = self.grouper.group_observations([obs1, obs2])
        
        # Should create separate scenes due to different locations
        assert len(scenes) == 2
    
    def test_multiple_observations_same_scene(self):
        # Create embeddings that are very similar
        emb = np.ones(384, dtype=np.float32) / np.sqrt(384)
        embedding_model = MockEmbeddingModel({
            "Scene: park | Characters: girl | Objects: balloon": emb,
            "Scene: park | Characters: girl | Objects: balloon": emb,
        })
        
        grouper = SceneGrouper(embedding_model, self.config)
        
        obs1 = create_test_observation(0, scene="park", characters=["girl"], objects=["balloon"])
        obs2 = create_test_observation(1, scene="park", characters=["girl"], objects=["balloon"])
        
        scenes = grouper.group_observations([obs1, obs2])
        
        # Should group into one scene due to high similarity
        assert len(scenes) == 1
        assert len(scenes[0].frame_indices) == 2
    
    def test_sequential_fallback(self):
        # Use very different embeddings to force fallback
        embedding_model = MockEmbeddingModel()
        config = SceneGroupConfig(
            similarity_threshold=0.9,  # Very high threshold
            use_sequence_fallback=True,
            sequence_window=2,
        )
        grouper = SceneGrouper(embedding_model, config)
        
        obs1 = create_test_observation(0, scene="park", characters=["girl"])
        obs2 = create_test_observation(1, scene="street", characters=["boy"])
        obs3 = create_test_observation(2, scene="room", characters=["cat"])
        
        scenes = grouper.group_observations([obs1, obs2, obs3])
        
        # Should use sequential grouping with window=2
        assert len(scenes) >= 2
        # First scene should have frames 0,1
        # Second scene should have frame 2
    
    def test_no_fallback(self):
        embedding_model = MockEmbeddingModel()
        config = SceneGroupConfig(
            similarity_threshold=0.9,
            use_sequence_fallback=False,
        )
        grouper = SceneGrouper(embedding_model, config)
        
        obs1 = create_test_observation(0, scene="park", characters=["girl"])
        obs2 = create_test_observation(1, scene="street", characters=["boy"])
        
        scenes = grouper.group_observations([obs1, obs2])
        
        # Each should be its own scene since no grouping happened
        assert len(scenes) == 2
    
    def test_max_scene_size(self):
        embedding_model = MockEmbeddingModel()
        emb = np.ones(384, dtype=np.float32) / np.sqrt(384)
        for i in range(10):
            embedding_model._embeddings_map[f"obs_{i}"] = emb
        
        config = SceneGroupConfig(
            similarity_threshold=0.5,
            max_scene_size=3,
        )
        grouper = SceneGrouper(embedding_model, config)
        
        observations = [
            create_test_observation(i, scene="park", characters=["girl"], objects=[f"obj_{i}"])
            for i in range(10)
        ]
        
        scenes = grouper.group_observations(observations)
        
        # Should respect max_scene_size
        for scene in scenes:
            assert len(scene.image_ids) <= 3
    
    def test_entity_overlap_clustering(self):
        # Create observations with overlapping entities but different scenes
        embedding_model = MockEmbeddingModel()
        
        # Park scene embedding
        park_emb = np.array([1.0] + [0.0]*383, dtype=np.float32)
        # Street scene embedding
        street_emb = np.array([0.0, 1.0] + [0.0]*382, dtype=np.float32)
        
        embedding_model._embeddings_map["Scene: park | Characters: girl | Objects: balloon"] = park_emb
        embedding_model._embeddings_map["Scene: street | Characters: girl | Objects: bicycle"] = street_emb
        
        # Also add the generic park_girl and street_girl embeddings for the merged scene text
        embedding_model._embeddings_map["park_girl"] = park_emb
        embedding_model._embeddings_map["street_girl"] = street_emb
        
        config = SceneGroupConfig(
            similarity_threshold=0.15,  # Lower threshold to allow clustering
            entity_overlap_weight=0.8,
            semantic_similarity_weight=0.2,
        )
        grouper = SceneGrouper(embedding_model, config)
        
        obs1 = create_test_observation(0, scene="park", characters=["girl"], objects=["balloon"])
        obs2 = create_test_observation(1, scene="street", characters=["girl"], objects=["bicycle"])
        
        scenes = grouper.group_observations([obs1, obs2])
        
        # Should group together due to entity overlap (girl appears in both)
        assert len(scenes) == 1
    
    def test_world_state_recurring_entities(self):
        embedding_model = MockEmbeddingModel()
        config = SceneGroupConfig(similarity_threshold=0.5)
        grouper = SceneGrouper(embedding_model, config)
        
        obs1 = create_test_observation(0, scene="park", characters=["girl"], objects=["balloon"])
        obs2 = create_test_observation(1, scene="park", characters=["girl"], objects=["balloon", "bicycle"])
        obs3 = create_test_observation(2, scene="park", characters=["boy"], objects=["balloon"])
        
        # Build world state with recurring entities
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=1, frames_present=[0, 1], is_recurring=True),
            WorldEntity(id="c2", label="boy", entity_type="character", first_frame=2, last_frame=2, frames_present=[2]),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="balloon", entity_type="object", first_frame=0, last_frame=2, frames_present=[0, 1, 2], is_recurring=True),
        ]
        
        scenes = grouper.group_observations([obs1, obs2, obs3], world_state)
        
        # Should identify recurring entities
        for scene in scenes:
            if "girl" in scene.dominant_entities:
                assert "girl" in scene.recurring_entities or "balloon" in scene.recurring_entities
    
    def test_generate_scene_summary(self):
        obs = create_test_observation(0, scene="peaceful park", characters=["girl", "boy"], objects=["balloon", "kite"], actions=["running", "playing"])
        
        scenes = self.grouper.group_observations([obs])
        
        summary = scenes[0].summary_text
        assert "peaceful park" in summary
        assert "girl" in summary or "boy" in summary
        assert "balloon" in summary or "kite" in summary
    
    def test_merge_observations_to_scene(self):
        # Create embeddings that are very similar
        emb = np.ones(384, dtype=np.float32) / np.sqrt(384)
        embedding_model = MockEmbeddingModel({
            "Scene: park | Characters: girl | Objects: balloon": emb,
            "Scene: park | Characters: girl | Objects: bicycle": emb,
        })
        
        config = SceneGroupConfig(similarity_threshold=0.5)
        grouper = SceneGrouper(embedding_model, config)
        
        obs1 = create_test_observation(0, scene="park", characters=["girl"], objects=["balloon"])
        obs2 = create_test_observation(1, scene="park", characters=["girl"], objects=["bicycle"])
        
        scenes = grouper.group_observations([obs1, obs2])
        
        assert len(scenes) == 1
        scene = scenes[0]
        assert scene.start_frame == 0
        assert scene.end_frame == 1
        assert "girl" in scene.dominant_entities
        assert "balloon" in scene.important_objects
        assert "bicycle" in scene.important_objects
    
    def test_create_single_scene(self):
        obs = create_test_observation(5, scene="beach", characters=["dog"], objects=["frisbee"], style="sunny")
        
        scenes = self.grouper.group_observations([obs])
        
        assert len(scenes) == 1
        assert scenes[0].start_frame == 5
        assert scenes[0].end_frame == 5
        assert scenes[0].location == "beach"
        assert scenes[0].environment == "sunny"


class TestSceneGroupConfig:
    def test_defaults(self):
        config = SceneGroupConfig()
        assert config.similarity_threshold == 0.65
        assert config.min_scene_size == 1
        assert config.max_scene_size == 20
        assert config.use_sequence_fallback is True
        assert config.sequence_window == 5
    
    def test_custom(self):
        config = SceneGroupConfig(
            similarity_threshold=0.8,
            max_scene_size=5,
            use_sequence_fallback=False,
        )
        assert config.similarity_threshold == 0.8
        assert config.max_scene_size == 5
        assert config.use_sequence_fallback is False


class TestCreateSceneGrouper:
    def test_factory(self):
        embedding_model = MockEmbeddingModel()
        grouper = create_scene_grouper(
            embedding_model,
            similarity_threshold=0.7,
            min_scene_size=2,
            max_scene_size=15,
        )
        
        assert isinstance(grouper, SceneGrouper)
        assert grouper._config.similarity_threshold == 0.7
        assert grouper._config.min_scene_size == 2
        assert grouper._config.max_scene_size == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])