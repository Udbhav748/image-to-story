"""Scene grouping for hierarchical memory organization."""
from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import uuid

from ..domain.schemas import VisualObservations, EvidenceRecord, WorldState
from ..memory.embeddings import EmbeddingModelInterface
from ..memory.hierarchical import SceneSummary


@dataclass
class SceneGroupConfig:
    """Configuration for scene grouping."""
    similarity_threshold: float = 0.65
    min_scene_size: int = 1
    max_scene_size: int = 20
    use_sequence_fallback: bool = True
    sequence_window: int = 5
    entity_overlap_weight: float = 0.4
    semantic_similarity_weight: float = 0.6


class SceneGrouper:
    """Group images into scenes using semantic similarity and entity overlap."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModelInterface,
        config: SceneGroupConfig | None = None,
    ):
        self._embedding_model = embedding_model
        self._config = config or SceneGroupConfig()
    
    def group_observations(
        self,
        observations: list[VisualObservations],
        world_state: WorldState | None = None,
    ) -> list[SceneSummary]:
        """
        Group observations into scenes.
        
        Uses semantic similarity of observations and entity overlap.
        Falls back to sequential grouping if semantic clustering is unreliable.
        """
        if not observations:
            return []
        
        if len(observations) == 1:
            return [self._create_single_scene(observations[0])]
        
        # Try semantic clustering first
        scenes = self._semantic_clustering(observations, world_state)
        
        # Validate and potentially fall back
        if self._config.use_sequence_fallback and not self._validate_scenes(scenes, observations):
            scenes = self._sequential_grouping(observations, world_state)
        
        # Ensure each scene has valid data
        scenes = [s for s in scenes if s.image_ids]
        
        return scenes
    
    def _create_single_scene(self, obs: VisualObservations) -> SceneSummary:
        """Create a scene summary for a single observation."""
        scene_id = f"scene_{obs.frame_id}_{uuid.uuid4().hex[:8]}"
        
        # Generate embedding for the scene
        scene_text = self._observation_to_text(obs)
        embedding = self._embedding_model.encode_single(scene_text) if scene_text else None
        
        entities = self._extract_entities(obs)
        
        return SceneSummary(
            scene_id=scene_id,
            image_ids=[obs.image_id],
            frame_indices=[obs.frame_id],
            dominant_entities=entities["characters"],
            important_objects=entities["objects"],
            actions=entities["actions"],
            location=obs.scene or "",
            environment=obs.style_or_mood or "",
            recurring_entities=[],
            key_evidence_ids=[e.id for e in obs.evidence_records],
            summary_text=self._generate_scene_summary(obs, entities),
            embedding=embedding,
            evidence_count=len(obs.evidence_records),
            start_frame=obs.frame_id,
            end_frame=obs.frame_id,
        )
    
    def _semantic_clustering(
        self,
        observations: list[VisualObservations],
        world_state: WorldState | None,
    ) -> list[SceneSummary]:
        """Group observations by semantic similarity."""
        # Create embeddings for each observation
        texts = [self._observation_to_text(obs) for obs in observations]
        embeddings = self._embedding_model.encode(texts)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # Build similarity matrix
        n = len(observations)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Also compute entity overlap matrix
        entity_matrix = self._compute_entity_overlap_matrix(observations, world_state)
        
        # Combined similarity
        combined = (
            self._config.semantic_similarity_weight * similarity_matrix +
            self._config.entity_overlap_weight * entity_matrix
        )
        
        # Simple greedy clustering
        scenes = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Start new scene
            scene_indices = [i]
            assigned.add(i)
            
            # Find similar observations
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                if combined[i, j] >= self._config.similarity_threshold:
                    scene_indices.append(j)
                    assigned.add(j)
                
                if len(scene_indices) >= self._config.max_scene_size:
                    break
            
            # Create scene from indices
            scene_obs = [observations[idx] for idx in scene_indices]
            scene = self._merge_observations_to_scene(scene_obs, world_state)
            scenes.append(scene)
        
        return scenes
    
    def _compute_entity_overlap_matrix(
        self,
        observations: list[VisualObservations],
        world_state: WorldState | None,
    ) -> np.ndarray:
        """Compute entity overlap similarity between observations."""
        n = len(observations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            entities_i = self._get_observation_entities(observations[i])
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                
                entities_j = self._get_observation_entities(observations[j])
                
                if not entities_i or not entities_j:
                    matrix[i, j] = 0.0
                    continue
                
                intersection = len(entities_i & entities_j)
                union = len(entities_i | entities_j)
                
                if union > 0:
                    matrix[i, j] = intersection / union
        
        return matrix
    
    def _get_observation_entities(self, obs: VisualObservations) -> set[str]:
        """Extract all entity labels from an observation."""
        entities = set()
        entities.update(obs.characters)
        entities.update(obs.od_labels)
        for e in obs.evidence_records:
            entities.add(e.entity)
        return entities
    
    def _sequential_grouping(
        self,
        observations: list[VisualObservations],
        world_state: WorldState | None,
    ) -> list[SceneSummary]:
        """Fallback: group by sequence with sliding window."""
        scenes = []
        window = self._config.sequence_window
        
        for i in range(0, len(observations), window):
            window_obs = observations[i:i + window]
            scene = self._merge_observations_to_scene(window_obs, world_state)
            scenes.append(scene)
        
        return scenes
    
    def _validate_scenes(
        self,
        scenes: list[SceneSummary],
        observations: list[VisualObservations],
    ) -> bool:
        """Validate that scenes are reasonable."""
        if not scenes:
            return False
        
        # Check that all observations are covered
        covered_frames = set()
        for scene in scenes:
            covered_frames.update(scene.frame_indices)
        
        all_frames = {obs.frame_id for obs in observations}
        if covered_frames != all_frames:
            return False
        
        # Check scene sizes
        for scene in scenes:
            if len(scene.image_ids) < self._config.min_scene_size:
                return False
            if len(scene.image_ids) > self._config.max_scene_size:
                return False
        
        return True
    
    def _merge_observations_to_scene(
        self,
        observations: list[VisualObservations],
        world_state: WorldState | None = None,
    ) -> SceneSummary:
        """Merge multiple observations into a scene summary."""
        if not observations:
            return SceneSummary(scene_id="empty")
        
        # Sort by frame_id
        observations = sorted(observations, key=lambda o: o.frame_id)
        
        scene_id = f"scene_{observations[0].frame_id}_{observations[-1].frame_id}_{uuid.uuid4().hex[:8]}"
        
        # Aggregate entities
        all_entities = defaultdict(int)
        all_objects = defaultdict(int)
        all_actions = defaultdict(int)
        all_evidence_ids = []
        all_frame_indices = []
        all_image_ids = []
        
        locations = []
        environments = []
        
        for obs in observations:
            entities = self._extract_entities(obs)
            
            for e in entities["characters"]:
                all_entities[e] += 1
            for e in entities["objects"]:
                all_objects[e] += 1
            for e in entities["actions"]:
                all_actions[e] += 1
            
            all_evidence_ids.extend([e.id for e in obs.evidence_records])
            all_frame_indices.append(obs.frame_id)
            all_image_ids.append(obs.image_id)
            
            if obs.scene:
                locations.append(obs.scene)
            if obs.style_or_mood:
                environments.append(obs.style_or_mood)
        
        # Determine dominant entities (appear in multiple frames)
        num_obs = len(observations)
        dominant_entities = [e for e, count in all_entities.items() if count >= max(1, num_obs // 2)]
        important_objects = [e for e, count in all_objects.items() if count >= 1]
        actions = list(all_actions.keys())
        
        # Most common location/environment
        location = max(set(locations), key=locations.count) if locations else ""
        environment = max(set(environments), key=environments.count) if environments else ""
        
        # Generate scene text and embedding
        scene_text = self._merge_observations_to_text(observations)
        embedding = self._embedding_model.encode_single(scene_text) if scene_text else None
        
        # Find recurring entities from world state if available, or by frequency
        recurring = []
        if world_state:
            for entity in world_state.get_all_entities():
                if entity.is_recurring:
                    recurring.append(entity.label)
        else:
            recurring = [e for e, count in all_entities.items() if count >= 2]
            recurring.extend([e for e, count in all_objects.items() if count >= 2])
        
        return SceneSummary(
            scene_id=scene_id,
            image_ids=all_image_ids,
            frame_indices=all_frame_indices,
            dominant_entities=dominant_entities,
            important_objects=important_objects,
            actions=actions,
            location=location,
            environment=environment,
            recurring_entities=recurring,
            key_evidence_ids=all_evidence_ids,
            summary_text=self._generate_merged_summary(observations, dominant_entities, important_objects, actions),
            embedding=embedding,
            evidence_count=len(all_evidence_ids),
            start_frame=min(all_frame_indices),
            end_frame=max(all_frame_indices),
        )
    
    def _extract_entities(self, obs: VisualObservations) -> dict[str, list[str]]:
        """Extract entity categories from observation."""
        return {
            "characters": list(obs.characters),
            "objects": list(obs.od_labels),
            "actions": list(obs.actions),
        }
    
    def _observation_to_text(self, obs: VisualObservations) -> str:
        """Convert observation to text for embedding."""
        parts = []
        if obs.scene:
            parts.append(f"Scene: {obs.scene}")
        if obs.detailed_caption:
            parts.append(obs.detailed_caption)
        if obs.characters:
            parts.append(f"Characters: {', '.join(obs.characters)}")
        if obs.od_labels:
            parts.append(f"Objects: {', '.join(obs.od_labels)}")
        if obs.actions:
            parts.append(f"Actions: {', '.join(obs.actions)}")
        if obs.style_or_mood:
            parts.append(f"Mood: {obs.style_or_mood}")
        # Include frame_id to differentiate sequential frames with identical content
        parts.append(f"Frame: {obs.frame_id}")
        return " | ".join(parts)
    
    def _merge_observations_to_text(self, observations: list[VisualObservations]) -> str:
        """Convert multiple observations to combined text."""
        parts = []
        for obs in observations:
            parts.append(self._observation_to_text(obs))
        return " | ".join(parts)
    
    def _generate_scene_summary(
        self,
        obs: VisualObservations,
        entities: dict[str, list[str]],
    ) -> str:
        """Generate human-readable scene summary."""
        parts = []
        
        if obs.scene:
            parts.append(f"Scene: {obs.scene}")
        
        if entities["characters"]:
            parts.append(f"Characters: {', '.join(entities['characters'])}")
        
        if entities["objects"]:
            parts.append(f"Objects: {', '.join(entities['objects'][:5])}")
        
        if entities["actions"]:
            parts.append(f"Actions: {', '.join(entities['actions'][:3])}")
        
        return ". ".join(parts) + "."
    
    def _generate_merged_summary(
        self,
        observations: list[VisualObservations],
        dominant_entities: list[str],
        important_objects: list[str],
        actions: list[str],
    ) -> str:
        """Generate summary for merged scene."""
        parts = []
        
        # Location from first observation with scene
        for obs in observations:
            if obs.scene:
                parts.append(f"Location: {obs.scene}")
                break
        
        if dominant_entities:
            parts.append(f"Recurring: {', '.join(dominant_entities)}")
        
        if important_objects:
            parts.append(f"Objects: {', '.join(important_objects[:8])}")
        
        if actions:
            parts.append(f"Actions: {', '.join(actions[:5])}")
        
        # Frame range
        frames = [str(o.frame_id) for o in observations]
        parts.append(f"Frames: {frames[0]}-{frames[-1]} ({len(frames)} images)")
        
        return ". ".join(parts) + "."


def create_scene_grouper(
    embedding_model: EmbeddingModelInterface,
    similarity_threshold: float = 0.65,
    min_scene_size: int = 1,
    max_scene_size: int = 20,
) -> SceneGrouper:
    """Factory function to create scene grouper."""
    config = SceneGroupConfig(
        similarity_threshold=similarity_threshold,
        min_scene_size=min_scene_size,
        max_scene_size=max_scene_size,
    )
    return SceneGrouper(embedding_model, config)