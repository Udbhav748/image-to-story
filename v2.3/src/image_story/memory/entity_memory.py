"""Extended entity tracking for cross-scene persistence."""
from __future__ import annotations
from typing import Any
from collections import defaultdict
import numpy as np

from ..domain.schemas import VisualObservations, EvidenceRecord, WorldState, WorldEntity, BoundingBox
from ..domain.enums import EvidenceType
from ..memory.embeddings import EmbeddingModelInterface
from ..memory.hierarchical import EntityMemory, SceneSummary
from ..context.world_state import EntityTracker


class EntityMemoryTracker:
    """Track entities across scenes with stable identities."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModelInterface | None = None,
        similarity_threshold: float = 0.8,
        alias_threshold: float = 0.7,
    ):
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold
        self._alias_threshold = alias_threshold
        self._base_tracker = EntityTracker(similarity_threshold)
        self._entity_counter = 0
        self._label_to_entity_id: dict[str, str] = {}
        self._entity_embeddings: dict[str, np.ndarray] = {}
    
    def build_entity_memory(
        self,
        scenes: list[SceneSummary],
        observations: list[VisualObservations],
        world_state: WorldState,
    ) -> dict[str, EntityMemory]:
        """
        Build persistent entity memories from scenes and world state.
        
        Creates EntityMemory objects with stable IDs that persist across scenes.
        """
        entity_memories: dict[str, EntityMemory] = {}
        
        # First, process world state entities for base identities
        for entity in world_state.get_all_entities():
            entity_memory = self._create_entity_memory_from_world_entity(entity)
            entity_memories[entity_memory.entity_id] = entity_memory
            self._label_to_entity_id[entity_memory.normalized_label] = entity_memory.entity_id
            for alias in entity_memory.aliases:
                self._label_to_entity_id[alias] = entity_memory.entity_id
        
        # Then enhance with scene-level information
        self._enhance_with_scenes(entity_memories, scenes, observations)
        
        # Enhance with world state recurring entity frames
        self._enhance_with_world_state(entity_memories, scenes, world_state)
        
        # Merge similar entities across scenes
        entity_memories = self._merge_similar_entities(entity_memories)
        
        return entity_memories
    
    def _create_entity_memory_from_world_entity(self, entity: WorldEntity) -> EntityMemory:
        """Create EntityMemory from WorldEntity."""
        normalized = self._normalize_label(entity.label)
        
        # Generate stable entity ID
        self._entity_counter += 1
        entity_id = f"{entity.entity_type}_{normalized}_{self._entity_counter}"
        
        # Compute entity embedding if possible
        embedding = None
        if self._embedding_model and entity.attributes.get("description"):
            try:
                embedding = self._embedding_model.encode_single(entity.attributes["description"])
            except Exception:
                pass
        
        return EntityMemory(
            entity_id=entity_id,
            normalized_label=normalized,
            aliases=[entity.label] + [normalized],
            entity_type=entity.entity_type,
            first_seen_scene="",
            last_seen_scene="",
            scene_ids=[],
            image_ids=[],
            evidence_ids=[],
            confidence=1.0,
            state="present",
            attributes=dict(entity.attributes),
            bounding_boxes={},
        )
    
    def _enhance_with_scenes(
        self,
        entity_memories: dict[str, EntityMemory],
        scenes: list[SceneSummary],
        observations: list[VisualObservations],
    ) -> None:
        """Enhance entity memories with scene and observation data."""
        # Map frame_id to scene_id
        frame_to_scene = {}
        for scene in scenes:
            for frame_idx in scene.frame_indices:
                frame_to_scene[frame_idx] = scene.scene_id
        
        # Map image_id to scene_id
        image_to_scene = {}
        for scene in scenes:
            for img_id in scene.image_ids:
                image_to_scene[img_id] = scene.scene_id
        
        # Process each observation for entity evidence
        for obs in observations:
            scene_id = image_to_scene.get(obs.image_id) or frame_to_scene.get(obs.frame_id)
            if not scene_id:
                continue
            
            # Group evidence by entity
            entity_evidence = defaultdict(list)
            for evidence in obs.evidence_records:
                entity_evidence[evidence.entity].append(evidence)
            
            for entity_label, evidence_list in entity_evidence.items():
                normalized = self._normalize_label(entity_label)
                
                # Find or create entity memory
                entity_id = self._label_to_entity_id.get(normalized)
                if not entity_id:
                    # Create new entity memory
                    entity_type = self._determine_entity_type(evidence_list)
                    self._entity_counter += 1
                    entity_id = f"{entity_type}_{normalized}_{self._entity_counter}"
                    
                    entity_memories[entity_id] = EntityMemory(
                        entity_id=entity_id,
                        normalized_label=normalized,
                        aliases=[entity_label, normalized],
                        entity_type=entity_type,
                        first_seen_scene=scene_id,
                        last_seen_scene=scene_id,
                        scene_ids=[scene_id],
                        image_ids=[obs.image_id],
                        evidence_ids=[e.id for e in evidence_list],
                        confidence=max(e.confidence for e in evidence_list),
                        state="present",
                    )
                    self._label_to_entity_id[normalized] = entity_id
                else:
                    # Update existing
                    entity_mem = entity_memories[entity_id]
                    if scene_id not in entity_mem.scene_ids:
                        entity_mem.scene_ids.append(scene_id)
                        entity_mem.last_seen_scene = scene_id
                    if obs.image_id not in entity_mem.image_ids:
                        entity_mem.image_ids.append(obs.image_id)
                    entity_mem.evidence_ids.extend([e.id for e in evidence_list])
                    entity_mem.confidence = max(entity_mem.confidence, max(e.confidence for e in evidence_list))
                    
                    # Update bounding boxes
                    for evidence in evidence_list:
                        if evidence.bbox:
                            entity_mem.bounding_boxes[scene_id] = evidence.bbox.to_list()

    def _enhance_with_world_state(
        self,
        entity_memories: dict[str, EntityMemory],
        scenes: list[SceneSummary],
        world_state: WorldState,
    ) -> None:
        """Enhance entity memories with world state recurring entity frames."""
        # Build frame to scene mapping
        frame_to_scene = {}
        for scene in scenes:
            for frame_idx in scene.frame_indices:
                frame_to_scene[frame_idx] = scene.scene_id
        
        # Process world state entities for recurring frames
        for entity in world_state.get_all_entities():
            if not entity.is_recurring:
                continue
            
            normalized = self._normalize_label(entity.label)
            entity_id = self._label_to_entity_id.get(normalized)
            if not entity_id or entity_id not in entity_memories:
                continue
            
            entity_mem = entity_memories[entity_id]
            
            # Add scenes for all frames where entity is present
            for frame_idx in entity.frames_present:
                scene_id = frame_to_scene.get(frame_idx)
                if scene_id and scene_id not in entity_mem.scene_ids:
                    entity_mem.scene_ids.append(scene_id)
                    # Update first/last seen
                    if not entity_mem.first_seen_scene or scene_id < entity_mem.first_seen_scene:
                        entity_mem.first_seen_scene = scene_id
                    if not entity_mem.last_seen_scene or scene_id > entity_mem.last_seen_scene:
                        entity_mem.last_seen_scene = scene_id
            
            # Sort scene_ids
            entity_mem.scene_ids.sort()
    
    def _determine_entity_type(self, evidence_list: list[EvidenceRecord]) -> str:
        """Determine entity type from evidence."""
        types = [e.type for e in evidence_list]
        if EvidenceType.PERSON in types:
            return "character"
        elif EvidenceType.OBJECT in types or EvidenceType.ENTITY in types:
            return "object"
        elif EvidenceType.LOCATION in types:
            return "location"
        return "object"
    
    def _merge_similar_entities(
        self,
        entity_memories: dict[str, EntityMemory],
    ) -> dict[str, EntityMemory]:
        """Merge entities that are likely the same across scenes."""
        # For now, use label-based merging
        # Could be enhanced with embedding similarity if _embedding_model available
        
        merged = {}
        processed = set()
        
        for entity_id, entity_mem in entity_memories.items():
            if entity_id in processed:
                continue
            
            # Find similar entities
            similar = [entity_mem]
            for other_id, other_mem in entity_memories.items():
                if other_id == entity_id or other_id in processed:
                    continue
                
                if self._entities_match(entity_mem, other_mem):
                    similar.append(other_mem)
                    processed.add(other_id)
            
            # Merge into single entity
            if len(similar) > 1:
                merged_entity = self._merge_entities(similar)
                merged[merged_entity.entity_id] = merged_entity
                for e in similar:
                    processed.add(e.entity_id)
                    self._label_to_entity_id[merged_entity.normalized_label] = merged_entity.entity_id
            else:
                merged[entity_id] = entity_mem
                processed.add(entity_id)
        
        return merged
    
    def _entities_match(self, e1: EntityMemory, e2: EntityMemory) -> bool:
        """Check if two entities are the same."""
        # Same type required
        if e1.entity_type != e2.entity_type:
            return False
        
        # Check label similarity
        if e1.normalized_label == e2.normalized_label:
            return True
        
        # Check aliases
        for alias1 in e1.aliases:
            for alias2 in e2.aliases:
                if self._normalize_label(alias1) == self._normalize_label(alias2):
                    return True
        
        # Do NOT merge based on scene overlap alone - different entities can appear in same scene
        # Only merge if labels/aliases match
        
        return False
    
    def _merge_entities(self, entities: list[EntityMemory]) -> EntityMemory:
        """Merge multiple entity memories into one."""
        # Use the first entity as base
        base = entities[0]
        
        # Merge all data
        all_aliases = set(base.aliases)
        all_scene_ids = set(base.scene_ids)
        all_image_ids = set(base.image_ids)
        all_evidence_ids = set(base.evidence_ids)
        all_attributes = dict(base.attributes)
        all_bboxes = dict(base.bounding_boxes)
        
        min_confidence = base.confidence
        first_scene = base.first_seen_scene
        last_scene = base.last_seen_scene
        
        for entity in entities[1:]:
            all_aliases.update(entity.aliases)
            all_scene_ids.update(entity.scene_ids)
            all_image_ids.update(entity.image_ids)
            all_evidence_ids.update(entity.evidence_ids)
            all_attributes.update(entity.attributes)
            all_bboxes.update(entity.bounding_boxes)
            
            min_confidence = min(min_confidence, entity.confidence)
            
            # Update scene range
            if entity.first_seen_scene and (not first_scene or entity.first_seen_scene < first_scene):
                first_scene = entity.first_seen_scene
            if entity.last_seen_scene and (not last_scene or entity.last_seen_scene > last_scene):
                last_scene = entity.last_seen_scene
        
        return EntityMemory(
            entity_id=base.entity_id,
            normalized_label=base.normalized_label,
            aliases=list(all_aliases),
            entity_type=base.entity_type,
            first_seen_scene=first_scene,
            last_seen_scene=last_scene,
            scene_ids=sorted(all_scene_ids),
            image_ids=sorted(all_image_ids),
            evidence_ids=sorted(all_evidence_ids),
            confidence=min_confidence,
            state="present",
            attributes=all_attributes,
            bounding_boxes=all_bboxes,
        )
    
    def _normalize_label(self, label: str) -> str:
        """Normalize entity label for matching."""
        return label.lower().strip().replace(" ", "_")
    
    def update_entity_state(
        self,
        entity_memories: dict[str, EntityMemory],
        scenes: list[SceneSummary],
    ) -> dict[str, EntityMemory]:
        """Update entity states based on scene presence."""
        # Build scene index for ordering
        scene_order = {s.scene_id: i for i, s in enumerate(scenes)}
        
        for entity_mem in entity_memories.values():
            if not entity_mem.scene_ids:
                continue
            
            # Check if entity disappeared
            last_scene_idx = max(scene_order.get(s, -1) for s in entity_mem.scene_ids)
            if last_scene_idx < len(scenes) - 1:
                # Check if entity appears in any later scene
                appears_later = any(
                    scene_order.get(s, -1) > last_scene_idx
                    for s in entity_mem.scene_ids
                )
                if not appears_later:
                    entity_mem.state = "disappeared"
        
        return entity_memories


def create_entity_memory_tracker(
    embedding_model: EmbeddingModelInterface | None = None,
    similarity_threshold: float = 0.8,
) -> EntityMemoryTracker:
    """Factory function to create entity memory tracker."""
    return EntityMemoryTracker(embedding_model, similarity_threshold)