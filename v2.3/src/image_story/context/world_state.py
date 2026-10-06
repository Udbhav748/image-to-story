"""World state and entity tracking across frames."""
from typing import Any
from collections import defaultdict

from ..domain.schemas import (
    WorldState,
    WorldEntity,
    VisualObservations,
    EvidenceRecord,
    EvidenceType,
    BoundingBox,
)
from ..domain.exceptions import EntityNotFoundError


class EntityTracker:
    """Track entities across multiple frames."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self._similarity_threshold = similarity_threshold
        self._entity_counter = 0
    
    def _generate_entity_id(self, label: str, entity_type: str) -> str:
        self._entity_counter += 1
        return f"{entity_type}_{label}_{self._entity_counter}".replace(" ", "_")
    
    def _normalize_label(self, label: str) -> str:
        return label.lower().strip()
    
    def _labels_match(self, label1: str, label2: str) -> bool:
        n1 = self._normalize_label(label1)
        n2 = self._normalize_label(label2)
        return n1 == n2 or n1 in n2 or n2 in n1
    
    def update_from_observations(self, world_state: WorldState, observations: VisualObservations) -> WorldState:
        """Update world state with new observations."""
        frame_id = observations.frame_id
        world_state.frame_count = max(world_state.frame_count, frame_id + 1)
        
        entity_evidence = defaultdict(list)
        for evidence in observations.evidence_records:
            entity_evidence[evidence.entity].append(evidence)
        
        for entity_label, evidence_list in entity_evidence.items():
            self._process_entity_evidence(world_state, entity_label, evidence_list, frame_id)
        
        self._update_recurring_status(world_state)
        self._detect_disappearances(world_state, frame_id)
        self._detect_open_loops(world_state)
        
        return world_state
    
    def _process_entity_evidence(
        self,
        world_state: WorldState,
        entity_label: str,
        evidence_list: list[EvidenceRecord],
        frame_id: int,
    ) -> None:
        entity_type = self._determine_entity_type(evidence_list)
        existing_entity = self._find_matching_entity(world_state, entity_label, entity_type)
        
        if existing_entity:
            self._update_existing_entity(existing_entity, evidence_list, frame_id)
        else:
            self._create_new_entity(world_state, entity_label, entity_type, evidence_list, frame_id)
    
    def _determine_entity_type(self, evidence_list: list[EvidenceRecord]) -> str:
        types = [e.type for e in evidence_list]
        if EvidenceType.PERSON in types:
            return "character"
        elif EvidenceType.OBJECT in types or EvidenceType.ENTITY in types:
            return "object"
        elif EvidenceType.LOCATION in types:
            return "location"
        return "object"
    
    def _find_matching_entity(
        self,
        world_state: WorldState,
        label: str,
        entity_type: str,
    ) -> WorldEntity | None:
        entity_list = getattr(world_state, f"{entity_type}s", [])
        for entity in entity_list:
            if self._labels_match(entity.label, label):
                return entity
        return None
    
    def _update_existing_entity(
        self,
        entity: WorldEntity,
        evidence_list: list[EvidenceRecord],
        frame_id: int,
    ) -> None:
        if frame_id not in entity.frames_present:
            entity.frames_present.append(frame_id)
            entity.frames_present.sort()
        
        entity.last_frame = max(entity.last_frame, frame_id)
        
        for evidence in evidence_list:
            if evidence.bbox:
                entity.bounding_boxes[frame_id] = evidence.bbox
            
            if evidence.action:
                entity.attributes.setdefault("actions", []).append(evidence.action)
            if evidence.relationship:
                entity.attributes.setdefault("relationships", []).append(evidence.relationship)
    
    def _create_new_entity(
        self,
        world_state: WorldState,
        label: str,
        entity_type: str,
        evidence_list: list[EvidenceRecord],
        frame_id: int,
    ) -> None:
        entity_id = self._generate_entity_id(label, entity_type)
        
        entity = WorldEntity(
            id=entity_id,
            label=label,
            entity_type=entity_type,
            first_frame=frame_id,
            last_frame=frame_id,
            frames_present=[frame_id],
        )
        
        for evidence in evidence_list:
            if evidence.bbox:
                entity.bounding_boxes[frame_id] = evidence.bbox
            if evidence.action:
                entity.attributes.setdefault("actions", []).append(evidence.action)
            if evidence.relationship:
                entity.attributes.setdefault("relationships", []).append(evidence.relationship)
        
        entity_list = getattr(world_state, f"{entity_type}s")
        entity_list.append(entity)
    
    def _update_recurring_status(self, world_state: WorldState) -> None:
        for entity in world_state.get_all_entities():
            entity.is_recurring = len(entity.frames_present) >= 2
    
    def _detect_disappearances(self, world_state: WorldState, current_frame: int) -> None:
        for entity in world_state.get_all_entities():
            # Check if entity was present in a previous frame but not in current frame
            if (entity.last_frame < current_frame and
                current_frame not in entity.frames_present and
                entity.disappearance_frame is None):
                entity.disappearance_frame = entity.last_frame
                
                world_state.open_loops.append({
                    "type": "disappearance",
                    "entity_id": entity.id,
                    "entity_label": entity.label,
                    "last_frame": entity.last_frame,
                    "description": f"{entity.label} disappeared after frame {entity.last_frame}",
                })
    
    def _detect_open_loops(self, world_state: WorldState) -> None:
        for entity in world_state.get_all_entities():
            if entity.disappearance_frame is not None:
                continue
            
            actions = entity.attributes.get("actions", [])
            if "holding" in actions or "carrying" in actions:
                later_actions = [a for a in actions if a not in ["holding", "carrying"]]
                if later_actions and "holding" not in later_actions and "carrying" not in later_actions:
                    world_state.open_loops.append({
                        "type": "object_transfer",
                        "entity_id": entity.id,
                        "entity_label": entity.label,
                        "description": f"{entity.label} was holding/carrying something but later not",
                    })


class WorldStateBuilder:
    """Build and maintain world state from observations."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self._tracker = EntityTracker(similarity_threshold)
        self._world_state = WorldState()
    
    @property
    def world_state(self) -> WorldState:
        return self._world_state
    
    def reset(self) -> None:
        self._world_state = WorldState()
        self._tracker = EntityTracker(self._tracker._similarity_threshold)
    
    def add_observations(self, observations: VisualObservations) -> WorldState:
        return self._tracker.update_from_observations(self._world_state, observations)
    
    def add_observations_batch(self, observations_list: list[VisualObservations]) -> WorldState:
        for obs in observations_list:
            self.add_observations(obs)
        return self._world_state
    
    def get_entity_timeline(self, entity_label: str) -> list[dict[str, Any]]:
        entity = self._world_state.get_entity_by_label(entity_label)
        if not entity:
            return []
        
        timeline = []
        for frame_id in entity.frames_present:
            bbox = entity.bounding_boxes.get(frame_id)
            timeline.append({
                "frame_id": frame_id,
                "bbox": bbox.to_list() if bbox else None,
                "actions": entity.attributes.get("actions", []),
                "relationships": entity.attributes.get("relationships", []),
            })
        return timeline
    
    def get_recurring_entities(self) -> list[WorldEntity]:
        return [e for e in self._world_state.get_all_entities() if e.is_recurring]
    
    def get_disappeared_entities(self) -> list[WorldEntity]:
        return [e for e in self._world_state.get_all_entities() if e.disappearance_frame is not None]
    
    def get_open_loops(self) -> list[dict[str, Any]]:
        return self._world_state.open_loops
    
    def to_dict(self) -> dict[str, Any]:
        return self._world_state.to_dict()