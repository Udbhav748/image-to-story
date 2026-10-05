"""State transition detection for narrative-relevant changes."""
from __future__ import annotations
from typing import Any
from collections import defaultdict
import uuid

from ..domain.schemas import EvidenceRecord, WorldEntity, BoundingBox
from ..memory.hierarchical import StateTransition, EntityMemory, SceneSummary
from ..domain.enums import EvidenceType


class StateTransitionDetector:
    """Detect narrative-relevant state changes across scenes."""
    
    def __init__(
        self,
        position_threshold: float = 0.3,  # IoU threshold for movement
        confidence_threshold: float = 0.5,
    ):
        self._position_threshold = position_threshold
        self._confidence_threshold = confidence_threshold
    
    def detect_transitions(
        self,
        entity_memories: dict[str, EntityMemory],
        scenes: list[SceneSummary],
        observations: list[VisualObservations],
    ) -> list[StateTransition]:
        """
        Detect state transitions across scenes.
        
        Types detected:
        - appeared: entity first appears
        - disappeared: entity disappears
        - moved: entity changes position significantly
        - changed_scene: entity appears in different scene context
        - changed_relationship: entity's relationships change
        - changed_state: entity's attributes change
        """
        transitions = []
        
        # Sort scenes by start_frame
        sorted_scenes = sorted(scenes, key=lambda s: s.start_frame)
        
        # Build scene->observation mapping
        scene_to_obs = defaultdict(list)
        for obs in observations:
            for scene in sorted_scenes:
                # Check frame_indices first, fallback to start_frame/end_frame range
                frame_indices = scene.frame_indices
                if frame_indices:
                    if obs.frame_id in frame_indices:
                        scene_to_obs[scene.scene_id].append(obs)
                        break
                else:
                    # Fallback: check if frame_id is within scene's frame range
                    if scene.start_frame <= obs.frame_id <= scene.end_frame:
                        scene_to_obs[scene.scene_id].append(obs)
                        break
        
        for entity_id, entity_mem in entity_memories.items():
            entity_transitions = self._detect_entity_transitions(
                entity_mem, sorted_scenes, scene_to_obs
            )
            transitions.extend(entity_transitions)
        
        return transitions
    
    def _detect_entity_transitions(
        self,
        entity_mem: EntityMemory,
        scenes: list[SceneSummary],
        scene_to_obs: dict[str, list[VisualObservations]],
    ) -> list[StateTransition]:
        """Detect transitions for a single entity."""
        transitions = []
        scene_ids = sorted(entity_mem.scene_ids, key=lambda s: next(
            (sc.start_frame for sc in scenes if sc.scene_id == s), 0
        ))
        
        if not scene_ids:
            return transitions
        
        # Track first appearance
        first_scene = scene_ids[0]
        transitions.append(StateTransition(
            entity_id=entity_mem.entity_id,
            entity_label=entity_mem.normalized_label,
            transition_type="appeared",
            from_scene="",
            to_scene=first_scene,
            description=f"{entity_mem.normalized_label} first appears in scene",
            confidence=0.9,
            evidence_ids=entity_mem.evidence_ids[:3],
        ))
        
        # Track scene-to-scene changes
        for i in range(1, len(scene_ids)):
            prev_scene = scene_ids[i - 1]
            curr_scene = scene_ids[i]
            
            # Check for position change
            move_transition = self._check_movement(
                entity_mem, prev_scene, curr_scene, scenes
            )
            if move_transition:
                transitions.append(move_transition)
            
            # Check for relationship change
            rel_transition = self._check_relationship_change(
                entity_mem, prev_scene, curr_scene, scene_to_obs
            )
            if rel_transition:
                transitions.append(rel_transition)
            
            # Check for state change
            state_transition = self._check_state_change(
                entity_mem, prev_scene, curr_scene, scene_to_obs
            )
            if state_transition:
                transitions.append(state_transition)
        
        # Track disappearance
        last_scene = scene_ids[-1]
        last_scene_obj = next((s for s in scenes if s.scene_id == last_scene), None)
        if last_scene_obj and last_scene_obj.end_frame < scenes[-1].end_frame:
            transitions.append(StateTransition(
                entity_id=entity_mem.entity_id,
                entity_label=entity_mem.normalized_label,
                transition_type="disappeared",
                from_scene=last_scene,
                to_scene="",
                from_frame=last_scene_obj.end_frame,
                description=f"{entity_mem.normalized_label} disappears after {last_scene}",
                confidence=0.8,
                evidence_ids=entity_mem.evidence_ids[-3:],
            ))
        
        # Track reappearance
        if entity_mem.state == "disappeared" and len(scene_ids) > 1:
            # Check if there's a gap then reappearance
            for i in range(1, len(scene_ids)):
                prev_idx = next((j for j, s in enumerate(scenes) if s.scene_id == scene_ids[i-1]), 0)
                curr_idx = next((j for j, s in enumerate(scenes) if s.scene_id == scene_ids[i]), 0)
                
                if curr_idx - prev_idx > 1:
                    transitions.append(StateTransition(
                        entity_id=entity_mem.entity_id,
                        entity_label=entity_mem.normalized_label,
                        transition_type="reappeared",
                        from_scene=scene_ids[i-1],
                        to_scene=scene_ids[i],
                        description=f"{entity_mem.normalized_label} reappears in {scene_ids[i]} after absence",
                        confidence=0.7,
                        evidence_ids=entity_mem.evidence_ids[-3:],
                    ))
                    break
        
        return transitions
    
    def _check_movement(
        self,
        entity_mem: EntityMemory,
        prev_scene: str,
        curr_scene: str,
        scenes: list[SceneSummary],
    ) -> StateTransition | None:
        """Check if entity moved significantly between scenes."""
        prev_bbox = entity_mem.bounding_boxes.get(prev_scene)
        curr_bbox = entity_mem.bounding_boxes.get(curr_scene)
        
        if not prev_bbox or not curr_bbox:
            return None
        
        iou = self._compute_iou(prev_bbox, curr_bbox)
        
        if iou < (1.0 - self._position_threshold):
            return StateTransition(
                entity_id=entity_mem.entity_id,
                entity_label=entity_mem.normalized_label,
                transition_type="moved",
                from_scene=prev_scene,
                to_scene=curr_scene,
                description=f"{entity_mem.normalized_label} moved significantly (IoU: {iou:.2f})",
                confidence=0.7,
                evidence_ids=[],
            )
        
        return None
    
    def _check_relationship_change(
        self,
        entity_mem: EntityMemory,
        prev_scene: str,
        curr_scene: str,
        scene_to_obs: dict[str, list[VisualObservations]],
    ) -> StateTransition | None:
        """Check if entity's relationships changed."""
        prev_obs = scene_to_obs.get(prev_scene, [])
        curr_obs = scene_to_obs.get(curr_scene, [])
        
        prev_relations = set()
        curr_relations = set()
        
        for obs in prev_obs:
            for evidence in obs.evidence_records:
                if evidence.entity == entity_mem.normalized_label and evidence.relationship:
                    prev_relations.add(evidence.relationship)
        
        for obs in curr_obs:
            for evidence in obs.evidence_records:
                if evidence.entity == entity_mem.normalized_label and evidence.relationship:
                    curr_relations.add(evidence.relationship)
        
        added = curr_relations - prev_relations
        removed = prev_relations - curr_relations
        
        if added or removed:
            desc_parts = []
            if added:
                desc_parts.append(f"gained: {', '.join(added)}")
            if removed:
                desc_parts.append(f"lost: {', '.join(removed)}")
            
            return StateTransition(
                entity_id=entity_mem.entity_id,
                entity_label=entity_mem.normalized_label,
                transition_type="changed_relationship",
                from_scene=prev_scene,
                to_scene=curr_scene,
                description=f"{entity_mem.normalized_label} relationship changed: {'; '.join(desc_parts)}",
                confidence=0.6,
                evidence_ids=[],
            )
        
        return None
    
    def _check_state_change(
        self,
        entity_mem: EntityMemory,
        prev_scene: str,
        curr_scene: str,
        scene_to_obs: dict[str, list[VisualObservations]],
    ) -> StateTransition | None:
        """Check if entity's state/attributes changed."""
        prev_attrs = self._extract_scene_attributes(entity_mem, prev_scene, scene_to_obs)
        curr_attrs = self._extract_scene_attributes(entity_mem, curr_scene, scene_to_obs)
        
        # Compare actions
        prev_actions = set(prev_attrs.get("actions", []))
        curr_actions = set(curr_attrs.get("actions", []))
        
        added_actions = curr_actions - prev_actions
        removed_actions = prev_actions - curr_actions
        
        if added_actions or removed_actions:
            desc_parts = []
            if added_actions:
                desc_parts.append(f"started: {', '.join(added_actions)}")
            if removed_actions:
                desc_parts.append(f"stopped: {', '.join(removed_actions)}")
            
            return StateTransition(
                entity_id=entity_mem.entity_id,
                entity_label=entity_mem.normalized_label,
                transition_type="changed_state",
                from_scene=prev_scene,
                to_scene=curr_scene,
                description=f"{entity_mem.normalized_label} state changed: {'; '.join(desc_parts)}",
                confidence=0.6,
                evidence_ids=[],
            )
        
        return None
    
    def _extract_scene_attributes(
        self,
        entity_mem: EntityMemory,
        scene_id: str,
        scene_to_obs: dict[str, list[VisualObservations]],
    ) -> dict[str, list[str]]:
        """Extract entity attributes from a scene's observations."""
        attrs = {"actions": [], "relationships": []}
        
        for obs in scene_to_obs.get(scene_id, []):
            # Check if this observation contains the entity
            has_entity = any(e.entity == entity_mem.normalized_label for e in obs.evidence_records)
            if not has_entity:
                continue
            
            # Collect all actions and relationships from this observation
            for evidence in obs.evidence_records:
                # For ACTION type evidence, the action is stored in entity field
                if evidence.type.value == "action" and evidence.entity:
                    attrs["actions"].append(evidence.entity)
                elif evidence.action:
                    attrs["actions"].append(evidence.action)
                if evidence.relationship:
                    attrs["relationships"].append(evidence.relationship)
        
        return attrs
    
    def _compute_iou(self, bbox1: list[float], bbox2: list[float]) -> float:
        """Compute Intersection over Union for two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def create_state_transition_detector(
    position_threshold: float = 0.3,
    confidence_threshold: float = 0.5,
) -> StateTransitionDetector:
    """Factory function to create state transition detector."""
    return StateTransitionDetector(position_threshold, confidence_threshold)