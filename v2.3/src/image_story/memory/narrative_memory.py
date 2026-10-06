"""Narrative memory for callback and foreshadowing retrieval."""
from __future__ import annotations
from typing import Any
from collections import defaultdict
import uuid

from ..domain.schemas import EvidenceRecord, WorldEntity, VisualObservations
from ..domain.enums import EvidenceType
from ..memory.hierarchical import (
    NarrativeElement,
    EntityMemory,
    SceneSummary,
    StateTransition,
)
from ..memory.entity_memory import EntityMemoryTracker


class NarrativeMemory:
    """Build and query narrative-relevant memory elements."""
    
    def __init__(
        self,
        importance_weights: dict[str, float] | None = None,
    ):
        self._weights = importance_weights or {
            "recurrence": 0.3,
            "emotional": 0.2,
            "visual_prominence": 0.2,
            "transition": 0.15,
            "uniqueness": 0.15,
        }
    
    def build_narrative_elements(
        self,
        entity_memories: dict[str, EntityMemory],
        scenes: list[SceneSummary],
        transitions: list[StateTransition],
        observations: list[VisualObservations],
        world_state: any = None,
    ) -> list[NarrativeElement]:
        """
        Build narrative elements from entities, scenes, and transitions.
        
        Creates elements for:
        - Recurring objects/characters
        - Important transitions
        - Visual motifs
        - Open loops
        - Potential callbacks
        """
        elements = []
        
        # 1. Recurring entities as narrative elements
        for entity_id, entity_mem in entity_memories.items():
            if len(entity_mem.scene_ids) >= 2:
                element = self._create_recurring_element(entity_mem, scenes)
                elements.append(element)
        
        # 2. Transitions as narrative elements
        for transition in transitions:
            if transition.transition_type in ("disappeared", "reappeared", "changed_relationship", "changed_state"):
                element = self._create_transition_element(transition, scenes)
                elements.append(element)
        
        # 3. Scene-specific important objects
        for scene in scenes:
            scene_elements = self._create_scene_elements(scene, observations)
            elements.extend(scene_elements)
        
        # 4. Open loops from transitions
        for transition in transitions:
            if transition.transition_type == "disappeared":
                element = self._create_open_loop_element(transition, scenes)
                elements.append(element)
        
        # 5. Assign narrative importance scores
        self._score_narrative_importance(elements, scenes, transitions)
        
        # 6. Mark potential callbacks
        self._mark_callback_potential(elements, scenes)
        
        return elements
    
    def _create_recurring_element(
        self,
        entity_mem: EntityMemory,
        scenes: list[SceneSummary],
    ) -> NarrativeElement:
        """Create narrative element for recurring entity."""
        # Find first and last scene objects
        first_scene_obj = next((s for s in scenes if s.scene_id == entity_mem.first_seen_scene), None)
        last_scene_obj = next((s for s in scenes if s.scene_id == entity_mem.last_seen_scene), None)
        
        return NarrativeElement(
            element_id=f"narr_recurring_{entity_mem.entity_id}",
            element_type="recurring_entity",
            label=entity_mem.normalized_label,
            description=f"{entity_mem.entity_type.capitalize()} '{entity_mem.normalized_label}' appears in {len(entity_mem.scene_ids)} scenes",
            first_scene=entity_mem.first_seen_scene,
            last_scene=entity_mem.last_seen_scene,
            scene_ids=entity_mem.scene_ids,
            evidence_ids=entity_mem.evidence_ids,
            recurrence_count=len(entity_mem.scene_ids),
            narrative_importance=0.0,  # Will be scored later
            associated_entities=[entity_mem.normalized_label],
            potential_callback=True,
        )
    
    def _create_transition_element(
        self,
        transition: StateTransition,
        scenes: list[SceneSummary],
    ) -> NarrativeElement:
        """Create narrative element from state transition."""
        element_type_map = {
            "disappeared": "object_disappearance",
            "reappeared": "object_reappearance",
            "changed_relationship": "relationship_change",
            "changed_state": "state_change",
            "moved": "movement",
        }
        
        return NarrativeElement(
            element_id=f"narr_trans_{transition.transition_id}",
            element_type=element_type_map.get(transition.transition_type, "transition"),
            label=f"{transition.entity_label} {transition.transition_type}",
            description=transition.description,
            first_scene=transition.from_scene,
            last_scene=transition.to_scene,
            scene_ids=[transition.from_scene, transition.to_scene] if transition.from_scene and transition.to_scene else [],
            evidence_ids=transition.evidence_ids,
            recurrence_count=1,
            narrative_importance=0.0,
            associated_entities=[transition.entity_label],
            potential_callback=transition.transition_type in ("disappeared", "reappeared"),
        )
    
    def _create_scene_elements(
        self,
        scene: SceneSummary,
        observations: list[VisualObservations],
    ) -> list[NarrativeElement]:
        """Create narrative elements specific to a scene."""
        elements = []
        
        # Dominant entities in scene
        for entity_label in scene.dominant_entities:
            if entity_label not in scene.recurring_entities:
                elements.append(NarrativeElement(
                    element_id=f"narr_scene_{scene.scene_id}_{uuid.uuid4().hex[:6]}",
                    element_type="scene_character",
                    label=entity_label,
                    description=f"Character '{entity_label}' prominent in {scene.scene_id}",
                    first_scene=scene.scene_id,
                    last_scene=scene.scene_id,
                    scene_ids=[scene.scene_id],
                    evidence_ids=[e for e in scene.key_evidence_ids if entity_label in e],
                    recurrence_count=1,
                    narrative_importance=0.0,
                    associated_entities=[entity_label],
                ))
        
        # Important objects in scene
        for obj_label in scene.important_objects[:3]:  # Top 3
            if obj_label not in scene.recurring_entities:
                elements.append(NarrativeElement(
                    element_id=f"narr_scene_{scene.scene_id}_{uuid.uuid4().hex[:6]}",
                    element_type="scene_object",
                    label=obj_label,
                    description=f"Object '{obj_label}' featured in {scene.scene_id}",
                    first_scene=scene.scene_id,
                    last_scene=scene.scene_id,
                    scene_ids=[scene.scene_id],
                    evidence_ids=[e for e in scene.key_evidence_ids if obj_label in e],
                    recurrence_count=1,
                    narrative_importance=0.0,
                    associated_entities=[obj_label],
                ))
        
        return elements
    
    def _create_open_loop_element(
        self,
        transition: StateTransition,
        scenes: list[SceneSummary],
    ) -> NarrativeElement:
        """Create open loop element from disappearance."""
        return NarrativeElement(
            element_id=f"narr_openloop_{transition.transition_id}",
            element_type="open_loop",
            label=f"Where is {transition.entity_label}?",
            description=f"{transition.entity_label} disappeared in {transition.from_scene} - fate unknown",
            first_scene=transition.from_scene,
            last_scene=transition.from_scene,
            scene_ids=[transition.from_scene],
            evidence_ids=transition.evidence_ids,
            recurrence_count=1,
            narrative_importance=0.0,
            associated_entities=[transition.entity_label],
            potential_callback=True,
            callback_payoff_scene=None,
        )
    
    def _score_narrative_importance(
        self,
        elements: list[NarrativeElement],
        scenes: list[SceneSummary],
        transitions: list[StateTransition],
    ) -> None:
        """Score narrative importance for each element."""
        for elem in elements:
            score = 0.0
            
            # Recurrence score
            score += self._weights.get("recurrence", 0.3) * min(elem.recurrence_count / 5.0, 1.0)
            
            # Transition score
            if elem.element_type in ("object_disappearance", "object_reappearance", "relationship_change", "state_change"):
                score += self._weights.get("transition", 0.15)
            
            # Emotional/visual prominence (heuristic)
            if "character" in elem.element_type:
                score += self._weights.get("emotional", 0.2)
            elif "object" in elem.element_type:
                score += self._weights.get("visual_prominence", 0.2)
            
            # Uniqueness (inverse of frequency)
            similar_count = sum(1 for e in elements if e.label == elem.label and e != elem)
            if similar_count == 0:
                score += self._weights.get("uniqueness", 0.15)
            
            # Open loop bonus
            if elem.element_type == "open_loop":
                score += 0.2
            
            elem.narrative_importance = min(score, 1.0)
    
    def _mark_callback_potential(
        self,
        elements: list[NarrativeElement],
        scenes: list[SceneSummary],
    ) -> None:
        """Mark elements with callback potential."""
        # Elements from early scenes that recur later are good callback candidates
        for i, scene in enumerate(scenes):
            scene_elements = [e for e in elements if scene.scene_id in e.scene_ids]
            
            for elem in scene_elements:
                # Check if this element appears in later scenes
                later_scenes = scenes[i+1:]
                later_appearance = any(
                    scene.scene_id in elem.scene_ids for scene in later_scenes
                )
                
                if later_appearance and elem.element_type in ("recurring_entity", "scene_object", "open_loop"):
                    elem.potential_callback = True
                    
                    # Find payoff scene (last appearance)
                    if elem.scene_ids:
                        elem.callback_payoff_scene = elem.scene_ids[-1]


class NarrativeMemoryRetriever:
    """Retrieve narrative elements for story generation."""
    
    def __init__(self, narrative_elements: list[NarrativeElement]):
        self._elements = narrative_elements
        self._by_scene = defaultdict(list)
        self._by_type = defaultdict(list)
        self._by_entity = defaultdict(list)
        
        for elem in narrative_elements:
            for scene_id in elem.scene_ids:
                self._by_scene[scene_id].append(elem)
            self._by_type[elem.element_type].append(elem)
            for entity in elem.associated_entities:
                self._by_entity[entity].append(elem)
    
    def retrieve_for_scene(
        self,
        scene_id: str,
        top_k: int = 10,
        include_potential_callbacks: bool = True,
    ) -> list[NarrativeElement]:
        """Get narrative elements relevant to a scene."""
        elements = self._by_scene.get(scene_id, [])
        
        # Also include potential callbacks from earlier scenes
        if include_potential_callbacks:
            scene_idx = -1
            # This would need scene order - simplified for now
            for elem in self._elements:
                if elem.potential_callback and scene_id not in elem.scene_ids:
                    elements.append(elem)
        
        # Sort by narrative importance
        elements.sort(key=lambda e: e.narrative_importance, reverse=True)
        
        return elements[:top_k]
    
    def retrieve_callback_candidates(
        self,
        current_scene_id: str,
        query: str = "",
        top_k: int = 10,
    ) -> list[NarrativeElement]:
        """Retrieve elements suitable for callbacks."""
        candidates = []
        
        for elem in self._elements:
            # Must be from earlier scene and have callback potential
            if not elem.potential_callback:
                continue
            
            if current_scene_id in elem.scene_ids:
                continue  # Already in current scene
            
            # Score for callback relevance
            score = elem.narrative_importance
            
            # Boost for open loops
            if elem.element_type == "open_loop" and elem.callback_payoff_scene is None:
                score += 0.5
            
            # Boost for query relevance
            if query:
                if elem.label.lower() in query.lower():
                    score += 0.3
                if elem.description.lower() in query.lower():
                    score += 0.2
            
            candidates.append((elem, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in candidates[:top_k]]
    
    def retrieve_by_type(
        self,
        element_type: str,
        top_k: int = 10,
    ) -> list[NarrativeElement]:
        """Retrieve elements by type."""
        elements = self._by_type.get(element_type, [])
        elements.sort(key=lambda e: e.narrative_importance, reverse=True)
        return elements[:top_k]
    
    def retrieve_by_entity(
        self,
        entity_label: str,
        top_k: int = 10,
    ) -> list[NarrativeElement]:
        """Retrieve elements associated with an entity."""
        elements = self._by_entity.get(entity_label, [])
        elements.sort(key=lambda e: e.narrative_importance, reverse=True)
        return elements[:top_k]
    
    def get_open_loops(self) -> list[NarrativeElement]:
        """Get all unresolved open loops."""
        return [e for e in self._elements 
                if e.element_type == "open_loop" and e.callback_payoff_scene is None]


def create_narrative_memory(
    importance_weights: dict[str, float] | None = None,
) -> NarrativeMemory:
    """Factory function to create narrative memory."""
    return NarrativeMemory(importance_weights)