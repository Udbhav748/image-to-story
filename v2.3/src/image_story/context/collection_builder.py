"""Collection-level context building with compression for large image sets."""
from __future__ import annotations
from typing import Any
from dataclasses import dataclass

from ..domain.schemas import (
    VisualObservations,
    WorldState,
    CreativePlan,
    RetrievedEvidence,
    EvidenceRecord,
    InformationClass,
)
from ..domain.enums import EvidenceType
from ..context.builder import ContextBuilder
from ..context.sequence import SequenceContextBuilder
from ..memory.hierarchical import (
    SceneSummary,
    CollectionMemory,
    EntityMemory,
    StateTransition,
    NarrativeElement,
    RetrievalResult,
)


@dataclass
class CollectionContextConfig:
    """Configuration for collection context building."""
    max_scenes_in_context: int = 5
    max_evidence_per_scene: int = 8
    max_entities: int = 10
    max_transitions: int = 8
    max_narrative_elements: int = 6
    max_total_words: int = 800
    include_scene_summaries: bool = True
    include_entity_profiles: bool = True
    include_transitions: bool = True
    include_callbacks: bool = True
    prioritize_current_scene: bool = True


class CollectionContextBuilder(ContextBuilder):
    """Build compressed context for large image collections."""
    
    def __init__(
        self,
        max_words: int = 800,
        config: CollectionContextConfig | None = None,
    ):
        super().__init__(max_words=max_words)
        self._config = config or CollectionContextConfig()
        self._sequence_builder = SequenceContextBuilder(max_total_words=max_words)
    
    def build_collection_context(
        self,
        collection: CollectionMemory,
        retrieval_result: RetrievalResult,
        creative_plan: CreativePlan | None = None,
        current_scene_id: str | None = None,
    ) -> str:
        """
        Build context for story generation from collection.
        
        Uses hierarchical retrieval to compress large collections into
        manageable context while preserving key information.
        """
        sections = []
        
        # 1. Collection overview
        sections.append("=== COLLECTION OVERVIEW ===")
        sections.append(f"Total scenes: {len(collection.scene_summaries)}")
        sections.append(f"Total images: {collection.total_images}")
        sections.append(f"Total evidence: {collection.total_evidence}")
        
        # 2. Relevant scenes (from retrieval)
        if retrieval_result.scenes:
            sections.append("\n=== RELEVANT SCENES ===")
            for scene in retrieval_result.scenes[:self._config.max_scenes_in_context]:
                sections.append(self._format_scene_summary(scene))
        
        # 3. Current scene detail (if specified)
        if current_scene_id and self._config.prioritize_current_scene:
            current_scene = collection.get_scene(current_scene_id)
            if current_scene:
                sections.append(f"\n=== CURRENT SCENE DETAIL: {current_scene_id} ===")
                sections.append(self._format_detailed_scene(current_scene))
        
        # 4. Key entities
        if retrieval_result.entities and self._config.include_entity_profiles:
            sections.append("\n=== KEY ENTITIES ===")
            for entity in retrieval_result.entities[:self._config.max_entities]:
                sections.append(self._format_entity_profile(entity))
        
        # 5. State transitions
        if retrieval_result.transitions and self._config.include_transitions:
            sections.append("\n=== STATE TRANSITIONS ===")
            for trans in retrieval_result.transitions[:self._config.max_transitions]:
                sections.append(f"- {trans.description}")
        
        # 6. Narrative elements / callbacks
        if retrieval_result.narrative_elements and self._config.include_callbacks:
            sections.append("\n=== NARRATIVE ELEMENTS ===")
            for elem in retrieval_result.narrative_elements[:self._config.max_narrative_elements]:
                if elem.potential_callback or elem.element_type == "open_loop":
                    sections.append(f"- [{elem.element_type}] {elem.description}")
        
        # 7. Hard facts from retrieved evidence
        if retrieval_result.evidence:
            sections.append("\n=== HARD FACTS (from retrieved evidence) ===")
            hard_facts = self._extract_hard_facts_from_evidence(retrieval_result.evidence)
            sections.extend(hard_facts[:15])
        
        # 8. Creative direction
        if creative_plan:
            sections.append("\n=== CREATIVE DIRECTION ===")
            sections.append(f"Genre: {creative_plan.genre}")
            sections.append(f"Tone: {creative_plan.tone}")
            if creative_plan.locked_facts:
                sections.append(f"LOCKED FACTS: {', '.join(creative_plan.locked_facts[:10])}")
        
        context = "\n".join(sections)
        return self._truncate_to_words(context, self._config.max_total_words)
    
    def build_multi_scene_context(
        self,
        scenes: list[SceneSummary],
        entities: list[EntityMemory],
        evidence: list[EvidenceRecord],
        creative_plan: CreativePlan | None = None,
    ) -> str:
        """
        Build context for multi-scene story (2-10 scenes).
        Uses more detailed per-scene information.
        """
        sections = ["=== SCENE SEQUENCE ==="]
        
        for i, scene in enumerate(scenes):
            sections.append(f"\n--- SCENE {i + 1}: {scene.scene_id} ---")
            sections.append(f"Frames: {scene.start_frame}-{scene.end_frame} ({len(scene.image_ids)} images)")
            if scene.location:
                sections.append(f"Location: {scene.location}")
            if scene.environment:
                sections.append(f"Environment: {scene.environment}")
            if scene.dominant_entities:
                sections.append(f"Characters: {', '.join(scene.dominant_entities[:5])}")
            if scene.important_objects:
                sections.append(f"Objects: {', '.join(scene.important_objects[:8])}")
            if scene.actions:
                sections.append(f"Actions: {', '.join(scene.actions[:5])}")
            if scene.summary_text:
                sections.append(f"Summary: {scene.summary_text}")
        
        # Continuity notes
        if len(scenes) > 1:
            sections.append(self._build_collection_continuity_notes(scenes, entities))
        
        # Entities
        if entities:
            sections.append("\n=== ENTITIES ===")
            for entity in entities[:10]:
                sections.append(f"- {entity.entity_type}: {entity.normalized_label} "
                              f"(scenes: {len(entity.scene_ids)}, confidence: {entity.confidence:.2f})")
        
        # Evidence
        if evidence:
            sections.append("\n=== KEY EVIDENCE ===")
            entity_counts = {}
            for ev in evidence:
                if ev.information_class == InformationClass.HARD_FACT:
                    key = f"{ev.entity}|{ev.type.value}"
                    entity_counts[key] = entity_counts.get(key, 0) + 1
            
            for key, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:15]:
                entity, etype = key.split("|", 1)
                if etype in ["person", "character"]:
                    sections.append(f"Character: {entity}")
                elif etype == "object":
                    sections.append(f"Object: {entity}")
                elif etype == "action":
                    sections.append(f"Action: {entity}")
        
        context = "\n".join(sections)
        
        if creative_plan:
            context += f"\n\n=== CREATIVE DIRECTION ===\nGenre: {creative_plan.genre}\nTone: {creative_plan.tone}"
            if creative_plan.locked_facts:
                context += f"\nLOCKED FACTS: {', '.join(creative_plan.locked_facts[:10])}"
        
        return self._truncate_to_words(context, self._config.max_total_words)
    
    def _format_scene_summary(self, scene: SceneSummary) -> str:
        """Format scene for context."""
        parts = [
            f"Scene {scene.scene_id}: {scene.summary_text}",
            f"  Frames: {scene.start_frame}-{scene.end_frame} ({len(scene.image_ids)} images)",
        ]
        
        if scene.dominant_entities:
            parts.append(f"  Characters: {', '.join(scene.dominant_entities[:4])}")
        if scene.important_objects:
            parts.append(f"  Objects: {', '.join(scene.important_objects[:6])}")
        if scene.actions:
            parts.append(f"  Actions: {', '.join(scene.actions[:4])}")
        if scene.recurring_entities:
            parts.append(f"  Recurring: {', '.join(scene.recurring_entities[:4])}")
        
        return "\n".join(parts)
    
    def _format_detailed_scene(self, scene: SceneSummary) -> str:
        """Format detailed scene view."""
        parts = [
            f"Location: {scene.location}" if scene.location else "",
            f"Environment: {scene.environment}" if scene.environment else "",
            f"Characters: {', '.join(scene.dominant_entities)}" if scene.dominant_entities else "",
            f"Objects: {', '.join(scene.important_objects[:10])}" if scene.important_objects else "",
            f"Actions: {', '.join(scene.actions[:6])}" if scene.actions else "",
        ]
        parts = [p for p in parts if p]
        return "\n".join(parts)
    
    def _format_entity_profile(self, entity: EntityMemory) -> str:
        """Format entity profile for context."""
        parts = [
            f"{entity.entity_type.capitalize()}: {entity.normalized_label}",
            f"  Scenes: {len(entity.scene_ids)} ({', '.join(entity.scene_ids[:5])})",
            f"  Confidence: {entity.confidence:.2f}",
            f"  State: {entity.state}",
        ]
        
        if entity.attributes.get("actions"):
            parts.append(f"  Actions: {', '.join(set(entity.attributes['actions']))}")
        if entity.attributes.get("relationships"):
            parts.append(f"  Relationships: {', '.join(set(entity.attributes['relationships']))}")
        
        return "\n".join(parts)
    
    def _build_collection_continuity_notes(
        self,
        scenes: list[SceneSummary],
        entities: list[EntityMemory],
    ) -> str:
        """Build continuity notes for multi-scene collection."""
        notes = ["\n=== CONTINUITY NOTES ==="]
        
        # Recurring entities
        recurring_chars = [e for e in entities if e.entity_type == "character" and len(e.scene_ids) >= 2]
        recurring_objs = [e for e in entities if e.entity_type == "object" and len(e.scene_ids) >= 2]
        
        if recurring_chars:
            notes.append(f"Recurring characters: {', '.join(e.normalized_label for e in recurring_chars)}")
        if recurring_objs:
            notes.append(f"Recurring objects: {', '.join(e.normalized_label for e in recurring_objs[:8])}")
        
        # Disappeared entities
        disappeared = [e for e in entities if e.state == "disappeared"]
        if disappeared:
            notes.append("Disappeared: " + ", ".join(e.normalized_label for e in disappeared))
        
        # Scene transitions
        notes.append("Scene progression: " + " -> ".join(s.scene_id for s in scenes))
        
        return "\n".join(notes)
    
    def _extract_hard_facts_from_evidence(self, evidence: list[EvidenceRecord]) -> list[str]:
        """Extract hard facts from evidence list."""
        facts = []
        entity_counts = {}
        
        for ev in evidence:
            if ev.information_class == InformationClass.HARD_FACT:
                key = f"{ev.entity}|{ev.type.value}"
                entity_counts[key] = entity_counts.get(key, 0) + 1
        
        for key, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            entity, etype = key.split("|", 1)
            if etype in ["person", "character"]:
                facts.append(f"Character present: {entity}")
            elif etype == "object":
                facts.append(f"Object visible: {entity}")
            elif etype == "action":
                facts.append(f"Action observed: {entity}")
            elif etype == "spatial_fact":
                facts.append(f"Spatial relation: {entity}")
            elif etype == "ocr":
                facts.append(f"Text visible: {entity}")
        
        return facts
    
    def build_multi_scene_prompt(
        self,
        context: str,
        creative_plan: CreativePlan | None = None,
        target_words: int = 250,
    ) -> str:
        """
        Build a story prompt from collection context.
        """
        parts = [f"Write a {creative_plan.genre if creative_plan else 'whimsical'} {creative_plan.tone if creative_plan else 'comedic'} story of approximately {target_words} words."]
        parts.append("")
        parts.append(context)
        parts.append("")
        parts.append("Write the story now:")
        return "\n".join(parts)
    
    def build_single_image_fallback(
        self,
        observation: VisualObservations,
        creative_plan: CreativePlan | None = None,
    ) -> str:
        """
        Fallback for single image - use existing single image context.
        """
        return self.build_single_image_context(observation)


def create_collection_context_builder(
    max_words: int = 800,
    max_scenes: int = 5,
    max_evidence_per_scene: int = 8,
) -> CollectionContextBuilder:
    """Factory function to create collection context builder."""
    config = CollectionContextConfig(
        max_scenes_in_context=max_scenes,
        max_evidence_per_scene=max_evidence_per_scene,
        max_total_words=max_words,
    )
    return CollectionContextBuilder(max_words, config)