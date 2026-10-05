"""Sequence-level context building for multi-frame stories."""
from typing import Any
from collections import Counter

from ..domain.schemas import VisualObservations, WorldState, WorldEntity, RetrievedEvidence
from ..domain.enums import EvidenceType
from .builder import ContextBuilder


class SequenceContextBuilder(ContextBuilder):
    """Build context for multi-image sequences with continuity tracking."""
    
    def __init__(self, max_words_per_image: int = 120, max_total_words: int = 800):
        super().__init__(max_words=max_total_words)
        self._max_words_per_image = max_words_per_image
    
    def build_sequence_context(
        self,
        observations: list[VisualObservations],
        world_state: WorldState,
        ranked_evidence: list[RetrievedEvidence],
        creative_plan: Any = None,
    ) -> str:
        """Build sequence-level context with continuity notes."""
        sections = ["=== SEQUENCE OF IMAGES (in order) ==="]
        
        for i, obs in enumerate(observations):
            sections.append(f"\n--- IMAGE {i + 1} ---")
            if obs.scene:
                sections.append(f"Location: {obs.scene}")
            if obs.characters:
                sections.append(f"Characters: {', '.join(obs.characters)}")
            if obs.od_labels:
                sections.append(f"Objects: {', '.join(obs.od_labels[:8])}")
            if obs.actions:
                sections.append(f"Actions: {', '.join(obs.actions)}")
            if obs.region_descriptions:
                sections.append(f"Details: {', '.join(obs.region_descriptions[:2])}")
            if obs.style_or_mood:
                sections.append(f"Mood: {obs.style_or_mood}")
        
        if len(observations) > 1:
            sections.append(self._build_continuity_notes(observations, world_state))
        
        sections.append("\n=== RETRIEVED EVIDENCE (ranked) ===")
        for item in ranked_evidence[:15]:
            ev = item.record
            info_class = ev.information_class.value.upper()
            sections.append(f"[{info_class}] Frame {ev.frame_id}: {ev.evidence_text[:120]}")
        
        context = "\n".join(sections)
        return self._truncate_to_words(context, self._max_words)
    
    def _build_continuity_notes(
        self,
        observations: list[VisualObservations],
        world_state: WorldState,
    ) -> str:
        notes = ["\n=== CONTINUITY NOTES ==="]
        
        recurring_chars = []
        recurring_objs = []
        
        for entity in world_state.get_all_entities():
            if entity.is_recurring:
                if entity.entity_type == "character":
                    recurring_chars.append(entity.label)
                else:
                    recurring_objs.append(entity.label)
        
        if recurring_chars:
            notes.append(f"Recurring characters (in 2+ frames): {', '.join(recurring_chars)}")
        if recurring_objs:
            notes.append(f"Recurring objects (in 2+ frames): {', '.join(recurring_objs[:10])}")
        
        disappeared = world_state.get_disappeared_entities()
        if disappeared:
            notes.append("Objects/entities that disappeared:")
            for e in disappeared:
                notes.append(f"  - {e.label} (last seen frame {e.last_frame})")
        
        open_loops = world_state.open_loops
        if open_loops:
            notes.append("Open narrative questions:")
            for loop in open_loops[:5]:
                notes.append(f"  - {loop.get('description', '')}")
        
        notes.append("Maintain character identity and location consistency across frames.")
        notes.append("Connect events naturally; do not reset the story at each image.")
        notes.append("Use recurring elements as callbacks and foreshadowing payoffs.")
        
        return "\n".join(notes)
    
    def build_multi_image_prompt(
        self,
        context: str,
        num_images: int,
        creative_plan: Any = None,
        target_words: int = 250,
    ) -> str:
        """Build prompt for multi-image story generation."""
        
        creativity_instruction = ""
        locked_facts = ""
        creative_freedom = ""
        if creative_plan:
            creativity_instruction = f"""
Creative direction:
- Genre: {creative_plan.genre}
- Tone: {creative_plan.tone}
- Creativity level: {creative_plan.creativity_level:.1f}/1.0
- Surprise level: {creative_plan.surprise_level:.1f}/1.0
- Humor level: {creative_plan.humor_level:.1f}/1.0
- Mystery level: {creative_plan.mystery_level:.1f}/1.0
"""
            if creative_plan.locked_facts:
                locked_facts = f"""
LOCKED VISUAL FACTS (MUST NOT CONTRADICT):
{', '.join(creative_plan.locked_facts[:15])}

These are visually verified facts. Do not contradict, add, or remove them.
"""
                creative_freedom = f"""
CREATIVE FREEDOM (ENCOURAGED):
You ARE ENCOURAGED to invent:
- Character personalities, quirks, attitudes, internal thoughts
- Motivations, desires, fears, hopes, secrets
- Dialogue, humor, irony, sarcasm, deadpan delivery
- Metaphors, similes, personification, narrative voice
- Backstories, relationships, emotional arcs
- Humor, irony, surprise, callbacks, narrative framing

These are NARRATIVE INVENTIONS - they do not need visual evidence.
They are encouraged to make the story engaging and meaningful.
"""
        
        prompt = f"""You are a creative storyteller. Here are {num_images} consecutive images from a sequence.

{context}

{locked_facts}{creativity_instruction}{creative_freedom}

Write ONE continuous story of about {target_words} words that follows these images in order, using only what the descriptions say. Maintain character identity and location consistency. Connect events naturally between frames. Do not describe each image separately.

RULES:
1. LOCKED VISUAL FACTS must NOT be contradicted, added to, or removed
2. SOFT INFERENCES are plausible interpretations - you may use or refine them
3. CREATIVE SPACE is where you invent: personalities, motivations, dialogue, humor, twists, metaphors
4. Any surprise/twist must REINTERPRET existing evidence, not invent unsupported objects/events
5. Resolve at least one open loop or mystery from the continuity notes
6. Include at least one callback to a recurring element (character, object, or detail)

Return ONLY the story."""
        
        return prompt