"""Context builder: constructs structured context for story generation."""
from typing import Any

from ..domain.schemas import (
    VisualObservations,
    WorldState,
    WorldEntity,
    RetrievedEvidence,
    EvidenceRecord,
    InformationClass,
    CreativePlan,
)
from ..domain.enums import EvidenceType


class ContextBuilder:
    """Build structured context separating facts, inferences, and creative space."""
    
    def __init__(self, max_words: int = 500):
        self._max_words = max_words
    
    def build_context(
        self,
        observations: list[VisualObservations],
        world_state: WorldState,
        ranked_evidence: list[RetrievedEvidence],
        creative_plan: CreativePlan | None = None,
    ) -> str:
        """Build complete context for story generation."""
        sections = []
        
        hard_facts = self._extract_hard_facts(observations, ranked_evidence)
        soft_inferences = self._extract_soft_inferences(observations, ranked_evidence)
        creative_space = self._extract_creative_space(world_state, creative_plan)
        
        if hard_facts:
            sections.append("=== HARD FACTS (visually grounded) ===")
            sections.extend(hard_facts)
        
        if soft_inferences:
            sections.append("\n=== SOFT INFERENCES (plausible interpretations) ===")
            sections.extend(soft_inferences)
        
        if creative_space:
            sections.append("\n=== CREATIVE SPACE (narrative invention allowed) ===")
            sections.extend(creative_space)
        
        if creative_plan:
            sections.append("\n=== CREATIVE DIRECTION ===")
            sections.append(f"Genre: {creative_plan.genre}")
            sections.append(f"Tone: {creative_plan.tone}")
            if creative_plan.characters:
                sections.append("Characters:")
                for char in creative_plan.characters:
                    sections.append(f"  - {char.get('label', '')}: {char.get('personality', '')} "
                                  f"(goal: {char.get('goal', '')})")
            if creative_plan.central_conflict:
                sections.append(f"Central Conflict: {creative_plan.central_conflict.get('description', '')}")
            if creative_plan.open_loops:
                sections.append("Open Loops to Resolve:")
                for loop in creative_plan.open_loops:
                    sections.append(f"  - {loop}")
        
        context = "\n".join(sections)
        return self._truncate_to_words(context, self._max_words)
    
    def _extract_hard_facts(
        self,
        observations: list[VisualObservations],
        ranked_evidence: list[RetrievedEvidence],
    ) -> list[str]:
        facts = []
        
        for obs in observations:
            if obs.scene:
                facts.append(f"Scene: {obs.scene}")
            if obs.style_or_mood:
                facts.append(f"Visual style: {obs.style_or_mood}")
        
        entity_counts = {}
        for item in ranked_evidence:
            ev = item.record
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
        
        return facts[:20]
    
    def _extract_soft_inferences(
        self,
        observations: list[VisualObservations],
        ranked_evidence: list[RetrievedEvidence],
    ) -> list[str]:
        inferences = []
        
        for item in ranked_evidence:
            ev = item.record
            if ev.information_class == InformationClass.SOFT_INFERENCE:
                if ev.type == EvidenceType.ACTION:
                    inferences.append(f"Possibly {ev.entity} (confidence: {ev.confidence:.2f})")
                elif ev.type == EvidenceType.RELATIONSHIP:
                    inferences.append(f"Possible relationship: {ev.relationship} involving {ev.entity}")
        
        for obs in observations:
            if obs.characters and obs.actions:
                for char in obs.characters[:2]:
                    for action in obs.actions[:2]:
                        inferences.append(f"{char} may be {action}")
        
        return inferences[:10]
    
    def _extract_creative_space(
        self,
        world_state: WorldState,
        creative_plan: CreativePlan | None,
    ) -> list[str]:
        space = []
        
        for entity in world_state.get_all_entities():
            if entity.entity_type == "character":
                space.append(f"Character '{entity.label}': personality, motivation, backstory open for invention")
            elif entity.entity_type == "object" and entity.disappearance_frame:
                space.append(f"Object '{entity.label}' disappeared - reason and significance open for invention")
        
        for loop in world_state.open_loops:
            space.append(f"Open question: {loop.get('description', '')}")
        
        if creative_plan:
            for char in creative_plan.characters:
                space.append(f"Character {char.get('label', '')}: {char.get('personality', '')} "
                           f"with goal '{char.get('goal', '')}'")
        
        space.append("Dialogue, internal thoughts, humor, and metaphorical framing are creative inventions")
        space.append("Surprise twists must reinterpret existing evidence, not contradict hard facts")
        
        return space[:10]
    
    def _truncate_to_words(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "... [truncated]"
    
    def build_story_prompt(
        self,
        context: str,
        creative_plan: CreativePlan | None = None,
        target_words: int = 250,
    ) -> str:
        """Build the final prompt for the story generator."""
        
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
        
        prompt = f"""You are a creative storyteller. Write a story of approximately {target_words} words based on the following structured visual evidence and creative direction.

{context}

{locked_facts}{creativity_instruction}{creative_freedom}

RULES:
1. LOCKED VISUAL FACTS must NOT be contradicted, added to, or removed
2. SOFT INFERENCES are plausible interpretations - you may use or refine them
3. CREATIVE SPACE is where you invent: personalities, motivations, dialogue, humor, twists, metaphors
4. Any surprise/twist must REINTERPRET existing evidence, not invent unsupported objects/events
5. Maintain continuity if this is a multi-frame sequence
6. Resolve at least one open loop or mystery
7. Include at least one callback/foreshadowing payoff if setup exists

Return ONLY the story."""
        
        return prompt
    
    def build_single_image_context(self, observations: VisualObservations) -> str:
        """Legacy-compatible single image context."""
        facts = []
        if observations.scene:
            facts.append(f"Scene: {observations.scene}")
        if observations.characters:
            facts.append(f"Characters: {', '.join(observations.characters)}")
        if observations.od_labels:
            facts.append(f"Objects: {', '.join(observations.od_labels[:10])}")
        if observations.actions:
            facts.append(f"Actions: {', '.join(observations.actions)}")
        if observations.spatial_relations:
            facts.append(f"Spatial: {', '.join(observations.spatial_relations[:3])}")
        if observations.style_or_mood:
            facts.append(f"Style: {observations.style_or_mood}")
        
        context = ". ".join(facts) + "."
        return self._truncate_to_words(context, 180)