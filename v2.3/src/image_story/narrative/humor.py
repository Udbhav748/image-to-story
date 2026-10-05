"""Humor engine for narrative generation."""
from typing import Any
import random

from ..domain.schemas import WorldEntity, WorldState, RetrievedEvidence
from ..domain.enums import HumorStyle, EvidenceType


class HumorEngine:
    """Generate humor elements grounded in visual evidence."""
    
    HUMOR_TEMPLATES = {
        HumorStyle.DEADPAN: [
            "{char} stared at the {object}. The {object} stared back. Neither blinked.",
            "It was a {adjective} day for a {noun}. {char} would know - they'd seen {number} of them.",
            "{char} sighed. '{dialogue}', they said to the {object}. The {object} remained unimpressed.",
        ],
        HumorStyle.AWKWARD: [
            "{char} waved at {other_char}, realized they were waving at a {object}, and kept waving anyway.",
            "The silence stretched. {char} cleared their throat. '{dialogue}', they offered. The {object} said nothing.",
            "{char} tried to {action} casually. The {object} fell over. {char} pretended it was intentional.",
        ],
        HumorStyle.SITUATIONAL: [
            "{char} needed a {tool} but only had a {wrong_tool}. It worked. Somehow.",
            "The {object} was exactly where {char} left it. This was suspicious.",
            "Everything was going according to plan. This worried {char} more than chaos would.",
        ],
        HumorStyle.MISUNDERSTANDING: [
            "{char} heard '{phrase}' and prepared for {wrong_preparation}. It was actually about {actual_topic}.",
            "{char} and {other_char} agreed perfectly on {topic}. They meant completely different things.",
            "The sign said '{sign_text}'. {char} interpreted this as '{misinterpretation}'.",
        ],
        HumorStyle.PERSONIFICATION: [
            "The {object} sighed. It had a long day of being a {object}.",
            "{char} swore the {object} rolled its eyes. {object}s don't have eyes. Probably.",
            "The {object} judged {char}'s life choices. It was a {object}; judgment was its purpose.",
        ],
        HumorStyle.CALLBACK: [
            "Just like the {previous_object} from {previous_frame}, this {object} also {behavior}.",
            "Remember when {char} said '{previous_quote}'? The {object} proved them right. Again.",
            "The {object} had the same {trait} as the {previous_object}. Coincidence? {char} thought not.",
        ],
    }
    
    # Evidence-grounded humor templates - use actual evidence
    GROUNDED_HUMOR_TEMPLATES = {
        "personification": [
            "The {object} sighed. It had a long day of being a {object}.",
            "{char} swore the {object} rolled its eyes. {object}s don't have eyes. Probably.",
        ],
        "situational_irony": [
            "{char} needed a {tool} but only had the {object}. It worked. Somehow.",
            "The {object} was exactly where {char} left it. This was suspicious.",
        ],
        "misunderstanding": [
            "{char} heard '{phrase}' and prepared for {wrong_preparation}. It was actually about {actual_topic}.",
            "The sign said '{sign_text}'. {char} interpreted this as '{misinterpretation}'.",
        ],
        "callback": [
            "Just like the {previous_object} from {previous_frame}, this {object} also {behavior}.",
            "Remember when {char} said '{previous_quote}'? The {object} proved them right. Again.",
        ],
    }
    
    def __init__(self, seed: int = 0, humor_level: float = 0.5):
        self._seed = seed
        self._rng = random.Random(seed)
        self._humor_level = humor_level
    
    def generate_humor_moments(
        self,
        world_state: WorldState,
        conflict: dict[str, Any],
        style: HumorStyle = HumorStyle.SITUATIONAL,
        count: int = 2,
        retrieved_evidence: list[RetrievedEvidence] | None = None,
    ) -> list[str]:
        """Generate humor moments for the story."""
        if self._rng.random() > self._humor_level:
            return []
        
        characters = [e for e in world_state.get_all_entities() if e.entity_type == "character"]
        objects = [e for e in world_state.get_all_entities() if e.entity_type == "object"]
        
        if not characters:
            return []
        
        char = self._rng.choice(characters)
        other_char = self._rng.choice(characters) if len(characters) > 1 else None
        obj = self._rng.choice(objects) if objects else None
        
        # Use evidence-grounded templates when evidence is available
        if retrieved_evidence:
            moments = self._generate_grounded_humor(
                char, other_char, objects, retrieved_evidence, world_state, count
            )
            if moments:
                return moments[:count]
        
        # Fallback to style-based templates
        templates = self.HUMOR_TEMPLATES.get(style, self.HUMOR_TEMPLATES[HumorStyle.SITUATIONAL])
        moments = []
        
        for _ in range(min(count, len(templates))):
            template = self._rng.choice(templates)
            moment = self._fill_humor_template(template, char, other_char, obj)
            moments.append(moment)
        
        return moments
    
    def _generate_grounded_humor(
        self,
        char: WorldEntity,
        other_char: WorldEntity | None,
        objects: list[WorldEntity],
        retrieved_evidence: list[RetrievedEvidence],
        world_state: WorldState,
        count: int,
    ) -> list[str]:
        """Generate humor grounded in actual evidence."""
        moments = []
        
        # Get hard facts from evidence
        hard_facts = [e.record.entity for e in retrieved_evidence if e.record.information_class.value == "hard_fact"]
        if not hard_facts:
            return []
        
        # Use actual evidence entities
        evidence_objects = [e for e in hard_facts if e in [o.label for o in objects]]
        if not evidence_objects:
            evidence_objects = hard_facts[:3]
        
        obj = self._rng.choice(evidence_objects)
        other_char = self._rng.choice([e for e in world_state.get_all_entities() if e.entity_type == "character" and e.label != char.label]) if len([e for e in world_state.get_all_entities() if e.entity_type == "character"]) > 1 else None
        
        # Grounded templates using actual evidence
        grounded_templates = [
            f"The {obj} was exactly where {char.label} left it. This was suspicious.",
            f"{char.label} needed a tool but only had the {obj}. It worked. Somehow.",
            f"Just like the {obj} from earlier, this situation also had a way of repeating itself.",
        ]
        
        for _ in range(min(2, len(grounded_templates))):
            template = self._rng.choice(grounded_templates)
            moments.append(template.format(
                char=char.label,
                obj=obj,
                other_char=other_char.label if other_char else "someone",
            ))
        
        return moments
    
    def _fill_humor_template(
        self,
        template: str,
        char: WorldEntity,
        other_char: WorldEntity | None,
        obj: WorldEntity | None,
    ) -> str:
        fillers = {
            "char": char.label,
            "other_char": other_char.label if other_char else "someone",
            "object": obj.label if obj else "thing",
            "adjective": "perfectly ordinary",
            "noun": "adventure",
            "number": "seventeen",
            "dialogue": "Well, that happened",
            "action": "pick it up",
            "absurd_identity": "secret agent",
            "topic": "the meaning of existence",
            "tool": "hammer",
            "wrong_tool": "banana",
            "phrase": "the eagle has landed",
            "wrong_preparation": "an alien invasion",
            "actual_topic": "a bird",
            "sign_text": "wet paint",
            "misinterpretation": "paint is wet, touch it",
            "previous_object": "red balloon",
            "previous_frame": "earlier",
            "behavior": "floated away",
            "previous_quote": "it'll come back",
            "trait": "mysterious disappearance habit",
            "effort": "building a complex machine",
            "mundane_result": "a sandwich",
            "unexpected_identity": "a key",
            "celebration": "tea",
        }
        
        for key, value in fillers.items():
            template = template.replace(f"{{{key}}}", value)
        
        return template
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng.seed(seed)
    
    def set_humor_level(self, level: float) -> None:
        self._humor_level = max(0.0, min(1.0, level))