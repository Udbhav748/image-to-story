"""Surprise and twist engine for narrative generation."""
from typing import Any
import random

from ..domain.schemas import WorldEntity, WorldState, RetrievedEvidence
from ..domain.enums import EvidenceType


class SurpriseEngine:
    """Generate grounded surprises and twists."""
    
    SURPRISE_PATTERNS = [
        "recontextualization",
        "hidden_significance",
        "causal_chain",
        "perspective_shift",
        "callback_payoff",
    ]
    
    RECONTEXTUALIZATION_TEMPLATES = [
        "The {object} wasn't {assumed_function} - it was {true_function}.",
        "What {char} thought was {mistaken_identity} turned out to be {true_identity}.",
        "The {action} {char} performed wasn't {assumed_reason} - it was {true_reason}.",
    ]
    
    HIDDEN_SIGNIFICANCE_TEMPLATES = [
        "The {object} {char} carried contained {hidden_content}, unknown to everyone.",
        "The {detail} in {location} was actually {significance}.",
        "{char}'s {habit} wasn't just a quirk - it was {true_purpose}.",
    ]
    
    CAUSAL_CHAIN_TEMPLATES = [
        "{char}'s {small_action} in frame {frame} caused {major_consequence} now.",
        "The {event} {char} witnessed earlier was the reason {current_situation}.",
        "A {detail} from {earlier_frame} explained {current_mystery}.",
    ]
    
    PERSPECTIVE_SHIFT_TEMPLATES = [
        "From {other_perspective}, the {situation} looked completely different.",
        "{char} had been seeing {situation} wrong - {reveal} changed everything.",
        "The {object} looked {appearance} to {char}, but {other_entity} saw {true_nature}.",
    ]
    
    def __init__(self, seed: int = 0, surprise_level: float = 0.6):
        self._seed = seed
        self._rng = random.Random(seed)
        self._surprise_level = surprise_level
        self._used_details: set[str] = set()
    
    def generate_surprise(
        self,
        world_state: WorldState,
        evidence_summary: str,
        foreshadowing_elements: list[dict[str, Any]],
        retrieved_evidence: list[RetrievedEvidence] | None = None,
    ) -> dict[str, Any] | None:
        """Generate a grounded surprise/twist."""
        if self._rng.random() > self._surprise_level:
            return None
        
        # Prefer patterns that use actual evidence
        available_patterns = self.SURPRISE_PATTERNS.copy()
        if not foreshadowing_elements:
            available_patterns = [p for p in available_patterns if p != "callback_payoff"]
        if not retrieved_evidence:
            available_patterns = [p for p in available_patterns if p not in ["recontextualization", "hidden_significance"]]
        
        # If callback_payoff is available and foreshadowing exists, prioritize it
        if foreshadowing_elements and "callback_payoff" in available_patterns:
            return self._generate_callback_payoff(foreshadowing_elements, world_state)
        
        pattern = self._rng.choice(available_patterns)
        
        if pattern == "recontextualization":
            return self._generate_recontextualization(world_state, retrieved_evidence)
        elif pattern == "hidden_significance":
            return self._generate_hidden_significance(world_state, retrieved_evidence)
        elif pattern == "causal_chain":
            return self._generate_causal_chain(world_state, retrieved_evidence)
        elif pattern == "perspective_shift":
            return self._generate_perspective_shift(world_state, retrieved_evidence)
        else:
            return None
    
    def _generate_recontextualization(
        self, 
        world_state: WorldState, 
        retrieved_evidence: list[RetrievedEvidence] | None = None
    ) -> dict[str, Any]:
        # Use actual evidence entities when available
        if retrieved_evidence:
            objects = [e.record.entity for e in retrieved_evidence if e.record.type.value == "object" and e.record.information_class.value == "hard_fact"]
            characters = [e.record.entity for e in retrieved_evidence if e.record.type.value in ["person", "character"] and e.record.information_class.value == "hard_fact"]
        else:
            objects = [e.label for e in world_state.get_all_entities() if e.entity_type == "object"]
            characters = [e.label for e in world_state.get_all_entities() if e.entity_type == "character"]
        
        if not objects or not characters:
            return None
        
        obj = self._rng.choice(objects)
        char = self._rng.choice(characters)
        
        template = self._rng.choice(self.RECONTEXTUALIZATION_TEMPLATES)
        description = template.format(
            object=obj,
            char=char,
            assumed_function="ordinary",
            true_function="the key to everything",
            mistaken_identity="a simple object",
            true_identity="a message carrier",
            action="waiting",
            assumed_reason="boredom",
            true_reason="watching for a signal",
        )
        
        return {
            "type": "recontextualization",
            "description": description,
            "grounded_entities": [obj, char],
            "pattern": "recontextualization",
        }
    
    def _generate_hidden_significance(
        self, 
        world_state: WorldState, 
        retrieved_evidence: list[RetrievedEvidence] | None = None
    ) -> dict[str, Any]:
        if retrieved_evidence:
            objects = [e.record.entity for e in retrieved_evidence if e.record.type.value == "object" and e.record.information_class.value == "hard_fact"]
            characters = [e.record.entity for e in retrieved_evidence if e.record.type.value in ["person", "character"] and e.record.information_class.value == "hard_fact"]
        else:
            objects = [e.label for e in world_state.get_all_entities() if e.entity_type == "object"]
            characters = [e.label for e in world_state.get_all_entities() if e.entity_type == "character"]
        
        if not objects or not characters:
            return None
        
        obj = self._rng.choice(objects)
        char = self._rng.choice(characters)
        
        template = self._rng.choice(self.HIDDEN_SIGNIFICANCE_TEMPLATES)
        description = template.format(
            object=obj,
            char=char,
            hidden_content="a map to somewhere important",
            detail="scratched mark",
            location="the doorway",
            significance="a tally of days",
            habit="tapping the pocket",
            true_purpose="checking the object is still there",
        )
        
        return {
            "type": "hidden_significance",
            "description": description,
            "grounded_entities": [obj, char],
            "pattern": "hidden_significance",
        }
    
    def _generate_causal_chain(
        self, 
        world_state: WorldState, 
        retrieved_evidence: list[RetrievedEvidence] | None = None
    ) -> dict[str, Any]:
        events = world_state.previous_events
        if not events:
            return None
        
        # Use evidence from retrieved_evidence if available
        if retrieved_evidence:
            details = [e.record.entity for e in retrieved_evidence if e.record.information_class.value == "hard_fact"]
            if details:
                detail = self._rng.choice(details)
            else:
                detail = "forgotten detail"
        else:
            detail = "forgotten detail"
        
        template = self._rng.choice(self.CAUSAL_CHAIN_TEMPLATES)
        
        description = template.format(
            char="the protagonist",
            small_action="small gesture",
            frame="1",
            major_consequence="the current revelation",
            event="earlier moment",
            current_situation="this exact moment",
            detail=detail,
            earlier_frame="the first frame",
            current_mystery="why things are this way",
        )
        
        return {
            "type": "causal_chain",
            "description": description,
            "grounded_entities": [detail] if detail != "forgotten detail" else [],
            "pattern": "causal_chain",
        }
    
    def _generate_perspective_shift(
        self, 
        world_state: WorldState, 
        retrieved_evidence: list[RetrievedEvidence] | None = None
    ) -> dict[str, Any]:
        characters = [e for e in world_state.get_all_entities() if e.entity_type == "character"]
        objects = [e for e in world_state.get_all_entities() if e.entity_type == "object"]
        
        if len(characters) < 2 and not objects:
            return None
        
        char = self._rng.choice(characters)
        other = self._rng.choice(characters) if len(characters) > 1 else (self._rng.choice(objects) if objects else None)
        
        template = self._rng.choice(self.PERSPECTIVE_SHIFT_TEMPLATES)
        description = template.format(
            other_perspective=other.label if other else "above",
            situation="scene",
            reveal="the hidden truth",
            object=objects[0].label if objects else "item",
            appearance="ordinary",
            true_nature="extraordinary",
        )
        
        return {
            "type": "perspective_shift",
            "description": description,
            "grounded_entities": [char.label] + ([other.label] if other else []),
            "pattern": "perspective_shift",
        }
    
    def _generate_callback_payoff(
        self,
        foreshadowing_elements: list[dict[str, Any]],
        world_state: WorldState,
    ) -> dict[str, Any]:
        element = self._rng.choice(foreshadowing_elements)
        detail = element.get("detail", "a small detail")
        
        return {
            "type": "callback_payoff",
            "description": f"The {detail} from earlier was the key to understanding everything now.",
            "grounded_entities": [detail],
            "pattern": "callback_payoff",
            "foreshadowing_source": element,
        }
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng.seed(seed)
    
    def set_surprise_level(self, level: float) -> None:
        self._surprise_level = max(0.0, min(1.0, level))