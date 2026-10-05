"""Conflict engine for narrative generation."""
from typing import Any
import random

from ..domain.schemas import WorldState, WorldEntity
from ..domain.enums import ConflictType


class ConflictEngine:
    """Generate narrative conflicts grounded in visual evidence."""
    
    CONFLICT_TEMPLATES = {
        ConflictType.MISUNDERSTANDING: [
            "{char1} thinks {char2} wants {object}, but {char2} actually wants {other_goal}.",
            "A simple gesture by {char1} is misinterpreted by {char2} as {misinterpretation}.",
            "{char1} and {char2} are at cross-purposes, each assuming the other knows the plan.",
        ],
        ConflictType.MISSING_OBJECT: [
            "{char1} discovers {object} is missing from where they left it.",
            "{object} has vanished, and {char1} must find it before {deadline}.",
            "The crucial {object} is gone - {char1} retraces their steps through {location}.",
        ],
        ConflictType.TIME_PRESSURE: [
            "{char1} has until {deadline} to {goal}, but {obstacle} stands in the way.",
            "The {event} starts in {time}, and {char1} is {distance} away.",
            "Every minute counts as {char1} races against {time_pressure} to {goal}.",
        ],
        ConflictType.SOCIAL_AWKWARDNESS: [
            "{char1} accidentally {awkward_action} in front of {char2}.",
            "{char1} must {social_task} but {complication} makes it painfully awkward.",
            "An introduction goes wrong when {char1} {mistake} instead of {proper_action}.",
        ],
        ConflictType.UNEXPECTED_DISCOVERY: [
            "{char1} finds {discovery} inside {object}, changing everything.",
            "Behind {location} lies {discovery} that {char1} never expected.",
            "What {char1} thought was {assumption} turns out to be {reality}.",
        ],
        ConflictType.COMPETING_GOALS: [
            "{char1} wants {goal1} while {char2} wants {goal2}, and only one can succeed.",
            "The {resource} can only go to one - {char1} and {char2} both need it.",
            "{char1}'s plan to {goal} directly conflicts with {char2}'s plan to {other_goal}.",
        ],
        ConflictType.MYSTERY: [
            "{char1} notices {clue} that doesn't make sense in {location}.",
            "A {mystery_object} appears with no explanation, and {char1} must investigate.",
            "Something about {scene} is wrong - {char1} spots {anomaly}.",
        ],
        ConflictType.EMBARRASSMENT: [
            "{char1}'s {secret} is accidentally revealed to {char2}.",
            "{char1} tries to {impress_action} but {failure} happens instead.",
            "{char1} is caught {embarrassing_act} by {char2} at the worst moment.",
        ],
        ConflictType.SURPRISING_COINCIDENCE: [
            "{char1} and {char2} both reach for {object} at the exact same moment.",
            "{char1} discovers {char2} is connected to {past_event} in an impossible way.",
            "The {object} {char1} lost years ago appears in {char2}'s possession.",
        ],
        ConflictType.ENVIRONMENTAL_OBSTACLE: [
            "{char1} must cross {obstacle} to reach {goal}, but {complication}.",
            "The {weather} makes {char1}'s journey to {location} treacherous.",
            "{char1}'s path is blocked by {obstacle}, forcing a creative solution.",
        ],
    }
    
    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = random.Random(seed)
    
    def generate_conflict(
        self,
        world_state: WorldState,
        evidence_summary: str,
        genre: str = "whimsical",
    ) -> dict[str, Any]:
        """Generate a conflict based on world state and evidence."""
        characters = [e for e in world_state.get_all_entities() if e.entity_type == "character"]
        objects = [e for e in world_state.get_all_entities() if e.entity_type == "object"]
        locations = [e for e in world_state.get_all_entities() if e.entity_type == "location"]
        
        if not characters:
            return {"type": "none", "description": "No characters for conflict"}
        
        char1 = self._rng.choice(characters)
        char2 = self._rng.choice(characters) if len(characters) > 1 else None
        obj = self._rng.choice(objects) if objects else None
        loc = self._rng.choice(locations) if locations else None
        
        open_loops = world_state.open_loops
        conflict_type = self._select_conflict_type(open_loops)
        
        template = self._rng.choice(self.CONFLICT_TEMPLATES[conflict_type])
        
        description = self._fill_template(
            template, char1, char2, obj, loc, conflict_type
        )
        
        return {
            "type": conflict_type.value,
            "description": description,
            "involved_entities": [char1.label] + ([char2.label] if char2 else []),
            "key_objects": [obj.label] if obj else [],
            "location": loc.label if loc else "",
            "grounded_in_evidence": True,
        }
    
    def _select_conflict_type(self, open_loops: list[dict]) -> ConflictType:
        if not open_loops:
            return self._rng.choice([
                ConflictType.MISUNDERSTANDING,
                ConflictType.UNEXPECTED_DISCOVERY,
                ConflictType.SOCIAL_AWKWARDNESS,
                ConflictType.MYSTERY,
            ])
        
        loop_types = [loop.get("type", "") for loop in open_loops]
        
        if "disappearance" in loop_types:
            return ConflictType.MISSING_OBJECT
        if "object_transfer" in loop_types:
            return ConflictType.UNEXPECTED_DISCOVERY
        
        return self._rng.choice(list(ConflictType))
    
    def _fill_template(
        self,
        template: str,
        char1: WorldEntity,
        char2: WorldEntity | None,
        obj: WorldEntity | None,
        loc: WorldEntity | None,
        conflict_type: ConflictType,
    ) -> str:
        fillers = {
            "char1": char1.label,
            "char2": char2.label if char2 else "someone",
            "object": obj.label if obj else "something important",
            "other_goal": "a different goal",
            "misinterpretation": "a threat",
            "deadline": "sundown",
            "goal": "accomplish their task",
            "obstacle": "an unexpected barrier",
            "event": "gathering",
            "time": "an hour",
            "distance": "far",
            "time_pressure": "the clock",
            "awkward_action": "says the wrong name",
            "social_task": "introduce themselves",
            "complication": "they've met before but forgot",
            "mistake": "insults their hat",
            "proper_action": "compliments it",
            "discovery": "a hidden note",
            "assumption": "ordinary",
            "reality": "extraordinary",
            "goal1": "keep the object",
            "goal2": "take the object",
            "resource": "the last seat",
            "clue": "a single footprint",
            "mystery_object": "sealed envelope",
            "scene": "the room",
            "anomaly": "a misplaced item",
            "secret": "hidden talent",
            "impress_action": "juggle",
            "failure": "drop everything",
            "embarrassing_act": "dancing alone",
            "past_event": "the lost expedition",
            "weather": "sudden storm",
        }
        
        for key, value in fillers.items():
            template = template.replace(f"{{{key}}}", value)
        
        return template
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng.seed(seed)