"""Continuity evaluation for multi-image stories."""
import re
from typing import Any
from collections import Counter

from ..domain.schemas import VisualObservations, WorldState, EvaluationResult
from ..context.world_state import EntityTracker


class ContinuityEvaluator:
    """Evaluate story continuity across multiple frames."""
    
    TRANSITION_WORDS = {
        "then", "next", "after", "afterwards", "later", "suddenly", "meanwhile",
        "before", "finally", "soon", "eventually", "following",
    }
    
    def __init__(self):
        self._entity_tracker = EntityTracker()
    
    def evaluate_continuity(
        self,
        observations: list[VisualObservations],
        story: str,
        world_state: WorldState | None = None,
    ) -> dict[str, Any]:
        """Evaluate continuity metrics for multi-image story."""
        
        if len(observations) < 2:
            return {
                "num_frames": len(observations),
                "recurring_entities": [],
                "recurring_count": 0,
                "entity_consistency": None,
                "transition_markers": 0,
                "adjacent_similarity": None,
            }
        
        # Build frame entity sets
        frame_entities = []
        for obs in observations:
            entities = set()
            for char in obs.characters:
                entities.add(self._normalize_entity(char))
            for obj in obs.od_labels:
                entities.add(self._normalize_entity(obj))
            frame_entities.append(entities)
        
        # Find recurring entities
        counts = Counter()
        for entities in frame_entities:
            for e in entities:
                counts[e] += 1
        recurring = sorted(e for e, c in counts.items() if c >= 2)
        
        # Check entity consistency in story
        story_normalized = " " + " ".join(self._normalize_entity(w) for w in re.findall(r"[a-z']+", story.lower())) + " "
        mentioned = [e for e in recurring if f" {e} " in story_normalized]
        entity_consistency = len(mentioned) / len(recurring) if recurring else None
        
        # Transition markers
        story_words = re.findall(r"[a-z]+", story.lower())
        transition_markers = sum(1 for w in story_words if w in self.TRANSITION_WORDS)
        
        # Adjacent similarity
        overlaps = []
        for a, b in zip(frame_entities, frame_entities[1:]):
            if a | b:
                overlaps.append(len(a & b) / len(a | b))
        adjacent_similarity = sum(overlaps) / len(overlaps) if overlaps else None
        
        return {
            "num_frames": len(observations),
            "recurring_entities": recurring,
            "recurring_count": len(recurring),
            "entity_consistency": round(entity_consistency, 4) if entity_consistency is not None else None,
            "transition_markers": transition_markers,
            "adjacent_similarity": round(adjacent_similarity, 4) if adjacent_similarity is not None else None,
        }
    
    def _normalize_entity(self, label: str) -> str:
        words = re.sub(r"\s+", " ", str(label).strip().lower()).split(" ")
        if words == [""]:
            return ""
        w = words[-1]
        irregulars = {"people": "person", "children": "child", "men": "man", "women": "woman", "feet": "foot"}
        if w in irregulars:
            w = irregulars[w]
        elif w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        words[-1] = w
        return " ".join(words)
    
    def evaluate_entity_consistency(
        self,
        world_state: WorldState,
        story: str,
    ) -> dict[str, Any]:
        """Evaluate how well story maintains entity consistency."""
        if not world_state:
            return {}
        
        story_lower = story.lower()
        results = {}
        
        for entity in world_state.get_all_entities():
            entity_name = entity.label.lower()
            mentioned = entity_name in story_lower
            frame_mentions = entity.frames_present
            
            results[entity.id] = {
                "label": entity.label,
                "entity_type": entity.entity_type,
                "frames_present": frame_mentions,
                "is_recurring": entity.is_recurring,
                "mentioned_in_story": mentioned,
                "disappearance_frame": entity.disappearance_frame,
            }
        
        return results
    
    def evaluate_narrative_coherence(
        self,
        story: str,
        world_state: WorldState | None = None,
    ) -> float:
        """Simple narrative coherence proxy."""
        # This is a placeholder for more sophisticated coherence evaluation
        # Could use entity grid, discourse parsing, etc.
        
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", story.strip()) if s.strip()]
        if len(sentences) < 2:
            return 0.0
        
        # Simple proxy: entity overlap between adjacent sentences
        sentence_entities = []
        for sent in sentences:
            entities = set(self._normalize_entity(w) for w in re.findall(r"[a-z']+", sent.lower()))
            sentence_entities.append(entities)
        
        overlaps = []
        for a, b in zip(sentence_entities, sentence_entities[1:]):
            if a | b:
                overlaps.append(len(a & b) / len(a | b))
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0