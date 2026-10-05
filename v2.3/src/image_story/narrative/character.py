"""Character system for narrative generation."""
from typing import Any
import random

from ..domain.schemas import WorldEntity, CreativePlan
from ..domain.enums import StoryGenre, StoryTone


class CharacterProfile:
    """Detailed character profile for story generation."""
    
    ARCHETYPES = {
        "hero": ["brave", "determined", "selfless", "curious"],
        "trickster": ["clever", "mischievous", "unpredictable", "witty"],
        "mentor": ["wise", "patient", "cryptic", "guiding"],
        "innocent": ["naive", "hopeful", "pure", "wondering"],
        "grump": ["cynical", "gruff", "secretly_kind", "complaining"],
        "dreamer": ["imaginative", "distracted", "hopeful", "whimsical"],
        "skeptic": ["analytical", "doubting", "grounded", "pragmatic"],
        "guardian": ["protective", "loyal", "stern", "watchful"],
    }
    
    QUIRKS = [
        "talks to inanimate objects",
        "collects unusual things",
        "speaks in rhymes when nervous",
        "has a lucky charm",
        "whistles when thinking",
        "organizes everything by color",
        "gives everything nicknames",
        "always carries a snack",
        "quotes fictional proverbs",
        "pretends to be a narrator",
        "counts steps compulsively",
        "names every stray animal",
        "taps fingers when thinking",
        "hums when nervous",
        "checks pockets repeatedly",
        "speaks to animals",
    ]
    
    DEEPER_TRAITS = [
        "secretly fears abandonment",
        "desperately wants to be understood",
        "hides pain behind humor",
        "carries guilt from past mistake",
        "yearns for connection but pushes people away",
        "believes they're not good enough",
        "secretly ambitious",
        "haunted by a past failure",
        "fiercely protective of loved ones",
        "struggles with self-doubt",
    ]
    
    INTERNAL_CONFLICTS = [
        "wants to help but fears getting hurt",
        "wants to be honest but fears rejection",
        "wants to lead but doubts their ability",
        "wants to trust but has been betrayed",
        "wants to stay but feels the need to run",
    ]
    
    def __init__(
        self,
        entity: WorldEntity,
        genre: StoryGenre = StoryGenre.WHIMSICAL,
        tone: StoryTone = StoryTone.COMEDIC,
        rng: random.Random = None,
    ):
        self.entity = entity
        self.genre = genre
        self.tone = tone
        self._rng = rng or random
        self.archetype = self._rng.choice(list(self.ARCHETYPES.keys()))
        self.personality_traits = self.ARCHETYPES[self.archetype][:]
        self.quirk = self._rng.choice(self.QUIRKS)
        self.deeper_trait = self._rng.choice(self.DEEPER_TRAITS)
        self.internal_conflict = self._rng.choice(self.INTERNAL_CONFLICTS)
        self.motivation = ""
        self.goal = ""
        self.emotional_state = "neutral"
        self.relationship_to_others = {}
        self._generate_motivation_and_goal()
    
    def _generate_motivation_and_goal(self) -> None:
        motivations = {
            "hero": ["to protect someone", "to do the right thing", "to prove themselves"],
            "trickster": ["to outsmart everyone", "to have fun", "to uncover secrets"],
            "mentor": ["to pass on knowledge", "to guide the lost", "to atone for past"],
            "innocent": ["to understand the world", "to find wonder", "to make friends"],
            "grump": ["to be left alone", "to fix what's broken", "to prove competence"],
            "dreamer": ["to make dreams real", "to find magic", "to escape reality"],
            "skeptic": ["to find the truth", "to debunk myths", "to be prepared"],
            "guardian": ["to keep watch", "to prevent harm", "to honor a promise"],
        }
        
        self.motivation = self._rng.choice(motivations.get(self.archetype, ["to understand"]))
        self.goal = f"{self.motivation} by {self._rng.choice(['exploring', 'investigating', 'creating', 'connecting', 'protecting', 'understanding'])}"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.entity.label,
            "archetype": self.archetype,
            "personality": ", ".join(self.personality_traits),
            "quirk": self.quirk,
            "deeper_trait": self.deeper_trait,
            "internal_conflict": self.internal_conflict,
            "motivation": self.motivation,
            "goal": self.goal,
            "emotional_state": self.emotional_state,
            "entity_type": self.entity.entity_type,
        }


class CharacterSystem:
    """Generate and manage character profiles for story."""
    
    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = random.Random(seed)
        self._profiles: dict[str, CharacterProfile] = {}
    
    def generate_profiles(
        self,
        world_state: Any,
        genre: StoryGenre = StoryGenre.WHIMSICAL,
        tone: StoryTone = StoryTone.COMEDIC,
    ) -> list[dict[str, Any]]:
        """Generate character profiles for all characters in world state."""
        self._rng.seed(self._seed)
        profiles = []
        
        for entity in world_state.get_all_entities():
            if entity.entity_type == "character":
                profile = CharacterProfile(entity, genre, tone, rng=self._rng)
                self._profiles[entity.id] = profile
                profiles.append(profile.to_dict())
        
        return profiles
    
    def get_profile(self, entity_id: str) -> CharacterProfile | None:
        return self._profiles.get(entity_id)
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng.seed(seed)