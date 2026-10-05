"""Creative story planner - orchestrates all narrative engines."""
from typing import Any
import random

from ..domain.schemas import (
    CreativePlan,
    StoryPlan,
    StoryBeat,
    WorldState,
    VisualObservations,
    RetrievedEvidence,
    EvidenceRecord,
    PipelineConfig,
)
from ..domain.enums import (
    StoryGenre,
    StoryTone,
    BeatType,
    ConflictType,
    HumorStyle,
)
from .character import CharacterSystem
from .conflict import ConflictEngine
from .humor import HumorEngine
from .surprise import SurpriseEngine


class CreativePlanner:
    """Main creative planning engine that orchestrates all narrative components."""
    
    def __init__(
        self,
        seed: int = 0,
        creativity_config: dict[str, float] | None = None,
        creative_budget: dict[str, int] | None = None,
        genre: str = "whimsical",
        tone: str = "comedic",
    ):
        self._seed = seed
        self._rng = random.Random(seed)
        self._creativity_config = creativity_config or {
            "creativity": 0.7,
            "surprise": 0.6,
            "humor": 0.5,
            "mystery": 0.4,
            "emotion": 0.5,
            "dialogue": 0.4,
            "metaphor": 0.3,
        }
        self._creative_budget = creative_budget or {
            "max_creative_claims": 6,
            "max_visual_inventions": 0,
            "max_soft_inferences": 3,
            "max_new_named_entities": 0,
        }
        self._genre = StoryGenre(genre) if isinstance(genre, str) else genre
        self._tone = StoryTone(tone) if isinstance(tone, str) else tone
        
        # Initialize sub-engines
        self._character_system = CharacterSystem(seed)
        self._conflict_engine = ConflictEngine(seed)
        self._humor_engine = HumorEngine(seed, self._creativity_config.get("humor", 0.5))
        self._surprise_engine = SurpriseEngine(seed, self._creativity_config.get("surprise", 0.6))
    
    @classmethod
    def from_config(cls, config: PipelineConfig, seed: int = 0) -> "CreativePlanner":
        """Create planner from pipeline config."""
        return cls(
            seed=seed,
            creativity_config=config.creativity_config,
            creative_budget=config.creative_budget,
            genre=config.genre,
            tone=config.tone,
        )
    
    def create_creative_plan(
        self,
        world_state: WorldState,
        observations: list[VisualObservations],
        ranked_evidence: list[RetrievedEvidence],
        collection_memory: "CollectionMemory | None" = None,
        retrieval_result: "RetrievalResult | None" = None,
    ) -> CreativePlan:
        """Create a comprehensive creative plan for the story.
        
        Args:
            world_state: Current world state
            observations: Visual observations
            ranked_evidence: Ranked evidence for grounding
            collection_memory: Optional V2.3 collection memory for enhanced planning
            retrieval_result: Optional V2.3 hierarchical retrieval result
        """
        
        # Extract hard facts and soft inferences from evidence
        hard_facts = self._extract_hard_facts(ranked_evidence)
        soft_inferences = self._extract_soft_inferences(ranked_evidence)
        locked_facts = self._extract_locked_facts(ranked_evidence)
        
        # V2.3: Enhance with collection memory if available
        if collection_memory:
            # Add recurring entities to hard facts
            for entity_label, entity_mem in collection_memory.global_entities.items():
                if len(entity_mem.scene_ids) > 1:
                    hard_facts.append(f"Recurring {entity_mem.entity_type}: {entity_label} (scenes: {len(entity_mem.scene_ids)})")
            
            # Add state transitions to soft inferences
            for transition in collection_memory.state_transitions:
                if transition.transition_type in ("disappeared", "reappeared", "moved"):
                    soft_inferences.append(f"{transition.entity_label} {transition.transition_type}: {transition.description}")
        
        # Generate character profiles
        characters = self._character_system.generate_profiles(
            world_state, self._genre, self._tone
        )
        
        # Generate central conflict (grounded in evidence)
        evidence_summary = self._summarize_evidence(ranked_evidence)
        conflict = self._conflict_engine.generate_conflict(
            world_state, evidence_summary, self._genre.value
        )
        
        # Generate humor moments (grounded in evidence)
        humor_moments = self._humor_engine.generate_humor_moments(
            world_state, conflict, HumorStyle.SITUATIONAL, count=2
        )
        
        # Generate surprise/twist (grounded in evidence)
        foreshadowing = self._identify_foreshadowing_opportunities(observations, world_state)
        surprise = self._surprise_engine.generate_surprise(
            world_state, evidence_summary, foreshadowing
        )
        
        # Identify open loops
        open_loops = [loop.get("description", "") for loop in world_state.open_loops]
        
        # V2.3: Plan callbacks using collection memory if available
        if collection_memory and retrieval_result:
            callback_plan = self._plan_callbacks_v23(collection_memory, retrieval_result)
        else:
            callback_plan = self._plan_callbacks(observations, world_state)
        
        # Design narrative arc
        narrative_arc = self._design_narrative_arc(conflict, surprise, open_loops)
        
        return CreativePlan(
            genre=self._genre.value,
            tone=self._tone.value,
            creativity_level=self._creativity_config.get("creativity", 0.7),
            surprise_level=self._creativity_config.get("surprise", 0.6),
            humor_level=self._creativity_config.get("humor", 0.5),
            mystery_level=self._creativity_config.get("mystery", 0.4),
            emotion_level=self._creativity_config.get("emotion", 0.5),
            dialogue_level=self._creativity_config.get("dialogue", 0.4),
            metaphor_level=self._creativity_config.get("metaphor", 0.3),
            characters=characters,
            central_conflict=conflict,
            open_loops=open_loops,
            foreshadowing_elements=foreshadowing,
            callback_plan=callback_plan,
            narrative_arc=narrative_arc,
            # New fields for V2.1
            hard_facts=hard_facts,
            soft_inferences=soft_inferences,
            locked_facts=locked_facts,
            creative_budget=self._creative_budget,
        )
    
    def _extract_hard_facts(self, ranked_evidence: list[RetrievedEvidence]) -> list[str]:
        """Extract hard visual facts from ranked evidence."""
        facts = []
        for item in ranked_evidence:
            ev = item.record
            if ev.information_class.value == "hard_fact":
                if ev.type.value in ["person", "character"]:
                    facts.append(f"Character: {ev.entity}")
                elif ev.type.value == "object":
                    facts.append(f"Object: {ev.entity}")
                elif ev.type.value == "action":
                    facts.append(f"Action: {ev.entity}")
                elif ev.type.value == "spatial_fact":
                    facts.append(f"Spatial: {ev.entity}")
                elif ev.type.value == "ocr":
                    facts.append(f"Text: {ev.entity}")
        return list(dict.fromkeys(facts))  # deduplicate preserving order
    
    def _extract_soft_inferences(self, ranked_evidence: list[RetrievedEvidence]) -> list[str]:
        """Extract soft inferences from ranked evidence."""
        inferences = []
        for item in ranked_evidence:
            ev = item.record
            if ev.information_class.value == "soft_inference":
                if ev.type.value == "action":
                    inferences.append(f"Possibly {ev.entity} (confidence: {ev.confidence:.2f})")
                elif ev.type.value == "relationship":
                    inferences.append(f"Possible relationship: {ev.relationship} involving {ev.entity}")
        return list(dict.fromkeys(inferences))
    
    def _extract_locked_facts(self, ranked_evidence: list[RetrievedEvidence]) -> list[str]:
        """Extract locked visual facts that must not be contradicted."""
        facts = []
        for item in ranked_evidence:
            ev = item.record
            if ev.information_class.value == "hard_fact":
                facts.append(f"{ev.entity}")
        return list(dict.fromkeys(facts))
    
    def _summarize_evidence(self, ranked_evidence: list[RetrievedEvidence]) -> str:
        parts = []
        for item in ranked_evidence[:10]:
            ev = item.record
            parts.append(f"{ev.entity} ({ev.type.value}): {ev.evidence_text[:80]}")
        return "; ".join(parts)
    
    def create_story_plan(
        self,
        creative_plan: CreativePlan,
        target_words: int = 250,
    ) -> StoryPlan:
        """Create detailed story plan with beats."""
        
        beats = []
        words_per_beat = target_words // 7
        
        # Beat 1: Setup
        beats.append(StoryBeat(
            beat_number=1,
            beat_type=BeatType.SETUP.value,
            description="Establish setting, introduce main character(s), show normal world",
            key_entities=[c.get("label", "") for c in creative_plan.characters[:2]],
            target_words=words_per_beat,
        ))
        
        # Beat 2: Goal
        beats.append(StoryBeat(
            beat_number=2,
            beat_type=BeatType.GOAL.value,
            description=f"Character wants: {creative_plan.characters[0].get('goal', 'something')} if creative_plan.characters else 'a goal'",
            key_entities=[c.get("label", "") for c in creative_plan.characters[:1]],
            creative_elements=[creative_plan.characters[0].get("quirk", "")] if creative_plan.characters else [],
            target_words=words_per_beat,
        ))
        
        # Beat 3: Conflict
        beats.append(StoryBeat(
            beat_number=3,
            beat_type=BeatType.CONFLICT.value,
            description=creative_plan.central_conflict.get("description", "Conflict arises"),
            key_entities=creative_plan.central_conflict.get("involved_entities", []),
            target_words=words_per_beat,
        ))
        
        # Beat 4: Escalation
        beats.append(StoryBeat(
            beat_number=4,
            beat_type=BeatType.ESCALATION.value,
            description="Conflict intensifies, stakes rise, character tries and fails",
            key_entities=[c.get("label", "") for c in creative_plan.characters[:2]],
            creative_elements=["rising tension", "failed attempt"],
            target_words=words_per_beat,
        ))
        
        # Beat 5: Surprise/Twist
        surprise_desc = creative_plan.narrative_arc[4].get("description", "Unexpected development") if len(creative_plan.narrative_arc) > 4 else "A surprise changes everything"
        beats.append(StoryBeat(
            beat_number=5,
            beat_type=BeatType.SURPRISE.value,
            description=surprise_desc,
            key_entities=[c.get("label", "") for c in creative_plan.characters[:2]],
            creative_elements=["recontextualization", "hidden significance revealed"],
            target_words=words_per_beat,
        ))
        
        # Beat 6: Resolution
        beats.append(StoryBeat(
            beat_number=6,
            beat_type=BeatType.RESOLUTION.value,
            description="Conflict resolves, character achieves or fails goal with growth",
            key_entities=[c.get("label", "") for c in creative_plan.characters[:2]],
            target_words=words_per_beat,
        ))
        
        # Beat 7: Callback/Payoff
        callback_desc = "A detail from earlier returns with new meaning"
        if creative_plan.callback_plan:
            callback_desc = creative_plan.callback_plan[0].get("description", callback_desc)
        
        beats.append(StoryBeat(
            beat_number=7,
            beat_type=BeatType.CALLBACK.value,
            description=callback_desc,
            key_entities=[c.get("label", "") for c in creative_plan.characters[:1]],
            creative_elements=["foreshadowing payoff", "emotional resonance"],
            target_words=words_per_beat,
        ))
        
        return StoryPlan(
            beats=beats,
            creative_plan=creative_plan,
            target_total_words=target_words,
        )
    
    def _identify_foreshadowing_opportunities(
        self,
        observations: list[VisualObservations],
        world_state: WorldState,
    ) -> list[dict[str, Any]]:
        """Identify details that could be used for foreshadowing."""
        opportunities = []
        
        for obs in observations:
            for region_desc in obs.region_descriptions[:2]:
                opportunities.append({
                    "detail": region_desc[:80],
                    "frame": obs.frame_id,
                    "type": "visual_detail",
                })
            
            for obj in obs.od_labels[:3]:
                opportunities.append({
                    "detail": obj,
                    "frame": obs.frame_id,
                    "type": "object",
                })
        
        # Also use world_state open_loops and recurring entities
        for entity in world_state.get_all_entities():
            if entity.disappearance_frame is not None:
                opportunities.append({
                    "detail": f"{entity.label} disappearance",
                    "frame": entity.disappearance_frame,
                    "type": "disappearance",
                })
            if entity.is_recurring:
                opportunities.append({
                    "detail": f"recurring {entity.entity_type}: {entity.label}",
                    "frame": entity.frames_present[-1] if entity.frames_present else 0,
                    "type": "recurring_entity",
                })
        
        # Also check world_state.open_loops for foreshadowing opportunities
        for loop in world_state.open_loops:
            if isinstance(loop, dict) and "description" in loop:
                opportunities.append({
                    "detail": loop["description"],
                    "frame": loop.get("last_frame", 0),
                    "type": "open_loop",
                })
        
        return opportunities[:8]
    
    def _plan_callbacks(
        self,
        observations: list[VisualObservations],
        world_state: WorldState,
    ) -> list[dict[str, Any]]:
        """Plan callbacks to recurring elements."""
        callbacks = []
        
        recurring = [e for e in world_state.get_all_entities() if e.is_recurring]
        for entity in recurring[:3]:
            callbacks.append({
                "element": entity.label,
                "type": entity.entity_type,
                "description": f"Recurring {entity.entity_type}: {entity.label} appears in frames {entity.frames_present}",
                "payoff_frame": entity.frames_present[-1] if entity.frames_present else 0,
            })
        
        return callbacks
    
    def _plan_callbacks_v23(
        self,
        collection_memory: "CollectionMemory",
        retrieval_result: "RetrievalResult",
    ) -> list[dict[str, Any]]:
        """Plan callbacks using V2.3 collection memory and hierarchical retrieval."""
        callbacks = []
        
        # Use narrative elements from retrieval result for callbacks
        for elem in retrieval_result.narrative_elements[:3]:
            if elem.potential_callback or elem.element_type == "open_loop":
                callbacks.append({
                    "element": elem.label,
                    "type": elem.element_type,
                    "description": elem.description,
                    "payoff_scene": elem.callback_payoff_scene,
                    "importance": elem.narrative_importance,
                })
        
        # Also use recurring entities from collection memory
        for entity_label, entity_mem in collection_memory.global_entities.items():
            if len(entity_mem.scene_ids) > 1 and len(callbacks) < 5:
                callbacks.append({
                    "element": entity_label,
                    "type": f"recurring_{entity_mem.entity_type}",
                    "description": f"Recurring {entity_mem.entity_type}: {entity_label} appears in {len(entity_mem.scene_ids)} scenes",
                    "payoff_scene": entity_mem.last_seen_scene,
                    "importance": 0.7,
                })
        
        return callbacks
    
    def _design_narrative_arc(
        self,
        conflict: dict[str, Any],
        surprise: dict[str, Any] | None,
        open_loops: list[str],
    ) -> list[dict[str, Any]]:
        """Design the narrative arc structure."""
        arc = [
            {"beat": "setup", "description": "Establish world and character"},
            {"beat": "inciting", "description": "Goal becomes clear"},
            {"beat": "conflict", "description": conflict.get("description", "Conflict emerges")},
            {"beat": "escalation", "description": "Stakes rise, attempts fail"},
            {"beat": "surprise", "description": surprise.get("description", "Twist/revelation") if surprise else "Unexpected turn"},
            {"beat": "resolution", "description": "Conflict resolves"},
            {"beat": "payoff", "description": "Callbacks and emotional closure"},
        ]
        return arc
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng.seed(seed)
        self._character_system.set_seed(seed)
        self._conflict_engine.set_seed(seed)
        self._humor_engine.set_seed(seed)
        self._surprise_engine.set_seed(seed)
    
    def update_config(self, creativity_config: dict[str, float]) -> None:
        self._creativity_config.update(creativity_config)
        self._humor_engine.set_humor_level(self._creativity_config.get("humor", 0.5))
        self._surprise_engine.set_surprise_level(self._creativity_config.get("surprise", 0.6))