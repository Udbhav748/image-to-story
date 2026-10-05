"""Narrative quality evaluation metrics."""
import re
from typing import Any
from collections import Counter

from ..domain.schemas import StoryDraft, CreativePlan, EvaluationResult


class NarrativeQualityEvaluator:
    """Evaluate narrative quality aspects."""
    
    def __init__(self):
        pass
    
    def evaluate_narrative_quality(
        self,
        story: StoryDraft,
        creative_plan: CreativePlan | None = None,
        world_state: Any = None,
    ) -> dict[str, Any]:
        """Evaluate various narrative quality metrics."""
        
        metrics = {}
        
        # Character development
        metrics["character_mentions"] = self._count_character_mentions(story, creative_plan)
        metrics["character_depth"] = self._assess_character_depth(story, creative_plan)
        
        # Plot structure
        metrics["has_setup"] = self._check_setup(story)
        metrics["has_conflict"] = self._check_conflict(story)
        metrics["has_resolution"] = self._check_resolution(story)
        metrics["has_surprise"] = self._check_surprise(story)
        metrics["has_callback"] = self._check_callback(story, creative_plan)
        
        # Emotional arc
        metrics["emotional_progression"] = self._assess_emotional_arc(story)
        
        # Dialogue
        metrics["dialogue_present"] = self._check_dialogue(story)
        
        # Humor
        metrics["humor_attempts"] = self._count_humor_attempts(story)
        
        # Overall narrative quality score
        metrics["narrative_quality_score"] = self._compute_narrative_score(metrics)
        
        return metrics
    
    def _count_character_mentions(self, story: StoryDraft, creative_plan: CreativePlan | None) -> int:
        if not creative_plan or not creative_plan.characters:
            return 0
        
        count = 0
        story_lower = story.text.lower()
        for char in creative_plan.characters:
            label = char.get("label", "").lower()
            if label:
                count += story_lower.count(label)
        return count
    
    def _assess_character_depth(self, story: StoryDraft, creative_plan: CreativePlan | None) -> float:
        if not creative_plan or not creative_plan.characters:
            return 0.0
        
        depth_indicators = [
            "felt", "thought", "wanted", "hoped", "feared", "remembered",
            "personality", "quirk", "habit", "dream", "motivation", "goal",
            "realized", "understood", "decided", "chose",
        ]
        
        story_lower = story.text.lower()
        found = sum(1 for ind in depth_indicators if ind in story_lower)
        return min(found / 5.0, 1.0)
    
    def _check_setup(self, story: StoryDraft) -> bool:
        setup_indicators = ["once", "began", "started", "was", "were", "stood", "sat", "lived"]
        story_lower = story.text.lower()
        return any(ind in story_lower.split()[:20] for ind in setup_indicators)
    
    def _check_conflict(self, story: StoryDraft) -> bool:
        conflict_indicators = [
            "but", "however", "problem", "trouble", "difficult", "challenge",
            "conflict", "struggle", "fight", "argue", "disagree", "wrong",
            "missing", "lost", "gone", "disappeared", "blocked", "stopped",
        ]
        story_lower = story.text.lower()
        return any(ind in story_lower for ind in conflict_indicators)
    
    def _check_resolution(self, story: StoryDraft) -> bool:
        resolution_indicators = [
            "finally", "eventually", "resolved", "solved", "found", "realized",
            "understood", "worked out", "happy", "satisfied", "peace", "calm",
            "ended", "concluded", "finished",
        ]
        story_lower = story.text.lower()
        words = story_lower.split()
        last_third = words[-len(words)//3:] if len(words) > 10 else words
        return any(ind in " ".join(last_third) for ind in resolution_indicators)
    
    def _check_surprise(self, story: StoryDraft) -> bool:
        surprise_indicators = [
            "suddenly", "unexpected", "surprise", "shock", "realized", "discovered",
            "turned out", "actually", "secret", "hidden", "revealed", "twist",
            "not what", "different from", "contrary",
        ]
        story_lower = story.text.lower()
        return any(ind in story_lower for ind in surprise_indicators)
    
    def _check_callback(self, story: StoryDraft, creative_plan: CreativePlan | None) -> bool:
        if not creative_plan or not creative_plan.callback_plan:
            return False
        
        story_lower = story.text.lower()
        for callback in creative_plan.callback_plan:
            element = callback.get("element", "").lower()
            if element and element in story_lower:
                return True
        return False
    
    def _assess_emotional_arc(self, story: StoryDraft) -> float:
        emotion_words = {
            "positive": {"happy", "joy", "delight", "wonder", "excitement", "hope", "love", "warm", "peaceful", "content"},
            "negative": {"sad", "angry", "fear", "worry", "anxious", "frustrated", "disappointed", "lonely", "cold", "dark"},
            "transition": {"then", "but", "however", "suddenly", "realized", "changed", "shifted"},
        }
        
        story_lower = story.text.lower()
        words = story_lower.split()
        
        pos_count = sum(1 for w in words if w in emotion_words["positive"])
        neg_count = sum(1 for w in words if w in emotion_words["negative"])
        trans_count = sum(1 for w in words if w in emotion_words["transition"])
        
        total_emotion = pos_count + neg_count
        if total_emotion == 0:
            return 0.0
        
        # Good arc has both positive and negative with transitions
        arc_score = min((pos_count + neg_count) / 10.0, 1.0) * min(trans_count / 3.0, 1.0)
        return min(arc_score, 1.0)
    
    def _check_dialogue(self, story: StoryDraft) -> bool:
        return '"' in story.text or "'" in story.text
    
    def _count_humor_attempts(self, story: StoryDraft) -> int:
        humor_indicators = [
            "awkward", "ridiculous", "absurd", "funny", "laugh", "smile", "grin",
            "ironic", "sarcastic", "deadpan", "unexpectedly", "coincidence",
            "personification", "talked to", "judged", "opinion",
        ]
        story_lower = story.text.lower()
        return sum(1 for ind in humor_indicators if ind in story_lower)
    
    def evaluate_narrative_quality(
        self,
        story: StoryDraft,
        creative_plan: CreativePlan | None = None,
        world_state: Any = None,
        verification_results: list = None,
    ) -> dict[str, Any]:
        """Evaluate various narrative quality metrics."""
        
        metrics = {}
        
        # Character development
        metrics["character_mentions"] = self._count_character_mentions(story, creative_plan)
        metrics["character_depth"] = self._assess_character_depth(story, creative_plan)
        
        # Plot structure
        metrics["has_setup"] = self._check_setup(story)
        metrics["has_conflict"] = self._check_conflict(story)
        metrics["has_resolution"] = self._check_resolution(story)
        metrics["has_surprise"] = self._check_surprise(story)
        metrics["has_callback"] = self._check_callback(story, creative_plan)
        
        # Emotional arc
        metrics["emotional_progression"] = self._assess_emotional_arc(story)
        
        # Dialogue
        metrics["dialogue_present"] = self._check_dialogue(story)
        
        # Humor
        metrics["humor_attempts"] = self._count_humor_attempts(story)
        
        # V2.1: Creative quality metrics
        metrics["creative_claims"] = self._count_creative_claims(verification_results)
        metrics["observed_claims"] = self._count_observed_claims(verification_results)
        metrics["inferred_claims"] = self._count_inferred_claims(verification_results)
        metrics["creative_quality_proxy"] = self._assess_creative_quality(story, creative_plan, verification_results)
        
        # Overall narrative quality score
        metrics["narrative_quality_score"] = self._compute_narrative_score(metrics)
        
        return metrics
    
    def _count_creative_claims(self, verification_results: list) -> int:
        if not verification_results:
            return 0
        return sum(1 for r in verification_results if r.claim.claim_classification == "creative")
    
    def _count_observed_claims(self, verification_results: list) -> int:
        if not verification_results:
            return 0
        return sum(1 for r in verification_results if r.claim.claim_classification == "observed")
    
    def _count_inferred_claims(self, verification_results: list) -> int:
        if not verification_results:
            return 0
        return sum(1 for r in verification_results if r.claim.claim_classification == "inferred")
    
    def _assess_creative_quality(
        self, 
        story: StoryDraft, 
        creative_plan: CreativePlan | None,
        verification_results: list
    ) -> float:
        """Assess creative quality - are creative claims actually creative, not just unsupported?"""
        if not verification_results:
            return 0.0
        
        creative_count = sum(1 for r in verification_results if r.claim.claim_classification == "creative")
        observed_count = sum(1 for r in verification_results if r.claim.claim_classification == "observed")
        
        if creative_count == 0:
            return 0.0
        
        # Good creative quality = creative claims present + observed claims supported
        creative_ratio = creative_count / max(1, len(verification_results))
        # Bonus for having both creative and observed content
        balance_bonus = 0.2 if observed_count > 0 else 0.0
        
        return min(creative_ratio + balance_bonus, 1.0)
    
    def _compute_narrative_score(self, metrics: dict[str, Any]) -> float:
        score = 0.0
        weights = {
            "character_depth": 0.15,
            "has_setup": 0.08,
            "has_conflict": 0.15,
            "has_resolution": 0.12,
            "has_surprise": 0.12,
            "has_callback": 0.08,
            "emotional_progression": 0.08,
            "creative_quality_proxy": 0.18,
        }
        
        for key, weight in weights.items():
            val = metrics.get(key, 0)
            if isinstance(val, bool):
                score += weight * (1.0 if val else 0.0)
            else:
                score += weight * float(val)
        
        return round(score, 3)