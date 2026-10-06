"""Evidence ranking for relevance and reliability."""
from typing import Any, Callable
from dataclasses import dataclass, field

from ..domain.schemas import RetrievedEvidence, EvidenceRecord, EvidenceConfidence, InformationClass
from ..domain.enums import EvidenceType


@dataclass
class RankingWeights:
    """Configurable weights for evidence ranking."""
    semantic_similarity: float = 0.45
    confidence: float = 0.30
    recurrence: float = 0.15
    recency: float = 0.10
    provenance_quality: float = 0.0
    token_budget: float = 0.0
    
    def normalize(self) -> "RankingWeights":
        total = (self.semantic_similarity + self.confidence + self.recurrence + 
                self.recency + self.provenance_quality + self.token_budget)
        if total == 0:
            return RankingWeights()
        return RankingWeights(
            semantic_similarity=self.semantic_similarity / total,
            confidence=self.confidence / total,
            recurrence=self.recurrence / total,
            recency=self.recency / total,
            provenance_quality=self.provenance_quality / total,
            token_budget=self.token_budget / total,
        )
    
    def to_dict(self) -> dict[str, float]:
        return {
            "semantic_similarity": self.semantic_similarity,
            "confidence": self.confidence,
            "recurrence": self.recurrence,
            "recency": self.recency,
            "provenance_quality": self.provenance_quality,
            "token_budget": self.token_budget,
        }


DEFAULT_WEIGHTS = RankingWeights().normalize()


class EvidenceRanker:
    """Rank retrieved evidence by relevance and reliability."""
    
    def __init__(
        self,
        weights: RankingWeights | None = None,
        max_context_tokens: int = 2000,
        token_estimator: Callable[[str], int] | None = None,
    ):
        self._weights = weights or DEFAULT_WEIGHTS
        self._max_context_tokens = max_context_tokens
        self._token_estimator = token_estimator or (lambda t: len(t.split()) * 1.3)
    
    @property
    def weights(self) -> RankingWeights:
        return self._weights
    
    def set_weights(self, weights: RankingWeights) -> None:
        self._weights = weights.normalize()
    
    def rank(
        self,
        retrieved: list[RetrievedEvidence],
        query: str = "",
        current_frame: int = 0,
        entity_recurrence: dict[str, int] | None = None,
    ) -> list[RetrievedEvidence]:
        """Rank evidence by combined score."""
        entity_recurrence = entity_recurrence or {}
        
        for item in retrieved:
            score = self._compute_score(item, query, current_frame, entity_recurrence)
            item.rank_score = score
            item.rank_factors = self._compute_factors(item, query, current_frame, entity_recurrence)
        
        ranked = sorted(retrieved, key=lambda x: x.rank_score, reverse=True)
        return ranked
    
    def _compute_score(
        self,
        item: RetrievedEvidence,
        query: str,
        current_frame: int,
        entity_recurrence: dict[str, int],
    ) -> float:
        w = self._weights
        evidence = item.record
        
        semantic = item.semantic_similarity
        
        conf_score = evidence.confidence
        
        recurrence_count = entity_recurrence.get(evidence.entity, 0)
        recurrence_score = min(recurrence_count / 5.0, 1.0)
        
        frame_diff = current_frame - evidence.frame_id
        recency_score = max(0.0, 1.0 - frame_diff / 10.0) if frame_diff >= 0 else 0.5
        
        provenance_score = self._provenance_quality(evidence)
        
        token_score = 1.0
        
        score = (
            w.semantic_similarity * semantic +
            w.confidence * conf_score +
            w.recurrence * recurrence_score +
            w.recency * recency_score +
            w.provenance_quality * provenance_score +
            w.token_budget * token_score
        )
        
        if evidence.information_class == InformationClass.HARD_FACT:
            score *= 1.1
        elif evidence.information_class == InformationClass.SOFT_INFERENCE:
            score *= 1.0
        else:
            score *= 0.8
        
        return min(score, 1.0)
    
    def _compute_factors(
        self,
        item: RetrievedEvidence,
        query: str,
        current_frame: int,
        entity_recurrence: dict[str, int],
    ) -> dict[str, float]:
        evidence = item.record
        
        recurrence_count = entity_recurrence.get(evidence.entity, 0)
        frame_diff = current_frame - evidence.frame_id
        
        return {
            "semantic_similarity": item.semantic_similarity,
            "confidence": evidence.confidence,
            "recurrence": min(recurrence_count / 5.0, 1.0),
            "recency": max(0.0, 1.0 - frame_diff / 10.0) if frame_diff >= 0 else 0.5,
            "provenance_quality": self._provenance_quality(evidence),
            "information_class_bonus": 1.1 if evidence.information_class == InformationClass.HARD_FACT else 1.0,
        }
    
    def _provenance_quality(self, evidence: EvidenceRecord) -> float:
        source_scores = {
            "grounding_dino": 1.0,
            "florence2": 0.9,
            "ocr": 0.85,
        }
        return source_scores.get(evidence.source.value, 0.5)
    
    def select_for_context(
        self,
        ranked: list[RetrievedEvidence],
        max_tokens: int | None = None,
    ) -> list[RetrievedEvidence]:
        """Select top evidence that fits within token budget."""
        max_tokens = max_tokens or self._max_context_tokens
        selected = []
        token_count = 0
        
        for item in ranked:
            evidence_text = self._evidence_to_text(item.record)
            tokens = int(self._token_estimator(evidence_text))
            
            if token_count + tokens > max_tokens and selected:
                break
            
            selected.append(item)
            token_count += tokens
        
        return selected
    
    def _evidence_to_text(self, evidence: EvidenceRecord) -> str:
        parts = []
        if evidence.entity:
            parts.append(f"{evidence.entity}")
        if evidence.type:
            parts.append(f"[{evidence.type.value}]")
        if evidence.action:
            parts.append(f"action:{evidence.action}")
        if evidence.relationship:
            parts.append(f"rel:{evidence.relationship}")
        if evidence.evidence_text:
            parts.append(evidence.evidence_text[:100])
        if evidence.confidence_class:
            parts.append(f"conf:{evidence.confidence_class.value}")
        return " ".join(parts)
    
    def rank_for_story_planning(
        self,
        retrieved: list[RetrievedEvidence],
        world_state_summary: str,
        current_frame: int,
        entity_recurrence: dict[str, int],
    ) -> list[RetrievedEvidence]:
        """Specialized ranking for story planning phase."""
        ranked = self.rank(retrieved, world_state_summary, current_frame, entity_recurrence)
        
        for item in ranked:
            if item.record.type in [EvidenceType.PERSON, EvidenceType.ACTION, EvidenceType.EVENT]:
                item.rank_score *= 1.15
            if item.record.information_class == InformationClass.CREATIVE_SPACE:
                item.rank_score *= 0.7
        
        return sorted(ranked, key=lambda x: x.rank_score, reverse=True)