"""Regression tests for Grounded Creativity V2.1 features."""
import pytest
from image_story.domain.schemas import (
    StoryClaim, VerificationResult, CreativePlan, PipelineConfig,
    EvidenceRecord, EvidenceType, SourceModel, InformationClass
)
from image_story.domain.enums import ClaimStatus, ClaimClassification
from image_story.evaluation.claims import ClaimExtractor, ClaimVerifier


class TestClaimClassification:
    """Tests for claim classification (OBSERVED/INFERRED/CREATIVE)."""
    
    def setup_method(self):
        self.extractor = ClaimExtractor()
    
    def test_creative_claim_classification(self):
        """Creative claims should be classified as 'creative'."""
        claim = StoryClaim(
            subject="he",
            relation="secretly planned",
            object="this adventure",
            original_sentence="He secretly planned this adventure.",
        )
        classification = self.extractor._classify_claim_classification(claim, "")
        assert classification == "creative"
    
    def test_observed_claim_classification(self):
        """Visually observable claims should be 'observed'."""
        claim = StoryClaim(
            subject="woman",
            relation="holds",
            object="red bag",
            original_sentence="The woman holds a red bag.",
        )
        classification = self.extractor._classify_claim_classification(claim, "")
        assert classification == "observed"
    
    def test_inferred_claim_classification(self):
        """Inferred claims should be 'inferred'."""
        claim = StoryClaim(
            subject="she",
            relation="appeared to be",
            object="waiting",
            original_sentence="She appeared to be waiting for someone.",
        )
        classification = self.extractor._classify_claim_classification(claim, "")
        assert classification == "inferred"
    
    def test_metaphor_claim_classification(self):
        """Metaphorical claims should be 'creative'."""
        claim = StoryClaim(
            subject="the cabinet",
            relation="stood as",
            object="a sentinel metaphor",
            original_sentence="The cabinet stood as a sentinel metaphor for his loneliness.",
        )
        classification = self.extractor._classify_claim_classification(claim, "")
        assert classification == "creative"
    
    def test_motivation_claim_classification(self):
        """Internal motivation claims should be 'creative'."""
        claim = StoryClaim(
            subject="she",
            relation="wanted",
            object="to escape",
            original_sentence="She wanted to escape her past.",
        )
        classification = self.extractor._classify_claim_classification(claim, "")
        assert classification == "creative"


class TestClaimVerificationWithClassification:
    """Tests for claim verification with classification awareness."""
    
    def test_creative_claims_not_penalized(self):
        """Creative claims should not count against grounding score."""
        claim = StoryClaim(
            id="claim_1",
            subject="he",
            relation="secretly wanted",
            object="to escape",
            original_sentence="He secretly wanted to escape.",
            claim_classification="creative",
        )
        
        vr = VerificationResult(
            claim_id=claim.id,
            claim=claim,
            status=ClaimStatus.UNSUPPORTED.value,
            confidence=0.5,
        )
        
        from image_story.evaluation.claims import ClaimVerifier
        verifier = ClaimVerifier()
        
        # Should not penalize creative claims
        score = verifier.compute_claim_grounding_score([vr])
        assert score == 1.0  # creative claims excluded from grounding
    
    def test_observed_claims_require_support(self):
        """Observed claims must be supported for good grounding."""
        claim = StoryClaim(
            id="claim_1",
            subject="woman",
            relation="holds",
            object="red bag",
            original_sentence="The woman holds a red bag.",
            claim_classification="observed",
        )
        
        vr = VerificationResult(
            claim_id=claim.id,
            claim=claim,
            status=ClaimStatus.UNSUPPORTED.value,
            confidence=0.5,
        )
        
        from image_story.evaluation.claims import ClaimVerifier
        verifier = ClaimVerifier()
        
        score = verifier.compute_claim_grounding_score([vr])
        assert score < 1.0  # Unsupported observed claim hurts grounding
    
    def test_creative_claim_allowed(self):
        """Creative claims should be allowed without visual evidence."""
        claim = StoryClaim(
            id="claim_1",
            subject="he",
            relation="dreamed of",
            object="flying",
            original_sentence="He dreamed of flying.",
            claim_classification="creative",
        )
        
        vr = VerificationResult(
            claim_id=claim.id,
            claim=claim,
            status=ClaimStatus.UNSUPPORTED.value,
            confidence=0.5,
        )
        
        from image_story.evaluation.claims import ClaimVerifier
        verifier = ClaimVerifier()
        
        score = verifier.compute_claim_grounding_score([vr])
        assert score == 1.0  # Creative claims don't need visual support


class TestCreativePlanWithBudget:
    """Tests for CreativePlan with creative budget."""
    
    def test_creative_plan_has_budget(self):
        from image_story.domain.schemas import CreativePlan
        plan = CreativePlan(
            genre="whimsical",
            tone="comedic",
            creative_budget={
                "max_creative_claims": 6,
                "max_visual_inventions": 0,
                "max_soft_inferences": 3,
                "max_new_named_entities": 0,
            },
        )
        
        assert plan.creative_budget["max_creative_claims"] == 6
        assert plan.creative_budget["max_visual_inventions"] == 0
    
    def test_creative_plan_has_locked_facts(self):
        from image_story.domain.schemas import CreativePlan
        plan = CreativePlan(
            genre="whimsical",
            tone="comedic",
            locked_facts=["woman", "red bag", "street"],
        )
        
        assert "woman" in plan.locked_facts
        assert "red bag" in plan.locked_facts


class TestLockedFactsInPrompt:
    """Tests that locked facts appear in the story prompt."""
    
    def test_locked_facts_in_prompt(self):
        from image_story.context.builder import ContextBuilder
        from image_story.domain.schemas import CreativePlan
        
        builder = ContextBuilder()
        creative_plan = CreativePlan(
            genre="whimsical",
            tone="comedic",
            locked_facts=["woman", "red bag", "street"],
        )
        
        prompt = builder.build_story_prompt(
            context="Test context",
            creative_plan=creative_plan,
            target_words=100,
        )
        
        assert "LOCKED VISUAL FACTS" in prompt
        assert "woman" in prompt
        assert "red bag" in prompt
        assert "street" in prompt
        assert "MUST NOT CONTRADICT" in prompt


class TestSurpriseGroundedInEvidence:
    """Tests that surprise engine uses actual evidence."""
    
    def test_surprise_uses_retrieved_evidence(self):
        from image_story.narrative.surprise import SurpriseEngine
        from image_story.domain.schemas import WorldState, WorldEntity, RetrievedEvidence, EvidenceRecord, EvidenceType, SourceModel, InformationClass
        
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="red backpack", entity_type="object", first_frame=0, last_frame=0),
        ]
        
        evidence = [
            EvidenceRecord(entity="red backpack", type=EvidenceType.OBJECT, frame_id=0, confidence=0.9,
                          source=SourceModel.GROUNDING_DINO, information_class=InformationClass.HARD_FACT),
        ]
        retrieved = [type('obj', (object,), {'record': e}) for e in evidence]
        
        engine = SurpriseEngine(seed=42, surprise_level=1.0)
        surprise = engine.generate_surprise(
            world_state, "girl backpack street", [], retrieved_evidence=retrieved
        )
        
        # Should generate a grounded surprise
        if surprise:
            assert "grounded_entities" in surprise
            assert "red backpack" in surprise.get("grounded_entities", [])


class TestHumorGroundedInEvidence:
    """Tests that humor engine uses actual evidence."""
    
    def test_humor_uses_evidence(self):
        from image_story.narrative.humor import HumorEngine
        from image_story.domain.schemas import WorldState, WorldEntity, RetrievedEvidence, EvidenceRecord, EvidenceType, SourceModel, InformationClass
        from image_story.domain.enums import HumorStyle
        
        world_state = WorldState()
        world_state.characters = [
            WorldEntity(id="c1", label="girl", entity_type="character", first_frame=0, last_frame=0),
        ]
        world_state.objects = [
            WorldEntity(id="o1", label="statue", entity_type="object", first_frame=0, last_frame=0),
        ]
        
        evidence = [
            EvidenceRecord(entity="statue", type=EvidenceType.OBJECT, frame_id=0, confidence=0.9,
                          source=SourceModel.GROUNDING_DINO, information_class=InformationClass.HARD_FACT),
        ]
        retrieved = [type('obj', (object,), {'record': e}) for e in evidence]
        
        engine = HumorEngine(seed=42, humor_level=1.0)
        moments = engine.generate_humor_moments(
            world_state, {}, HumorStyle.SITUATIONAL, count=2, retrieved_evidence=retrieved
        )
        
        # Should generate grounded humor
        assert len(moments) > 0
        assert any("statue" in m for m in moments)


class TestCreativeQualityMetrics:
    """Tests for creative quality metrics."""
    
    def test_creative_quality_proxy(self):
        from image_story.evaluation.narrative import NarrativeQualityEvaluator
        from image_story.domain.schemas import StoryDraft, VerificationResult, StoryClaim
        from image_story.domain.enums import ClaimStatus
        
        story = StoryDraft(text="A creative story with personality and humor.")
        
        claim = StoryClaim(
            id="c1", subject="he", relation="secretly wanted", object="to escape",
            original_sentence="He secretly wanted to escape.", claim_classification="creative"
        )
        vr = VerificationResult(claim_id="c1", claim=claim, status=ClaimStatus.UNSUPPORTED.value, confidence=0.5)
        
        evaluator = NarrativeQualityEvaluator()
        quality = evaluator._assess_creative_quality(None, None, [vr])
        
        assert quality > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])