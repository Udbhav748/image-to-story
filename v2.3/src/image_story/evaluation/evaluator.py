"""Main evaluator combining all evaluation components."""
import time
from typing import Any
from PIL import Image

from ..domain.schemas import (
    EvaluationResult,
    StoryDraft,
    VisualObservations,
    WorldState,
    CreativePlan,
    ExperimentManifest,
    PipelineArtifacts,
)
from ..domain.enums import EvaluationMetric
from .grounding import GroundingEvaluator, compute_grounding_score
from .continuity import ContinuityEvaluator
from .narrative import NarrativeQualityEvaluator
from .claims import ClaimExtractor, ClaimVerifier
from ..vision.verifier import VisualVerifier


class ComprehensiveEvaluator:
    """Comprehensive story evaluation combining all metrics."""
    
    def __init__(self, device: str = "cpu"):
        self._device = device
        self._grounding_evaluator = GroundingEvaluator(device)
        self._continuity_evaluator = ContinuityEvaluator()
        self._narrative_evaluator = NarrativeQualityEvaluator()
        self._claim_extractor = ClaimExtractor()
        self._visual_verifier = VisualVerifier(device=device)
        self._claim_verifier = ClaimVerifier(self._visual_verifier)
    
    def evaluate(
        self,
        artifacts: PipelineArtifacts,
        image: Image.Image | None = None,
    ) -> EvaluationResult:
        """Run comprehensive evaluation on pipeline artifacts."""
        
        t0 = time.perf_counter()
        
        story = artifacts.story_draft
        observations = artifacts.observations
        world_state = artifacts.world_state
        creative_plan = artifacts.creative_plan
        evidence_records = []
        for obs in observations:
            evidence_records.extend(obs.evidence_records)
        
        # Build caption/context for NLI
        caption = self._build_caption(observations)
        
        # Grounding evaluation (CLIP, NLI, attribute)
        grounding_result = self._grounding_evaluator.evaluate(image, caption, story) if image else None
        
        # Claim extraction and verification
        claims = self._claim_extractor.extract_claims(story, caption)
        verification_results = []
        claim_grounding = 0.0
        if claims:
            if image:
                verification_results = self._claim_verifier.verify_claims(claims, image, evidence_records)
            else:
                verification_results = self._claim_verifier._textual_verification_batch(claims, evidence_records)
            
            claim_grounding = self._claim_verifier.compute_claim_grounding_score(verification_results)
            artifacts.claims = claims
            artifacts.verification_results = verification_results
        
        # Continuity evaluation
        continuity_result = {}
        if len(observations) > 1:
            continuity_result = self._continuity_evaluator.evaluate_continuity(observations, story.text, world_state)
        
        # Narrative quality
        narrative_metrics = self._narrative_evaluator.evaluate_narrative_quality(
            story, creative_plan, world_state
        )
        
        # Aggregate results
        result = EvaluationResult()
        
        if grounding_result:
            result.grounding_score = grounding_result.grounding_score
            result.clip_image_story_mean = grounding_result.clip_image_story_mean
            result.nli_contra_mean = grounding_result.nli_contra_mean
            result.attribute_conflict = grounding_result.attribute_conflict
            result.repetition_rate = grounding_result.repetition_rate
            result.length_valid = grounding_result.length_valid
            result.grounding_pass = grounding_result.grounding_pass
            result.eval_clip_s = grounding_result.eval_clip_s
            result.eval_nli_s = grounding_result.eval_nli_s
            result.eval_rules_s = grounding_result.eval_rules_s
        
        result.claim_grounding_score = claim_grounding
        result.supported_claims = sum(1 for r in verification_results if r.status == "supported")
        result.unsupported_claims = sum(1 for r in verification_results if r.status == "unsupported")
        result.contradicted_claims = sum(1 for r in verification_results if r.status == "contradicted")
        total_claims = len(verification_results)
        result.claim_support_rate = result.supported_claims / total_claims if total_claims > 0 else 0.0
        
        result.continuity_score = continuity_result.get("entity_consistency") or 0.0
        result.narrative_coherence = narrative_metrics.get("narrative_quality_score", 0.0)
        result.contradiction_count = result.contradicted_claims
        
        result.eval_total_s = round(time.perf_counter() - t0, 3)
        
        return result
    
    def _build_caption(self, observations: list[VisualObservations]) -> str:
        """Build caption from observations for NLI."""
        parts = []
        for obs in observations:
            if obs.detailed_caption:
                parts.append(obs.detailed_caption)
            elif obs.scene:
                parts.append(obs.scene)
        return " ".join(parts) if parts else "An image."
    
    def evaluate_batch(
        self,
        artifacts_list: list[PipelineArtifacts],
        images: list[Image.Image] | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple pipeline runs."""
        results = []
        for i, artifacts in enumerate(artifacts_list):
            image = images[i] if images and i < len(images) else None
            result = self.evaluate(artifacts, image)
            results.append(result)
        return results
    
    def _textual_verification_batch(
        self,
        claims: list,
        evidence_records: list,
    ) -> list:
        """Batch textual verification fallback."""
        results = []
        for claim in claims:
            result = self._claim_verifier._textual_verification(claim, evidence_records)
            results.append(result)
        return results