"""Independent visual verification for claim checking."""
import torch
from PIL import Image
from typing import Any

from ..domain.schemas import EvidenceRecord, VerificationResult, StoryClaim, BoundingBox
from ..domain.enums import ClaimStatus
from .florence import Florence2Model
from .grounding import GroundingDINOModel


class VisualVerifier:
    """Independent visual verification of claims against image evidence.
    
    Uses separate model instances from the main perception pipeline
    to avoid circular validation.
    """
    
    def __init__(
        self,
        florence_model: Florence2Model | None = None,
        grounding_model: GroundingDINOModel | None = None,
        device: str = "cpu",
    ):
        self._florence = florence_model
        self._grounding = grounding_model
        self._device = device
        self._clip_model = None
        self._clip_processor = None
    
    def _load_clip(self) -> None:
        if self._clip_model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model.to(self._device)
    
    def verify_claim(
        self,
        claim: StoryClaim,
        image: Image.Image,
        evidence_records: list[EvidenceRecord],
    ) -> VerificationResult:
        """Verify a claim against visual evidence.
        
        Uses multiple verification strategies:
        1. Direct evidence matching (exact entity/action match)
        2. GroundingDINO re-detection for specific entities
        3. CLIP similarity for semantic verification
        """
        supporting = []
        contradicting = []
        
        # Strategy 1: Direct evidence matching
        for evidence in evidence_records:
            if self._evidence_supports_claim(evidence, claim):
                supporting.append(evidence)
            elif self._evidence_contradicts_claim(evidence, claim):
                contradicting.append(evidence)
        
        # Strategy 2: GroundingDINO re-detection for key entities
        if self._grounding and self._grounding.enabled and self._grounding.is_loaded:
            grounding_evidence = self._verify_with_grounding(claim, image)
            for ev in grounding_evidence:
                if self._evidence_supports_claim(ev, claim):
                    supporting.append(ev)
                elif self._evidence_contradicts_claim(ev, claim):
                    contradicting.append(ev)
        
        # Strategy 3: CLIP similarity
        clip_score = self._verify_with_clip(claim, image)
        
        # Determine status
        if contradicting:
            status = ClaimStatus.CONTRADICTED
            confidence = 0.9
        elif supporting:
            status = ClaimStatus.SUPPORTED
            confidence = min(0.7 + clip_score * 0.3, 1.0)
        else:
            status = ClaimStatus.UNSUPPORTED
            confidence = 1.0 - clip_score * 0.5
        
        notes = f"CLIP similarity: {clip_score:.3f}"
        if supporting:
            notes += f"; {len(supporting)} supporting evidence"
        if contradicting:
            notes += f"; {len(contradicting)} contradicting evidence"
        
        return VerificationResult(
            claim_id=claim.id,
            claim=claim,
            status=status.value,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            confidence=confidence,
            notes=notes,
        )
    
    def _evidence_supports_claim(self, evidence: EvidenceRecord, claim: StoryClaim) -> bool:
        """Check if evidence directly supports the claim."""
        claim_terms = {
            claim.subject.lower(),
            claim.relation.lower(),
            claim.object.lower(),
        }
        evidence_text = evidence.evidence_text.lower()
        evidence_entity = evidence.entity.lower()
        
        # Check if claim entities appear in evidence
        matches = sum(1 for term in claim_terms if term and term in evidence_text)
        matches += sum(1 for term in claim_terms if term and term in evidence_entity)
        
        return matches >= 2
    
    def _evidence_contradicts_claim(self, evidence: EvidenceRecord, claim: StoryClaim) -> bool:
        """Check if evidence contradicts the claim (simplified heuristic)."""
        # Check for color/material conflicts
        claim_text = claim.to_natural_language().lower()
        evidence_text = evidence.evidence_text.lower()
        
        # Simple contradiction: if evidence says "red" and claim says "blue" for same object
        color_words = {"red", "blue", "green", "yellow", "black", "white", "brown", "orange", "purple", "pink", "grey", "gray"}
        material_words = {"wood", "metal", "plastic", "glass", "stone", "leather", "fabric", "ceramic"}
        
        claim_colors = color_words & set(claim_text.split())
        evidence_colors = color_words & set(evidence_text.split())
        
        if claim_colors and evidence_colors and claim_colors != evidence_colors:
            # Check if they refer to the same entity
            claim_entities = {claim.subject.lower(), claim.object.lower()}
            evidence_entities = {evidence.entity.lower()}
            if claim_entities & evidence_entities:
                return True
        
        return False
    
    def _verify_with_grounding(self, claim: StoryClaim, image: Image.Image) -> list[EvidenceRecord]:
        """Re-detect key entities using GroundingDINO."""
        phrases = [claim.subject, claim.object]
        phrases = [p for p in phrases if p]
        if not phrases:
            return []
        
        try:
            return self._grounding.detect_with_phrases(image, phrases, frame_id=0)
        except Exception:
            return []
    
    def _verify_with_clip(self, claim: StoryClaim, image: Image.Image) -> float:
        """Use CLIP to compute image-claim similarity."""
        self._load_clip()
        
        claim_text = claim.to_natural_language()
        inputs = self._clip_processor(
            text=[claim_text], images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            similarity = logits_per_image[0][0].item()
        
        # Normalize to 0-1 range (CLIP cosine similarity is typically 0.1-0.3 for matching)
        normalized = max(0.0, min(1.0, (similarity - 0.15) / 0.15))
        return normalized
    
    def verify_multiple_claims(
        self,
        claims: list[StoryClaim],
        image: Image.Image,
        evidence_records: list[EvidenceRecord],
    ) -> list[VerificationResult]:
        """Verify multiple claims efficiently."""
        results = []
        for claim in claims:
            result = self.verify_claim(claim, image, evidence_records)
            results.append(result)
        return results