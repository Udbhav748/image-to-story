"""Defects found by type checking during the reorganisation.

Each test corresponds to an `AttributeError`, `NameError` or wrong-signature bug
that existed before the reorganisation and that only mypy surfaced.
"""
from unittest.mock import MagicMock, patch

import pytest

from image_story.domain.enums import ClaimStatus, EvidenceType, InformationClass, SourceModel
from image_story.domain.schemas import (
    EvidenceRecord,
    ExperimentManifest,
    PipelineArtifacts,
    StoryClaim,
    StoryDraft,
    VerificationResult,
)
from image_story.evaluation.evaluator import ComprehensiveEvaluator
from image_story.verification.claims import ClaimExtractor, ClaimVerifier


def make_claim(**overrides) -> StoryClaim:
    defaults = {
        "subject": "balloon",
        "relation": "is",
        "object": "blue",
        "original_sentence": "The balloon is blue.",
        "claim_type": InformationClass.HARD_FACT,
        "claim_classification": "observed",
    }
    defaults.update(overrides)
    return StoryClaim(**defaults)


def make_evidence(entity: str, text: str, confidence: float = 0.9) -> EvidenceRecord:
    return EvidenceRecord(
        entity=entity,
        type=EvidenceType.OBJECT,
        frame_id=0,
        confidence=confidence,
        source=SourceModel.FLORENCE2,
        evidence_text=text,
        information_class=InformationClass.HARD_FACT,
    )


class TestTextualVerificationDetectsContradictions:
    """`_textual_verification` built an empty `contradicting` list and never
    added to it, so `ClaimStatus.CONTRADICTED` was unreachable on the textual
    path: contradictory claims were reported as merely unsupported."""

    def test_conflicting_colour_is_contradicted(self):
        verifier = ClaimVerifier()
        claim = make_claim()
        evidence = [make_evidence("balloon", "Detected red balloon")]
        result = verifier._textual_verification(claim, evidence)
        assert result.status == ClaimStatus.CONTRADICTED.value
        assert len(result.contradicting_evidence) == 1

    def test_matching_colour_is_supported(self):
        verifier = ClaimVerifier()
        claim = make_claim()
        evidence = [make_evidence("balloon", "Detected blue balloon")]
        result = verifier._textual_verification(claim, evidence)
        assert result.status == ClaimStatus.SUPPORTED.value
        assert result.contradicting_evidence == []

    def test_non_attribute_claim_about_the_entity_is_supported(self):
        verifier = ClaimVerifier()
        claim = make_claim(subject="girl", relation="holds", object="balloon")
        evidence = [make_evidence("girl", "Person holding balloon")]
        result = verifier._textual_verification(claim, evidence)
        assert result.status == ClaimStatus.SUPPORTED.value

    def test_unknown_entity_is_unsupported(self):
        verifier = ClaimVerifier()
        claim = make_claim()
        result = verifier._textual_verification(claim, [make_evidence("dog", "Detected brown dog")])
        assert result.status == ClaimStatus.UNSUPPORTED.value

    def test_contradiction_wins_over_support(self):
        verifier = ClaimVerifier()
        claim = make_claim()
        evidence = [
            make_evidence("balloon", "Detected blue balloon"),
            make_evidence("balloon", "Detected red balloon"),
        ]
        result = verifier._textual_verification(claim, evidence)
        assert result.status == ClaimStatus.CONTRADICTED.value


class TestTextualVerificationBatchExists:
    """`ComprehensiveEvaluator` called the private
    `ClaimVerifier._textual_verification_batch`, which does not exist: evaluating
    without an image raised AttributeError."""

    def test_batch_method_exists(self):
        results = ClaimVerifier().textual_verification_batch(
            [make_claim()], [make_evidence("balloon", "Detected blue balloon")]
        )
        assert len(results) == 1
        assert isinstance(results[0], VerificationResult)

    def test_evaluator_runs_without_an_image(self):
        """The end-to-end symptom: no image -> claim verification used to crash."""
        artifacts = PipelineArtifacts(run_id="test", manifest=ExperimentManifest())
        artifacts.story_draft = StoryDraft(text="The balloon is blue. The girl held it.")
        artifacts.observations = []

        evaluator = ComprehensiveEvaluator(device="cpu")
        with patch.object(ClaimExtractor, "extract_claims", return_value=[make_claim()]):
            result = evaluator.evaluate(artifacts, image=None)

        assert isinstance(result, object)
        assert hasattr(result, "claim_support_rate")

    def test_evaluator_returns_zeros_without_a_story(self):
        """`artifacts.story_draft` is None until generation finishes; the
        evaluator used to raise AttributeError on `story.text`."""
        artifacts = PipelineArtifacts(run_id="test", manifest=ExperimentManifest())
        evaluator = ComprehensiveEvaluator(device="cpu")
        result = evaluator.evaluate(artifacts, image=None)
        assert result.grounding_score == 0.0


class TestRepairStoryWorks:
    """`ClaimVerifier.repair_story` called `_extract_claims_from_sentence` and
    `_classify_claim_classification` on itself; both live on `ClaimExtractor`, so
    the first call raised AttributeError."""

    def _contradicted_result(self, claim: StoryClaim) -> VerificationResult:
        return VerificationResult(
            claim_id=claim.id,
            claim=claim,
            status=ClaimStatus.CONTRADICTED.value,
            confidence=0.9,
        )

    def test_repair_does_not_raise(self):
        verifier = ClaimVerifier()
        claim = make_claim()
        story = "The balloon is blue. The girl smiled."
        result = verifier.repair_story(story, [self._contradicted_result(claim)], [])
        repaired, report = result
        assert isinstance(repaired, str)
        assert isinstance(report, list)

    def test_verifier_owns_an_extractor(self):
        assert isinstance(ClaimVerifier()._extractor, ClaimExtractor)

    def test_an_injected_extractor_is_used(self):
        extractor = MagicMock()
        extractor._extract_claims_from_sentence.return_value = []
        verifier = ClaimVerifier(extractor=extractor)
        verifier.repair_story("The balloon is blue.", [], [])
        extractor._extract_claims_from_sentence.assert_called()


class TestOptionalModelsAreAnnotated:
    """`vocab_type: callable` and `world_state: any` were used as annotations;
    `callable` is a builtin function, not a type."""

    def test_faiss_search_filter_is_callable_typed(self):
        import inspect

        from image_story.memory.faiss_store import FAISSVectorStore

        source = inspect.getsource(FAISSVectorStore.search)
        assert "filter_fn: callable" not in source

    def test_narrative_memory_world_state_is_optional_any(self):
        import inspect

        from image_story.memory.narrative import NarrativeMemory

        source = inspect.getsource(NarrativeMemory.build_narrative_elements)
        assert "world_state: any" not in source


class TestEvaluationWithoutStoryIsSafe:
    @pytest.mark.parametrize("story", [None])
    def test_no_story_is_a_zero_result(self, story):
        artifacts = PipelineArtifacts(run_id="test", manifest=ExperimentManifest())
        artifacts.story_draft = story
        result = ComprehensiveEvaluator(device="cpu").evaluate(artifacts, image=None)
        assert result.narrative_coherence == 0.0
