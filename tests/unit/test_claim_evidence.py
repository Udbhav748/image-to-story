"""Claim -> evidence linking and the bounded repair loop (verification.claims)."""
from image_story.domain.enums import ClaimStatus
from image_story.domain.schemas import EvidenceRecord, StoryClaim, VerificationResult
from image_story.verification.claims import ClaimVerifier


def _claim(subject, relation, obj, sentence):
    return StoryClaim(subject=subject, relation=relation, object=obj, original_sentence=sentence)


def _evidence(entity, text, eid):
    return EvidenceRecord(id=eid, entity=entity, evidence_text=text, confidence=0.9)


class StubVisualVerifier:
    """Stands in for VisualVerifier: supports every claim with the given records."""

    def __init__(self, supporting):
        self._supporting = supporting
        self.calls = 0

    def verify_claim(self, claim, image, evidence_records):
        self.calls += 1
        return VerificationResult(
            claim_id=claim.id,
            claim=claim,
            status=ClaimStatus.SUPPORTED.value,
            supporting_evidence=list(self._supporting),
            confidence=0.8,
        )


def test_textual_verification_populates_evidence_ids_with_real_ids():
    evidence = [
        _evidence("balloon", "Detected red balloon near the girl", "obs_red"),
        _evidence("tree", "Detected green tree", "obs_tree"),
    ]
    claim = _claim("balloon", "is", "red", "The balloon is red.")
    assert claim.evidence_ids == []

    results = ClaimVerifier().textual_verification_batch([claim], evidence)

    assert claim.evidence_ids, "evidence_ids should be populated after verification"
    known = {e.id for e in evidence}
    assert set(claim.evidence_ids) <= known
    assert "obs_red" in claim.evidence_ids
    assert results[0].claim is claim


def test_contradicting_evidence_is_also_linked():
    evidence = [_evidence("balloon", "Detected red balloon", "obs_red")]
    claim = _claim("balloon", "is", "blue", "The balloon is blue.")

    results = ClaimVerifier().textual_verification_batch([claim], evidence)

    assert results[0].status == ClaimStatus.CONTRADICTED.value
    assert claim.evidence_ids == ["obs_red"]


def test_visual_verifier_path_links_evidence_ids_to_real_records():
    evidence = [
        _evidence("girl", "A girl holding a kite", "obs_girl"),
        _evidence("kite", "A kite in the sky", "obs_kite"),
    ]
    stub = StubVisualVerifier(supporting=evidence)
    claim = _claim("girl", "holds", "kite", "The girl holds the kite.")

    ClaimVerifier(visual_verifier=stub).verify_claims([claim], image=object(), evidence_records=evidence)

    assert stub.calls == 1
    assert claim.evidence_ids == ["obs_girl", "obs_kite"]
    assert set(claim.evidence_ids) <= {e.id for e in evidence}


def test_claim_with_no_matching_evidence_has_no_links():
    evidence = [_evidence("tree", "Detected green tree", "obs_tree")]
    claim = _claim("dragon", "flies", "castle", "The dragon flies over the castle.")

    results = ClaimVerifier().textual_verification_batch([claim], evidence)

    assert results[0].status == ClaimStatus.UNSUPPORTED.value
    assert claim.evidence_ids == []


def test_repair_rewrites_contradicted_colour_and_stops_when_stable():
    evidence = [_evidence("balloon", "Detected red balloon", "obs_red")]
    story = "The balloon is blue. The girl smiled."

    repaired, report = ClaimVerifier().repair_story(story, [], evidence, max_attempts=3)

    assert repaired == "The balloon is red. The girl smiled."
    assert len(report) == 1
    assert report[0]["attempt"] == 1
    assert report[0]["original"] == "The balloon is blue."
    assert report[0]["repaired"] == "The balloon is red."
    assert report[0]["issues"][0]["replaced"] == "blue"


def test_repair_is_bounded_by_max_attempts():
    evidence = [_evidence("balloon", "Detected red balloon", "obs_red")]
    story = "The balloon is blue."

    repaired, report = ClaimVerifier().repair_story(story, [], evidence, max_attempts=0)
    assert repaired == story
    assert report == []


def test_repair_leaves_uncontradicted_text_alone():
    evidence = [_evidence("balloon", "Detected red balloon", "obs_red")]
    story = "The balloon is red. The girl smiled."

    repaired, report = ClaimVerifier().repair_story(story, [], evidence)

    assert repaired == story
    assert report == []
