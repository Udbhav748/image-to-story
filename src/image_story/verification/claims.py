"""Claim extraction and verification for story evaluation."""
import re
from typing import Any

from ..domain.enums import ClaimStatus, InformationClass
from ..domain.schemas import EvidenceRecord, StoryClaim, StoryDraft, VerificationResult
from .visual import VisualVerifier


def _content_words(text: str) -> set[str]:
    """Lower-cased word tokens, punctuation stripped."""
    return set(re.findall(r"[a-z]+", text.lower()))


# Function words ignored when matching a claim subject against evidence entities.
_STOPWORDS = {"a", "an", "the", "this", "that", "his", "her", "its", "their"}

# Colour words used for attribute-conflict detection during textual verification.
_COLORS = {
    "red", "blue", "green", "yellow", "black", "white",
    "brown", "orange", "purple", "pink", "grey", "gray",
}


class ClaimExtractor:
    """Extract verifiable claims from generated story."""

    def __init__(self):
        self._nlp = None

    def _load_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = "fallback"

    def extract_claims(self, story: StoryDraft, context: str = "") -> list[StoryClaim]:
        """Extract claims from story text."""
        self._load_nlp()

        claims = []
        sentences = self._split_sentences(story.text)

        for i, sentence in enumerate(sentences):
            sentence_claims = self._extract_claims_from_sentence(sentence, i)
            claims.extend(sentence_claims)

        # Classify claims after extraction
        for claim in claims:
            claim.claim_classification = self._classify_claim_classification(claim, context)

        return claims

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def _extract_claims_from_sentence(self, sentence: str, sent_idx: int) -> list[StoryClaim]:
        claims = []

        if self._nlp and self._nlp != "fallback":
            doc = self._nlp(sentence)
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    subject = self._find_subject(token)
                    obj = self._find_object(token)
                    if subject and obj:
                        claim = StoryClaim(
                            subject=subject.text,
                            relation=token.lemma_,
                            object=obj.text,
                            original_sentence=sentence,
                            claim_type=self._classify_claim_type(sentence, subject.text, obj.text),
                            confidence=0.8,
                        )
                        claims.append(claim)
        else:
            claims.extend(self._fallback_extract(sentence))

        return claims

    def _find_subject(self, verb_token) -> Any:
        for child in verb_token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child
        return None

    def _find_object(self, verb_token) -> Any:
        for child in verb_token.children:
            if child.dep_ in ("dobj", "pobj", "attr"):
                return child
        return None

    def _classify_claim_type(self, sentence: str, subject: str, obj: str) -> InformationClass:
        creative_markers = [
            "seemed", "appeared", "felt", "thought", "wondered", "imagined",
            "perhaps", "maybe", "possibly", "might", "could be",
            "personality", "motivation", "dream", "hope", "fear",
            "secretly", "hidden", "secret", "internal", "mental",
            "funny", "awkward", "surprisingly", "ironically",
            "metaphor", "symbolized", "represented",
            "dreamed", "hoped", "feared", "imagined",
            "wanted", "desired", "wished",
            "judged", "opinion", "attitude", "perspective",
            "ironic", "sarcastic", "deadpan", "absurd", "ridiculous",
            "pretended", "pretending", "imagined",
        ]

        sentence_lower = sentence.lower()

        # Safe creative phrases that should NEVER be penalized
        safe_creative_phrases = [
            "secretly", "hidden", "secret", "internal", "mental",
            "personality", "motivation", "dream", "hope", "fear",
            "metaphor", "symbolized", "represented",
            "judged", "opinion", "attitude", "perspective",
            "ironic", "sarcastic", "deadpan", "absurd", "ridiculous",
            "pretended", "pretending", "imagined",
            "wanted", "desired", "wished", "hoped", "dreamed",
        ]

        # First: Check for SAFE creative phrases - these are ALWAYS creative
        if any(phrase in sentence_lower for phrase in safe_creative_phrases):
            return InformationClass.CREATIVE_SPACE

        # Check for creative markers (but exclude inference phrases)
        has_creative_marker = any(marker in sentence_lower for marker in creative_markers)
        has_inference_phrase = any(inf in sentence_lower for inf in [
            "appeared to be", "seemed to be", "looked like", "looked as if", "as if"
        ])

        if has_creative_marker and not has_inference_phrase:
            return InformationClass.CREATIVE_SPACE

        # Check inference phrases
        inference_phrases = [
            "appeared to be", "seemed to be", "looked like", "looked as if", "as if",
            "might be", "probably", "likely", "perhaps", "possibly", "maybe", "seemed to be",
            "appeared to be", "looked as if", "as if"
        ]
        if any(inf in sentence_lower for inf in inference_phrases):
            return InformationClass.SOFT_INFERENCE

        visual_words = ["wearing", "holding", "standing", "sitting", "walking", "carrying",
                       "red", "blue", "green", "large", "small", "next to", "in front of",
                       "behind", "under", "above", "inside", "outside", "visible", "saw",
                       "looked like", "appeared to be"]
        if any(word in sentence_lower for word in visual_words):
            return InformationClass.HARD_FACT

        return InformationClass.SOFT_INFERENCE

    def _classify_claim_classification(self, claim: StoryClaim, context: str) -> str:
        """Classify claim as OBSERVED, INFERRED, or CREATIVE."""
        sentence_lower = claim.original_sentence.lower()

        # Creative markers - internal states, personality, metaphor, humor, attitude
        creative_markers = [
            "seemed", "appeared", "felt", "thought", "wondered", "imagined",
            "perhaps", "maybe", "possibly", "might", "could be",
            "personality", "motivation", "dream", "hope", "fear",
            "secretly", "hidden", "secret", "internal", "mental",
            "funny", "awkward", "surprisingly", "ironically",
            "metaphor", "symbolized", "represented",
            "dreamed", "hoped", "feared", "imagined",
            "wanted", "desired", "wished",
            "judged", "opinion", "attitude", "perspective",
            "ironic", "sarcastic", "deadpan", "absurd", "ridiculous",
            "pretended", "pretending", "imagined",
        ]

        # Visual/observable words - direct visual facts.
        # Kept for reference: claim classification currently relies on
        # `inference_markers` and `creative_markers` below, not on this list.
        _visual_words = ["wearing", "holding", "standing", "sitting", "walking", "carrying",
                         "red", "blue", "green", "large", "small", "next to", "in front of",
                         "behind", "under", "above", "inside", "outside", "visible", "saw",
                         "looked like", "appeared to be"]

        # Inference markers - qualified statements (uncertainty about visual facts)
        inference_markers = ["appeared to be", "seemed to be", "looked like", "might be", "probably",
                           "likely", "perhaps", "possibly", "maybe", "seemed to be",
                           "appeared to be", "looked as if", "as if"]

        # Safe creative phrases that should NEVER be penalized
        safe_creative_phrases = [
            "secretly", "hidden", "secret", "internal", "mental",
            "personality", "motivation", "dream", "hope", "fear",
            "metaphor", "symbolized", "represented",
            "judged", "opinion", "attitude", "perspective",
            "ironic", "sarcastic", "deadpan", "absurd", "ridiculous",
            "pretended", "pretending", "imagined",
            "wanted", "desired", "wished", "hoped", "dreamed",
        ]

        # First: Check for SAFE creative phrases - these are ALWAYS creative
        if any(phrase in sentence_lower for phrase in safe_creative_phrases):
            return "creative"

        # Check for creative markers (but exclude inference phrases)
        has_creative_marker = any(marker in sentence_lower for marker in creative_markers)
        has_inference_phrase = any(inf in sentence_lower for inf in [
            "appeared to be", "seemed to be", "looked like", "looked as if", "as if"
        ])

        if has_creative_marker and not has_inference_phrase:
            return "creative"

        # Check inference markers (qualified uncertainty about visual facts)
        if any(inf in sentence_lower for inf in inference_markers):
            return "inferred"

        # Check visual words (direct visual facts)
        if any(word in sentence_lower for word in [
            "wearing", "holding", "standing", "sitting", "walking", "carrying",
            "red", "blue", "green", "large", "small", "next to", "in front of",
            "behind", "under", "above", "inside", "outside", "visible", "saw",
        ]):
            return "observed"

        # Default to inferred for factual statements that aren't clearly visual
        return "inferred"

    def _fallback_extract(self, sentence: str) -> list[StoryClaim]:
        """Simple regex-based claim extraction fallback."""
        claims = []

        patterns = [
            r"(\w+(?:\s+\w+)*)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+(?:has|have|had)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+(?:holds?|hold|carries?|carry)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+(?:stands?|stand|sits?|sit)\s+(?:\w+\s+)*(\w+(?:\s+\w+)*)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                relation = match.group(0).split(subject)[1].split(obj)[0].strip()

                # Classify claim type based on sentence content
                claim_type = self._classify_claim_type(sentence, subject, obj)

                claim = StoryClaim(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    original_sentence=sentence,
                    claim_type=claim_type,
                    confidence=0.6,
                )
                claims.append(claim)

        return claims


class ClaimVerifier:
    """Verify claims against visual evidence."""

    def __init__(
        self,
        visual_verifier: VisualVerifier | None = None,
        extractor: ClaimExtractor | None = None,
    ):
        self._visual_verifier = visual_verifier
        # `repair_story` needs to re-extract claims from a sentence and classify
        # them; both live on ClaimExtractor. It used to call them on `self` and
        # raise AttributeError on first use.
        self._extractor = extractor or ClaimExtractor()

    @staticmethod
    def _link_evidence(
        claim: StoryClaim,
        supporting: list[EvidenceRecord],
        contradicting: list[EvidenceRecord],
    ) -> None:
        """Record the ids of the evidence that supports or contradicts a claim.

        Ids are taken from the EvidenceRecord objects themselves, so every id in
        `claim.evidence_ids` resolves to an evidence record that was checked.
        """
        ids: list[str] = []
        for record in [*supporting, *contradicting]:
            if record.id not in ids:
                ids.append(record.id)
        claim.evidence_ids = ids

    def verify_claims(
        self,
        claims: list[StoryClaim],
        image: Any,  # PIL Image
        evidence_records: list[EvidenceRecord],
    ) -> list[VerificationResult]:
        """Verify all claims against visual evidence."""
        results = []

        for claim in claims:
            if self._visual_verifier and image:
                result = self._visual_verifier.verify_claim(claim, image, evidence_records)
                self._link_evidence(
                    result.claim, result.supporting_evidence, result.contradicting_evidence
                )
            else:
                result = self._textual_verification(claim, evidence_records)
            results.append(result)

        return results

    def _textual_verification(
        self,
        claim: StoryClaim,
        evidence_records: list[EvidenceRecord],
    ) -> VerificationResult:
        """Verify one claim against evidence by text overlap and attribute conflict.

        A claim about an entity the evidence describes is supported. A claim that
        asserts a different colour or material for the same entity is
        contradicted. This is a lexical check, not a model: it catches attribute
        swaps ("a red balloon" vs "a blue balloon") and misses everything else.
        """
        supporting: list[EvidenceRecord] = []
        contradicting: list[EvidenceRecord] = []

        claim_terms = {
            claim.subject.lower(),
            claim.relation.lower(),
            claim.object.lower(),
        }
        claim_text = claim.to_natural_language().lower()
        claim_colors = _COLORS & _content_words(claim_text)

        subject_words = _content_words(claim.subject) - _STOPWORDS

        for evidence in evidence_records:
            evidence_text = evidence.evidence_text.lower()
            evidence_entity = evidence.entity.lower()

            matches = sum(1 for term in claim_terms if term and term in evidence_text)
            matches += sum(1 for term in claim_terms if term and term in evidence_entity)

            # Same entity, incompatible colour attribute -> contradiction.
            same_entity = bool(subject_words & _content_words(evidence_entity))
            if claim_colors and same_entity:
                evidence_colors = _COLORS & _content_words(evidence_text)
                if evidence_colors and not (claim_colors & evidence_colors):
                    contradicting.append(evidence)
                    continue

            if matches >= 2:
                supporting.append(evidence)
            elif matches == 1 and evidence.confidence > 0.8:
                supporting.append(evidence)

        if contradicting:
            status = ClaimStatus.CONTRADICTED
            confidence = 0.9
        elif supporting:
            status = ClaimStatus.SUPPORTED
            confidence = min(0.6 + len(supporting) * 0.1, 0.9)
        else:
            status = ClaimStatus.UNSUPPORTED
            confidence = 0.5

        self._link_evidence(claim, supporting, contradicting)
        return VerificationResult(
            claim_id=claim.id,
            claim=claim,
            status=status.value,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            confidence=confidence,
            notes=f"Textual verification: {len(supporting)} supporting, {len(contradicting)} contradicting",
        )

    def textual_verification_batch(
        self,
        claims: list[StoryClaim],
        evidence_records: list[EvidenceRecord],
    ) -> list[VerificationResult]:
        """Verify claims without an image, using evidence text only."""
        return [self._textual_verification(claim, evidence_records) for claim in claims]

    def compute_claim_grounding_score(self, results: list[VerificationResult]) -> float:
        """Compute overall claim grounding score."""
        if not results:
            return 0.0

        supported = 0.0
        contradicted = 0.0
        total = 0
        creative_count = 0

        for r in results:
            claim = r.claim
            # Handle missing claim_classification gracefully
            classification = getattr(claim, 'claim_classification', 'inferred')
            # Creative claims don't count against grounding
            if classification == "creative":
                creative_count += 1
                continue
            # Observed claims must be supported
            elif classification == "observed":
                if r.status == ClaimStatus.SUPPORTED.value:
                    supported += 1
                elif r.status == ClaimStatus.CONTRADICTED.value:
                    contradicted += 1
            # Inferred claims are softer
            else:  # inferred
                if r.status == ClaimStatus.SUPPORTED.value:
                    supported += 0.5
                elif r.status == ClaimStatus.CONTRADICTED.value:
                    contradicted += 0.5
            total += 1

        # Adjust total to exclude creative claims
        total = total - creative_count

        if total <= 0:
            return 1.0  # all claims were creative

        return max(0.0, (supported - contradicted * 0.5) / total)

    def repair_story(
        self,
        story: str,
        verification_results: list[VerificationResult],
        evidence_records: list[EvidenceRecord],
        max_attempts: int = 2,
    ) -> tuple[str, list[dict]]:
        """Repair colour attributes that contradict evidence, in a bounded loop.

        Each attempt re-extracts the claims of every sentence, re-verifies them
        textually against `evidence_records`, and rewrites a sentence only when
        an observed claim is CONTRADICTED: the contradicted colour word is
        replaced by the colour the evidence reports for the same entity. The
        loop stops when an attempt changes nothing or after `max_attempts`.

        Only observed claims are touched. Creative, inferred and uncontradicted
        sentences are left as they are. `verification_results` is accepted for
        API compatibility; repair decisions are made from `evidence_records`.

        Returns the repaired story and a report with one entry per changed
        sentence per attempt: {"attempt", "original", "repaired", "issues"}.
        """
        if max_attempts < 1:
            return story, []

        current = story
        repair_report: list[dict] = []

        for attempt in range(1, max_attempts + 1):
            sentences = _split_text(current)
            changed = False
            next_sentences: list[str] = []

            for sentence in sentences:
                repaired, issues = self._repair_sentence(sentence, evidence_records)
                next_sentences.append(repaired)
                if repaired != sentence:
                    changed = True
                    repair_report.append({
                        "attempt": attempt,
                        "original": sentence,
                        "repaired": repaired,
                        "issues": issues,
                    })

            current = " ".join(next_sentences)
            if not changed:
                break

        return current, repair_report

    def _repair_sentence(
        self,
        sentence: str,
        evidence_records: list[EvidenceRecord],
    ) -> tuple[str, list[dict]]:
        """Rewrite the contradicted colour words of one sentence, if any."""
        issues: list[dict] = []
        repaired = sentence

        for claim in self._extractor._extract_claims_from_sentence(sentence, 0):
            claim.claim_classification = self._extractor._classify_claim_classification(claim, "")
            if claim.claim_classification != "observed":
                continue

            result = self._textual_verification(claim, evidence_records)
            if result.status != ClaimStatus.CONTRADICTED.value:
                continue

            claim_colors = _COLORS & _content_words(claim.to_natural_language().lower())
            evidence_colors: set[str] = set()
            for record in result.contradicting_evidence:
                evidence_colors |= _COLORS & _content_words(record.evidence_text.lower())
            # Only swap when the evidence names exactly one replacement colour.
            if len(evidence_colors) != 1:
                continue
            replacement = next(iter(evidence_colors))

            for wrong in sorted(claim_colors - evidence_colors):
                pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
                new_sentence, count = pattern.subn(replacement, repaired, count=1)
                if count:
                    issues.append({
                        "claim": claim.to_natural_language(),
                        "status": result.status,
                        "replaced": wrong,
                        "with": replacement,
                    })
                    repaired = new_sentence

        return repaired, issues


def _split_text(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
