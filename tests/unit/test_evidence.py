"""Evidence layer: canonical text projection, provenance records, confidence banding."""
import pytest

from image_story.domain.enums import EvidenceConfidence, EvidenceType, SourceModel
from image_story.domain.schemas import EvidenceRecord
from image_story.evidence import (
    IndexedEvidenceRecord,
    SceneEvidenceLink,
    classify_confidence,
    confidence_meets,
    evidence_to_text,
)


def make_evidence(**overrides) -> EvidenceRecord:
    defaults = {
        "entity": "girl",
        "type": EvidenceType.PERSON,
        "action": "sitting",
        "relationship": None,
        "frame_id": 3,
        "evidence_text": "Person detected: girl",
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return EvidenceRecord(**defaults)


class TestEvidenceToText:
    def test_includes_all_present_fields(self):
        text = evidence_to_text(make_evidence(relationship="holding balloon"))
        assert "Entity: girl" in text
        assert "Type: person" in text
        assert "Action: sitting" in text
        assert "Relation: holding balloon" in text
        assert "Frame: 3" in text

    def test_frame_can_be_omitted(self):
        assert "Frame:" not in evidence_to_text(make_evidence(), include_frame=False)

    def test_confidence_is_opt_in(self):
        assert "Confidence" not in evidence_to_text(make_evidence())
        assert "Confidence: high" in evidence_to_text(make_evidence(), include_confidence=True)

    def test_text_limit_truncates_evidence_text(self):
        text = evidence_to_text(make_evidence(evidence_text="x" * 200), text_limit=10)
        assert "x" * 10 in text
        assert "x" * 11 not in text

    def test_separator_is_configurable(self):
        text = evidence_to_text(make_evidence(entity=None, action=None, evidence_text=""), separator=",")
        assert "Type: person" in text

    def test_empty_record_is_empty(self):
        record = EvidenceRecord(entity="", type=EvidenceType.OBJECT, evidence_text="", frame_id=-1)
        assert evidence_to_text(record) == "Type: object"

    def test_negative_frame_is_omitted(self):
        assert "Frame:" not in evidence_to_text(make_evidence(frame_id=-1))

    def test_output_is_deterministic(self):
        record = make_evidence()
        assert evidence_to_text(record) == evidence_to_text(record)


class TestIndexedEvidenceRecord:
    def test_round_trip(self):
        indexed = IndexedEvidenceRecord(
            evidence=make_evidence(),
            image_id="img_1",
            scene_id="scene_1",
            frame_id=3,
            collection_id="coll_1",
            index_position=7,
        )
        restored = IndexedEvidenceRecord.from_dict(indexed.to_dict())
        assert restored.image_id == "img_1"
        assert restored.scene_id == "scene_1"
        assert restored.collection_id == "coll_1"
        assert restored.index_position == 7
        assert restored.evidence.entity == "girl"

    def test_default_index_position(self):
        indexed = IndexedEvidenceRecord(
            evidence=make_evidence(),
            image_id="i",
            scene_id="s",
            frame_id=0,
            collection_id="c",
        )
        assert indexed.index_position == -1
        assert IndexedEvidenceRecord.from_dict(indexed.to_dict()).index_position == -1


class TestSceneEvidenceLink:
    def test_defaults_to_full_relevance(self):
        link = SceneEvidenceLink(scene_id="s1", evidence_id="obs_1", evidence_index=0)
        assert link.relevance_score == 1.0


class TestConfidenceBanding:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.95, EvidenceConfidence.HIGH),
            (0.85, EvidenceConfidence.HIGH),
            (0.84, EvidenceConfidence.MEDIUM),
            (0.5, EvidenceConfidence.MEDIUM),
            (0.49, EvidenceConfidence.LOW),
            (0.0, EvidenceConfidence.LOW),
        ],
    )
    def test_thresholds(self, score, expected):
        assert classify_confidence(score) is expected

    def test_classify_matches_schema_post_init(self):
        for score in (0.0, 0.4, 0.5, 0.7, 0.85, 0.99):
            record = EvidenceRecord(confidence=score, source=SourceModel.FLORENCE2)
            assert classify_confidence(score) is record.confidence_class

    def test_meets_is_ordered(self):
        assert confidence_meets(0.9, EvidenceConfidence.HIGH)
        assert confidence_meets(0.9, EvidenceConfidence.MEDIUM)
        assert confidence_meets(0.9, EvidenceConfidence.LOW)
        assert not confidence_meets(0.6, EvidenceConfidence.HIGH)
        assert not confidence_meets(0.2, EvidenceConfidence.MEDIUM)
