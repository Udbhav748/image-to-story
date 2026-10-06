"""ComprehensiveEvaluator wiring from grounding results (grounding stubbed)."""
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from image_story.domain.schemas import EvaluationResult, StoryDraft
from image_story.evaluation.evaluator import ComprehensiveEvaluator


def _evaluator_with_grounding(grounding_result: EvaluationResult | None) -> tuple[ComprehensiveEvaluator, Mock]:
    ev = ComprehensiveEvaluator(device="cpu")
    grounding = Mock()
    grounding.evaluate.return_value = grounding_result
    ev._grounding_evaluator = grounding
    ev._claim_extractor = Mock(extract_claims=Mock(return_value=[]))
    ev._narrative_evaluator = Mock(evaluate_narrative_quality=Mock(return_value={}))
    return ev, grounding


def _artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        story_draft=StoryDraft(text="A cat sat. A dog ran."),
        observations=[],
        world_state=None,
        creative_plan=None,
    )


def test_clip_min_and_nli_max_are_copied_from_grounding():
    grounding_result = EvaluationResult(
        grounding_score=0.7,
        clip_image_story_mean=0.31,
        clip_image_story_min=0.21,
        nli_contra_mean=0.2,
        nli_contra_max=0.88,
        length_valid=True,
        grounding_pass=True,
    )
    ev, _ = _evaluator_with_grounding(grounding_result)

    result = ev.evaluate(_artifacts(), Image.new("RGB", (4, 4)))

    assert result.clip_image_story_min == 0.21
    assert result.nli_contra_max == 0.88
    assert result.clip_image_story_mean == 0.31


def test_frames_are_passed_to_grounding_when_given():
    ev, grounding = _evaluator_with_grounding(EvaluationResult())
    first = Image.new("RGB", (4, 4), (10, 0, 0))
    second = Image.new("RGB", (4, 4), (200, 0, 0))

    ev.evaluate(_artifacts(), first, frames=[first, second])

    frames_arg = grounding.evaluate.call_args.args[0]
    assert frames_arg == [first, second]


def test_defaults_to_single_image_when_no_frames():
    ev, grounding = _evaluator_with_grounding(EvaluationResult())
    image = Image.new("RGB", (4, 4))

    ev.evaluate(_artifacts(), image)

    assert grounding.evaluate.call_args.args[0] == [image]
