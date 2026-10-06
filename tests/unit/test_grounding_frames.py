"""CLIP grounding over all frames, and the story length window.

CLIP and NLI are stubbed: no model weights are loaded.
"""
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from image_story.config.defaults import STORY_WORD_MAX, STORY_WORD_MIN
from image_story.domain.schemas import StoryDraft
from image_story.evaluation.grounding import GroundingEvaluator

# Per-frame, per-sentence CLIP similarities, keyed by the frame's red value.
FAKE_SIMS = {10: [0.2, 0.4], 200: [0.4, 0.6]}


def _frame(red: int) -> Image.Image:
    return Image.new("RGB", (4, 4), (red, 0, 0))


def _fake_clip_sims(image, texts):
    return torch.tensor(FAKE_SIMS[image.getpixel((0, 0))[0]])


def _stub_evaluator() -> GroundingEvaluator:
    ev = GroundingEvaluator("cpu")
    # Non-None sentinels make _load_clip / _load_nli return early.
    ev._clip_model = object()
    ev._clip_processor = object()
    ev._nli_model = object()
    ev._nli_tokenizer = object()
    ev._contra_idx = 0
    return ev


def test_clip_sims_are_averaged_across_frames():
    ev = _stub_evaluator()
    frames = [_frame(10), _frame(200)]
    with patch.object(ev, "_compute_clip_sims", side_effect=_fake_clip_sims):
        sims = ev._compute_clip_sims_across_frames(frames, ["a", "b"])
    assert torch.allclose(sims, torch.tensor([0.3, 0.5]))


def test_evaluate_uses_all_frames_for_clip_mean_and_min():
    ev = _stub_evaluator()
    story = StoryDraft(text="A cat sat. A dog ran.")
    with (
        patch.object(ev, "_compute_clip_sims", side_effect=_fake_clip_sims),
        patch.object(ev, "_compute_nli_contradictions", return_value=torch.tensor([0.0, 0.0])),
    ):
        result = ev.evaluate([_frame(10), _frame(200)], "a caption", story)
    # Per-sentence means are [0.3, 0.5]; mean = 0.4, min = 0.3.
    assert result.clip_image_story_mean == pytest.approx(0.4)
    assert result.clip_image_story_min == pytest.approx(0.3)


def test_single_image_still_accepted():
    ev = _stub_evaluator()
    story = StoryDraft(text="A cat sat. A dog ran.")
    with (
        patch.object(ev, "_compute_clip_sims", side_effect=_fake_clip_sims),
        patch.object(ev, "_compute_nli_contradictions", return_value=torch.tensor([0.0, 0.0])),
    ):
        result = ev.evaluate(_frame(10), "a caption", story)
    assert result.clip_image_story_mean == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("word_count", "expected"),
    [(STORY_WORD_MIN - 1, False), (STORY_WORD_MIN, True), (STORY_WORD_MAX, True), (STORY_WORD_MAX + 1, False)],
)
def test_length_valid_uses_shared_window(word_count, expected):
    ev = _stub_evaluator()
    story = StoryDraft(text=" ".join(["word"] * word_count) + ".")
    with (
        patch.object(ev, "_compute_clip_sims", side_effect=lambda img, texts: torch.tensor([0.3] * len(texts))),
        patch.object(ev, "_compute_nli_contradictions", side_effect=lambda cap, texts: torch.zeros(len(texts))),
    ):
        result = ev.evaluate(_frame(10), "a caption", story)
    assert result.length_valid is expected
