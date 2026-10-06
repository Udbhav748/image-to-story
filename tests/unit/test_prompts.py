"""Invent-free prompt construction (generation.prompts)."""
from types import SimpleNamespace

from image_story.generation.prompts import (
    build_beat_prompt,
    build_multi_image_prompt,
    build_single_image_prompt,
)

FACTS = ["A red ball on the grass.", "Two dogs by a fence."]


def _plan():
    return SimpleNamespace(
        genre="comedy", tone="warm", creativity_level=0.5,
        surprise_level=0.5, humor_level=0.5, mystery_level=0.2,
    )


def test_single_prompt_contains_hard_fact_block_and_forbid_rule():
    prompt = build_single_image_prompt("ctx", _plan(), 80, hard_facts=FACTS)
    assert "HARD FACTS" in prompt
    for fact in FACTS:
        assert f"- {fact}" in prompt
    assert "Do not name any person, place, or animal breed" in prompt


def test_multi_prompt_contains_hard_fact_block_and_forbid_rule():
    prompt = build_multi_image_prompt("ctx", 2, None, 200, hard_facts=FACTS)
    assert "HARD FACTS" in prompt
    assert "- Two dogs by a fence." in prompt
    assert "Do not name any person, place, or animal breed" in prompt


def test_beat_prompt_contains_forbid_rule():
    beat = SimpleNamespace(beat_type="rising", description="tension", target_words=40)
    prompt = build_beat_prompt(beat, "ctx", ["Earlier."], hard_facts=FACTS)
    assert "HARD FACTS" in prompt
    assert "Do not name any person, place, or animal breed" in prompt


def test_empty_hard_facts_still_states_the_rule():
    prompt = build_multi_image_prompt("ctx", 1)
    assert "HARD FACTS" in prompt
    assert "Do not name any person, place, or animal breed" in prompt
