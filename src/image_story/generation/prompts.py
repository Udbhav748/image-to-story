"""Prompt templates for story generation."""
from collections.abc import Sequence
from typing import Any

SYSTEM_PROMPT = """You are a creative storyteller who writes grounded, engaging stories.
You carefully distinguish between visually verified facts, plausible inferences, and creative inventions.
Your stories have character, humor, surprise, and emotional resonance while never contradicting hard visual evidence."""

NO_INVENTION_RULES = """NO INVENTED NAMES OR IDENTITIES:
- Do not name any person, place, or animal breed that is not stated in the visual evidence.
- Refer to unnamed people and animals descriptively (for example "a man" or "the dog") instead of inventing names or breeds.
- Do not name a location unless the visual evidence names it."""


def _hard_facts_block(hard_facts: Sequence[str]) -> str:
    """Render the hard-fact block that the story must not contradict."""
    facts = [f.strip() for f in hard_facts if f and f.strip()]
    if facts:
        lines = "\n".join(f"- {fact}" for fact in facts)
    else:
        lines = "- (no separate fact list: treat every statement in the visual evidence as a hard fact)"
    return f"HARD FACTS (from the visual evidence; state them as given and do not contradict them):\n{lines}"


def build_single_image_prompt(
    context: str,
    creative_plan: Any = None,
    target_words: int = 100,
    *,
    hard_facts: Sequence[str] = (),
) -> str:
    """Build prompt for single image story."""

    creativity_instruction = ""
    if creative_plan:
        creativity_instruction = f"""
Creative direction:
- Genre: {creative_plan.genre}
- Tone: {creative_plan.tone}
- Creativity: {creative_plan.creativity_level:.1f}/1.0
- Surprise: {creative_plan.surprise_level:.1f}/1.0
- Humor: {creative_plan.humor_level:.1f}/1.0
- Mystery: {creative_plan.mystery_level:.1f}/1.0
"""

    return f"""{SYSTEM_PROMPT}

Write a story of approximately {target_words} words based on this visual evidence:

{context}

{_hard_facts_block(hard_facts)}

{creativity_instruction}

RULES:
1. HARD FACTS (visually grounded) - MUST NOT contradict
2. SOFT INFERENCES - plausible interpretations you may use or refine
3. CREATIVE SPACE - invent personalities, motivations, dialogue, humor, twists
4. Surprises must REINTERPRET evidence, not invent unsupported objects/events
5. Give characters distinct personalities and motivations
6. Include at least one moment of humor, surprise, or emotional depth
7. {NO_INVENTION_RULES}

Return ONLY the story."""


def build_multi_image_prompt(
    context: str,
    num_images: int,
    creative_plan: Any = None,
    target_words: int = 250,
    *,
    hard_facts: Sequence[str] = (),
) -> str:
    """Build prompt for multi-image story."""

    creativity_instruction = ""
    if creative_plan:
        creativity_instruction = f"""
Creative direction:
- Genre: {creative_plan.genre}
- Tone: {creative_plan.tone}
- Creativity: {creative_plan.creativity_level:.1f}/1.0
- Surprise: {creative_plan.surprise_level:.1f}/1.0
- Humor: {creative_plan.humor_level:.1f}/1.0
- Mystery: {creative_plan.mystery_level:.1f}/1.0
"""

    return f"""{SYSTEM_PROMPT}

Here are {num_images} consecutive images from a sequence:

{context}

{_hard_facts_block(hard_facts)}

{creativity_instruction}

Write ONE continuous story of about {target_words} words that follows these images in order.
Maintain character identity and location consistency. Connect events naturally between frames.
Do not describe each image separately.

RULES:
1. HARD FACTS (visually grounded) - MUST NOT contradict
2. SOFT INFERENCES - plausible interpretations you may use or refine
3. CREATIVE SPACE - invent personalities, motivations, dialogue, humor, twists
4. Surprises must REINTERPRET evidence, not invent unsupported objects/events
5. Resolve at least one open loop or mystery from the continuity notes
6. Include at least one callback to a recurring element
7. Give characters distinct personalities and motivations
8. {NO_INVENTION_RULES}

Return ONLY the story."""


def build_beat_prompt(
    beat: Any,
    context: str,
    previous_beats: list[str],
    creative_plan: Any = None,
    *,
    hard_facts: Sequence[str] = (),
) -> str:
    """Build prompt for a specific story beat."""

    previous_text = "\n\n".join(previous_beats) if previous_beats else "(beginning of story)"

    return f"""{SYSTEM_PROMPT}

Write the next beat of a story.

Beat: {beat.beat_type}
Description: {beat.description}
Target words: ~{beat.target_words}

Previous story:
{previous_text}

Context (visual evidence):
{context}

{_hard_facts_block(hard_facts)}

{NO_INVENTION_RULES}

Write ONLY this beat's continuation. Maintain continuity with previous beats.
Focus on: {beat.beat_type} - {beat.description}"""


VERIFICATION_PROMPT = """You are a careful fact-checker. Extract specific claims from a story and compare them against visual evidence.

For each sentence in the story, identify:
1. Subject (who/what)
2. Relation (action/state)
3. Object (who/what affected)
4. Whether it's a HARD FACT, SOFT INFERENCE, or CREATIVE FICTION

Story: {{story}}

Visual Evidence: {{evidence}}

Return claims as JSON array with fields: subject, relation, object, original_sentence, claim_type, confidence."""
