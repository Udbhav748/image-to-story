"""Canonical configuration defaults and mode presets.

`PipelineConfig` in `domain/schemas.py` is the only definition of the
configuration contract. The per-mode presets that used to be a 130-line if/elif
chain inside `PipelineConfig.from_mode` are declared here, so a new mode is added
in one place and the schema stays a plain data contract.
"""
from __future__ import annotations

from typing import Any


def _memory(**overrides: Any) -> dict[str, Any]:
    """Scene/retrieval knobs shared by every mode."""
    base: dict[str, Any] = {
        "scene_similarity_threshold": 0.65,
        "max_scene_size": 20,
        "scene_retrieval_threshold": 0.5,
        "evidence_retrieval_threshold": 0.4,
    }
    base.update(overrides)
    return base


def _budget(safe: int, risky: int) -> dict[str, Any]:
    """Risk-aware creative budget (V2.2). `forbidden_visual` is always 0."""
    return {
        "safe_creative": {"max": safe, "used": 0},      # personality, humor, dialogue, metaphor
        "risky_inferred": {"max": risky, "used": 0},    # motivations, uncertain actions
        "forbidden_visual": {"max": 0, "used": 0},      # new objects, colors, materials, people
    }


#: Default risk-aware creative budget; `PipelineConfig` copies this per instance.
DEFAULT_CREATIVE_BUDGET: dict[str, dict[str, int]] = _budget(safe=8, risky=3)

DEFAULT_CREATIVITY: dict[str, float] = {
    "creativity": 0.7,
    "surprise": 0.6,
    "humor": 0.5,
    "mystery": 0.4,
    "emotion": 0.5,
    "dialogue": 0.4,
    "metaphor": 0.3,
}

FULL_CREATIVITY: dict[str, float] = {
    "creativity": 0.8,
    "surprise": 0.7,
    "humor": 0.6,
    "mystery": 0.5,
    "emotion": 0.6,
    "dialogue": 0.5,
    "metaphor": 0.4,
}

BASELINE_CREATIVITY: dict[str, float] = {
    "creativity": 0.3,
    "surprise": 0.2,
    "humor": 0.2,
    "mystery": 0.1,
    "emotion": 0.2,
    "dialogue": 0.2,
    "metaphor": 0.1,
}

#: Presets applied by `PipelineConfig.from_mode`, keyed by mode name.
#: Values are identical to the presets the pipeline used before this table
#: existed; see `tests/unit/test_config_defaults.py` which pins them.
MODE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "use_grounding_dino": False,
        "use_ocr": False,
        "use_faiss": False,
        "use_creative_planner": False,
        "use_verification": False,
        "target_story_words": 100,
        **_memory(top_k_scenes=3, top_k_evidence_per_scene=5, max_scenes_in_context=3),
        "creative_budget": _budget(safe=2, risky=1),
    },
    "standard": {
        "use_grounding_dino": True,
        "use_ocr": False,
        "use_faiss": True,
        "use_creative_planner": True,
        "use_verification": True,
        "target_story_words": 250,
        **_memory(top_k_scenes=5, top_k_evidence_per_scene=10, max_scenes_in_context=5),
        "creative_budget": _budget(safe=8, risky=3),
    },
    "full": {
        "use_grounding_dino": True,
        "use_ocr": True,
        "use_faiss": True,
        "use_creative_planner": True,
        "use_verification": True,
        "target_story_words": 300,
        "creativity_config": FULL_CREATIVITY,
        **_memory(top_k_scenes=7, top_k_evidence_per_scene=15, max_scenes_in_context=7),
        "creative_budget": _budget(safe=10, risky=4),
    },
    "baseline": {
        "use_grounding_dino": False,
        "use_ocr": False,
        "use_faiss": False,
        "use_creative_planner": False,
        "use_verification": False,
        "target_story_words": 100,
        "creativity_config": BASELINE_CREATIVITY,
        **_memory(top_k_scenes=3, top_k_evidence_per_scene=5, max_scenes_in_context=3),
        "creative_budget": _budget(safe=2, risky=1),
    },
    "collection": {
        "use_grounding_dino": True,
        "use_ocr": False,
        "use_faiss": True,
        "use_creative_planner": True,
        "use_verification": True,
        "target_story_words": 300,
        **_memory(top_k_scenes=10, top_k_evidence_per_scene=20, max_scenes_in_context=10),
        "creative_budget": _budget(safe=10, risky=4),
    },
}
