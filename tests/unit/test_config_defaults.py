"""Mode presets are pinned so that reorganisation cannot silently change them.

`PipelineConfig.from_mode` used to be a 130-line if/elif chain inside the schema.
It now reads `config.defaults.MODE_PRESETS`. These tests pin every value that
chain produced, so the refactor is provably behaviour-preserving.
"""
import pytest

from image_story.config.defaults import (
    BASELINE_CREATIVITY,
    DEFAULT_CREATIVE_BUDGET,
    DEFAULT_CREATIVITY,
    FULL_CREATIVITY,
    MODE_PRESETS,
)
from image_story.domain.enums import PipelineMode
from image_story.domain.schemas import PipelineConfig

SHARED_MEMORY = {
    "scene_similarity_threshold": 0.65,
    "max_scene_size": 20,
    "scene_retrieval_threshold": 0.5,
    "evidence_retrieval_threshold": 0.4,
}


def test_every_mode_has_a_preset():
    assert set(MODE_PRESETS) == {"fast", "standard", "full", "baseline", "collection"}


def test_preset_keys_are_valid_config_fields():
    valid = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    for mode, preset in MODE_PRESETS.items():
        unknown = set(preset) - valid
        assert not unknown, f"{mode} preset has unknown fields: {unknown}"


@pytest.mark.parametrize("mode", sorted(MODE_PRESETS))
def test_from_mode_sets_mode_and_preset(mode):
    config = PipelineConfig.from_mode(mode)
    assert config.mode == mode
    for key, value in MODE_PRESETS[mode].items():
        assert getattr(config, key) == value, f"{mode}.{key}"


def test_fast_preset():
    config = PipelineConfig.from_mode("fast")
    assert config.use_grounding_dino is False
    assert config.use_ocr is False
    assert config.use_faiss is False
    assert config.use_creative_planner is False
    assert config.use_verification is False
    assert config.target_story_words == 100
    assert config.top_k_scenes == 3
    assert config.top_k_evidence_per_scene == 5
    assert config.max_scenes_in_context == 3
    assert config.creative_budget["safe_creative"]["max"] == 2
    assert config.creative_budget["risky_inferred"]["max"] == 1


def test_standard_preset():
    config = PipelineConfig.from_mode("standard")
    assert config.use_grounding_dino is True
    assert config.use_ocr is False
    assert config.use_faiss is True
    assert config.use_creative_planner is True
    assert config.use_verification is True
    assert config.target_story_words == 250
    assert config.top_k_scenes == 5
    assert config.top_k_evidence_per_scene == 10
    assert config.max_scenes_in_context == 5
    assert config.creative_budget["safe_creative"]["max"] == 8
    assert config.creative_budget["risky_inferred"]["max"] == 3


def test_full_preset():
    config = PipelineConfig.from_mode("full")
    assert config.use_ocr is True
    assert config.target_story_words == 300
    assert config.top_k_scenes == 7
    assert config.top_k_evidence_per_scene == 15
    assert config.max_scenes_in_context == 7
    assert config.creativity_config == FULL_CREATIVITY
    assert config.creative_budget["safe_creative"]["max"] == 10
    assert config.creative_budget["risky_inferred"]["max"] == 4


def test_baseline_preset():
    config = PipelineConfig.from_mode("baseline")
    assert config.use_faiss is False
    assert config.use_creative_planner is False
    assert config.use_verification is False
    assert config.target_story_words == 100
    assert config.creativity_config == BASELINE_CREATIVITY


def test_collection_preset():
    config = PipelineConfig.from_mode("collection")
    assert config.top_k_scenes == 10
    assert config.top_k_evidence_per_scene == 20
    assert config.max_scenes_in_context == 10
    assert config.target_story_words == 300


@pytest.mark.parametrize("mode", sorted(MODE_PRESETS))
def test_presets_share_memory_thresholds(mode):
    config = PipelineConfig.from_mode(mode)
    for key, value in SHARED_MEMORY.items():
        assert getattr(config, key) == value


@pytest.mark.parametrize("mode", sorted(MODE_PRESETS))
def test_forbidden_visual_budget_is_always_zero(mode):
    """Inventing new objects/people is never allowed, in any mode."""
    config = PipelineConfig.from_mode(mode)
    assert config.creative_budget["forbidden_visual"]["max"] == 0


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        PipelineConfig.from_mode("nonsense")


def test_enum_and_string_modes_agree():
    from image_story.config.settings import get_pipeline_config

    assert get_pipeline_config(PipelineMode.FAST).to_dict() == get_pipeline_config("fast").to_dict()


def test_defaults_are_not_shared_between_instances():
    """Mutable default factories must not leak state across configs."""
    a = PipelineConfig()
    b = PipelineConfig()
    a.creative_budget["safe_creative"]["max"] = 99
    a.creativity_config["creativity"] = 0.1
    assert b.creative_budget["safe_creative"]["max"] == 8
    assert b.creativity_config["creativity"] == pytest.approx(0.7)


def test_module_defaults_match_schema_defaults():
    assert PipelineConfig().creativity_config == DEFAULT_CREATIVITY
    assert PipelineConfig().creative_budget == DEFAULT_CREATIVE_BUDGET
