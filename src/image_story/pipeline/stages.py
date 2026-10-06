"""Canonical stage names.

The orchestrator records per-stage durations in `PipelineArtifacts.runtime`.
Declaring the keys here keeps producers and consumers (evaluation, reports,
tests) from drifting apart.
"""
from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    """Stage identifiers as they appear in `PipelineArtifacts.runtime`."""

    VISION = "vision_s"
    WORLD_STATE = "world_state_s"
    COLLECTION_MEMORY = "collection_memory_s"
    RETRIEVAL = "retrieval_s"
    RANKING = "ranking_s"
    PLANNING = "planning_s"
    CONTEXT = "context_s"
    GENERATION = "generation_s"
    EVALUATION = "evaluation_s"


#: Stages present in every run.
CORE_STAGES: tuple[Stage, ...] = (
    Stage.VISION,
    Stage.WORLD_STATE,
    Stage.RETRIEVAL,
    Stage.RANKING,
    Stage.PLANNING,
    Stage.CONTEXT,
    Stage.GENERATION,
)

#: Stages that only appear when evaluation is requested.
OPTIONAL_STAGES: tuple[Stage, ...] = (Stage.EVALUATION, Stage.COLLECTION_MEMORY)


def runtime_keys_for(*, hierarchical: bool = False, evaluate: bool = True) -> list[str]:
    """Runtime keys expected for a run of the given shape."""
    stages = list(CORE_STAGES)
    if hierarchical:
        stages.append(Stage.COLLECTION_MEMORY)
    if evaluate:
        stages.append(Stage.EVALUATION)
    return [s.value for s in stages]
