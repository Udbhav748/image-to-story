"""Artifact persistence.

Single writer for `PipelineArtifacts`. Previously `CollectionPipeline` imported
`save_artifacts` from the top-level `main_v2.py` script, which broke the moment
the entry point moved; it now lives here in the package.
"""
from __future__ import annotations

import json
import os

from ..domain.schemas import PipelineArtifacts


def save_artifacts(artifacts: PipelineArtifacts, output_dir: str) -> str:
    """Write artifacts, story, evaluation and context to `output_dir`.

    Returns the directory written to.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"{artifacts.run_id}_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts.to_dict(), f, indent=2, ensure_ascii=False)

    if artifacts.story_draft:
        with open(os.path.join(output_dir, f"{artifacts.run_id}_story.txt"), "w", encoding="utf-8") as f:
            f.write(artifacts.story_draft.text)

    if artifacts.evaluation:
        with open(os.path.join(output_dir, f"{artifacts.run_id}_evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(artifacts.evaluation.to_dict(), f, indent=2, ensure_ascii=False)

    if artifacts.context:
        with open(os.path.join(output_dir, f"{artifacts.run_id}_context.txt"), "w", encoding="utf-8") as f:
            f.write(artifacts.context)

    return output_dir
