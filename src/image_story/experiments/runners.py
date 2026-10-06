"""Experiment runners: repeat a configured run over an image set and aggregate.

This is what `scripts/run_experiment.py` drives. Aggregation lives in the package
so an experiment is reproducible from a config plus an image list, rather than
from a bespoke script.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from ..config.settings import get_pipeline_config
from ..domain.schemas import PipelineArtifacts, PipelineConfig
from ..experiments.artifacts import save_artifacts
from ..experiments.manifests import build_manifest
from ..observability.logging import get_logger

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

logger = get_logger("image_story.experiments.runners")

#: Metrics averaged across images in a summary.
SUMMARY_METRICS = (
    "grounding_score",
    "clip_image_story_mean",
    "nli_contra_mean",
    "repetition_rate",
    "claim_support_rate",
    "continuity_score",
    "narrative_coherence",
)


@dataclass
class ExperimentResult:
    """Outcome for one image in an experiment run."""

    image_path: str
    run_id: str
    success: bool
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    story: str = ""
    artifacts_dir: str = ""


@dataclass
class ExperimentSummary:
    """Aggregate of an experiment run. Means cover successful images only."""

    name: str
    config: dict[str, Any]
    results: list[ExperimentResult] = field(default_factory=list)

    @property
    def successful(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> list[ExperimentResult]:
        return [r for r in self.results if not r.success]

    def mean(self, key: str) -> float:
        values = [float(r.metrics[key]) for r in self.successful if r.metrics.get(key) is not None]
        return mean(values) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        successful = self.successful
        return {
            "name": self.name,
            "config": self.config,
            "total": len(self.results),
            "successful": len(successful),
            "failed": len(self.failed),
            "metrics": {key: self.mean(key) for key in SUMMARY_METRICS},
            "grounding_pass": sum(
                1 for r in successful if r.metrics.get("grounding_pass")
            ),
            "failures": [
                {"image": r.image_path, "error": r.error} for r in self.failed
            ],
        }

    def write(self, path: str | Path) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)


def find_images(paths: Sequence[str]) -> list[str]:
    """Expand files and directories to image files, sorted by file name."""
    found: dict[str, Path] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = [
                f for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ]
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files = [path]
        else:
            files = []
        for f in files:
            found[str(f)] = f
    return [str(f) for f in sorted(found.values(), key=lambda f: f.name.lower())]


def _to_result(artifacts: PipelineArtifacts, image_path: str, artifacts_dir: str | Path) -> ExperimentResult:
    metrics = artifacts.evaluation.to_dict() if artifacts.evaluation else {}
    return ExperimentResult(
        image_path=image_path,
        run_id=artifacts.run_id,
        success=True,
        metrics=metrics,
        story=artifacts.story_draft.text if artifacts.story_draft else "",
        artifacts_dir=str(artifacts_dir),
    )


def run_experiment(
    image_paths: Sequence[str],
    *,
    name: str = "experiment",
    mode: str = "standard",
    config: PipelineConfig | None = None,
    multi: bool = False,
    use_collection_memory: bool = False,
    evaluate: bool = True,
    save_artifacts_dir: str | Path | None = None,
    repo_root: str | Path = ".",
) -> ExperimentSummary:
    """Run the pipeline over `image_paths` and aggregate the results.

    With `multi`, all images are processed as one ordered sequence. Otherwise
    each image is an independent run. A failure on one image is recorded and the
    run continues.
    """
    from ..pipeline.orchestrator import create_orchestrator

    config = config or get_pipeline_config(mode)
    images = list(image_paths)
    summary = ExperimentSummary(name=name, config=config.to_dict())
    logger.info(
        "Running %s over %d image(s), mode=%s, multi=%s, collection_memory=%s",
        name, len(images), config.mode, multi, use_collection_memory,
    )

    orchestrator = create_orchestrator(config)
    try:
        if multi:
            if len(images) < 2:
                raise ValueError("multi mode requires at least 2 images")
            batches: list[list[str]] = [images]
        else:
            batches = [[image] for image in images]

        for batch in batches:
            label = batch[0] if len(batch) == 1 else f"{len(batch)}-image sequence"
            out_dir = Path(save_artifacts_dir) / Path(batch[0]).stem if save_artifacts_dir else None
            try:
                if use_collection_memory:
                    artifacts = orchestrator.run_collection(batch, evaluate=evaluate)
                elif len(batch) > 1:
                    artifacts = orchestrator.run_multi_image(batch, evaluate=evaluate)
                else:
                    artifacts = orchestrator.run_single_image(batch[0], evaluate=evaluate)

                if out_dir is not None:
                    save_artifacts(artifacts, str(out_dir))
                summary.results.append(_to_result(artifacts, label, out_dir or ""))
            except Exception as e:  # noqa: BLE001 - one bad input must not stop the run
                logger.exception("Failed on %s: %s", label, e)
                summary.results.append(
                    ExperimentResult(image_path=label, run_id="", success=False, error=str(e))
                )
    finally:
        orchestrator.cleanup()

    manifest = build_manifest(
        config,
        dataset_name=name,
        image_paths=images,
        repo_root=repo_root,
    )
    logger.info(
        "%s: %d/%d succeeded (manifest run_id=%s, commit=%s)",
        name, len(summary.successful), len(summary.results),
        manifest.run_id, manifest.git_commit[:8] or "unknown",
    )
    return summary
