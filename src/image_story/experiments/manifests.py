"""Experiment manifests.

`ExperimentManifest` itself is a domain contract (`domain/schemas.py`); this
module builds and populates one so that every run records how it was produced.
"""
from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..domain.schemas import ExperimentManifest, PipelineConfig


def git_commit(repo_root: str | Path = ".") -> str:
    """Current git commit, or an empty string outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def dataset_hash(paths: Iterable[str | Path]) -> str:
    """Stable hash of an image set: hashes the sorted path+size pairs.

    Deliberately cheap (no image decoding) so that manifest creation does not
    dominate run time.
    """
    digest = hashlib.sha256()
    for path in sorted(str(p) for p in paths):
        digest.update(path.encode("utf-8"))
        try:
            digest.update(str(Path(path).stat().st_size).encode("utf-8"))
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()


def build_manifest(
    config: PipelineConfig,
    *,
    run_id: str | None = None,
    dataset_name: str = "",
    image_paths: Iterable[str | Path] = (),
    vision_model: str = "",
    language_model: str = "",
    embedding_model: str = "",
    repo_root: str | Path = ".",
) -> ExperimentManifest:
    """Create a populated manifest for a run."""
    manifest = ExperimentManifest(
        git_commit=git_commit(repo_root),
        vision_model=vision_model,
        language_model=language_model,
        embedding_model=embedding_model,
        dataset=dataset_name,
        seed=config.seed,
        device=config.device,
        config=config.to_dict(),
    )
    if run_id:
        manifest.run_id = run_id
    paths = list(image_paths)
    if paths:
        manifest.dataset_hash = dataset_hash(paths)
    return manifest


def manifest_from_dict(data: dict[str, Any]) -> ExperimentManifest:
    """Rebuild a manifest from `to_dict()` output."""
    return ExperimentManifest(
        git_commit=data.get("git_commit", ""),
        vision_model=data.get("vision_model", ""),
        language_model=data.get("language_model", ""),
        embedding_model=data.get("embedding_model", ""),
        dataset=data.get("dataset", ""),
        dataset_hash=data.get("dataset_hash", ""),
        seed=data.get("seed", 0),
        device=data.get("device", "cpu"),
        config=data.get("config", {}),
        timestamp=data.get("timestamp", ""),
        run_id=data.get("run_id", ""),
    )
