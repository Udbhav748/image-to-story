#!/usr/bin/env python3
"""Aggregate `*_evaluation.json` artifact files into a comparison table.

Reads the artifact directories written by `scripts/run_experiment.py`
(`<artifacts>/experiments/<name>/`) or directly by the CLI
(`<artifacts>/<run_id>/`), and prints per-directory means.

Example:
    python scripts/build_report.py artifacts/experiments
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

METRICS = (
    "grounding_score",
    "clip_image_story_mean",
    "nli_contra_mean",
    "claim_support_rate",
    "continuity_score",
    "narrative_coherence",
)


def parse_results(dir_path: Path) -> list[dict]:
    """Load every `*_evaluation.json` under `dir_path` (recursively)."""
    results = []
    for path in sorted(dir_path.rglob("*_evaluation.json")):
        with open(path, encoding="utf-8") as fp:
            results.append(json.load(fp))
    return results


def summarize(results: list[dict]) -> dict[str, float]:
    def avg(key: str) -> float:
        values = [r[key] for r in results if r.get(key) is not None]
        return mean(values) if values else 0.0

    summary = {key: avg(key) for key in METRICS}
    summary["runs"] = len(results)
    summary["grounding_pass"] = sum(1 for r in results if r.get("grounding_pass"))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate evaluation artifacts into a report")
    parser.add_argument(
        "root",
        nargs="?",
        default="artifacts",
        help="Directory to scan for *_evaluation.json (default: artifacts)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"No such directory: {root}")
        return 1

    rows = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        results = parse_results(child)
        if results:
            rows.append((child.name, summarize(results)))

    direct = parse_results(root)
    if direct:
        rows.append((root.name, summarize(direct)))

    if not rows:
        print(f"No evaluation artifacts found under {root}")
        return 1

    width = max(len(name) for name, _ in rows) + 2
    print(f"{'run'.ljust(width)}{'runs':>6}{'ground':>10}{'clip':>8}{'nli':>8}{'claims':>9}{'cont':>8}{'narr':>8}")
    print("-" * (width + 47))
    for name, summary in rows:
        print(
            f"{name.ljust(width)}{summary['runs']:>6}"
            f"{summary['grounding_score']:>10.3f}"
            f"{summary['clip_image_story_mean']:>8.3f}"
            f"{summary['nli_contra_mean']:>8.3f}"
            f"{summary['claim_support_rate']:>9.3f}"
            f"{summary['continuity_score']:>8.3f}"
            f"{summary['narrative_coherence']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
