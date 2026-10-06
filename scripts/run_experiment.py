#!/usr/bin/env python3
"""Run a named experiment and write its summary JSON.

Examples:
    python scripts/run_experiment.py benchmarks/challenge-8-images --name v2-standard
    python scripts/run_experiment.py benchmarks/challenge-8-images --collection \\
        --name v2-collection --summary artifacts/experiments/v2-collection.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_story.config.settings import get_pipeline_config, settings  # noqa: E402
from image_story.experiments.runners import find_images, run_experiment  # noqa: E402
from image_story.observability.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a named Image -> Story experiment")
    parser.add_argument("paths", nargs="+", help="Image files or directories")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--mode", default="standard", choices=["fast", "standard", "full", "baseline"])
    parser.add_argument("--multi", action="store_true", help="One story across all images")
    parser.add_argument("--collection", action="store_true", help="Use hierarchical collection memory")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", default=settings.artifacts_dir)
    parser.add_argument(
        "--summary",
        default=None,
        help="Summary JSON path (default: <output-dir>/experiments/<name>.json)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()

    images = find_images(args.paths)
    if not images:
        print("No images found")
        return 1

    config = get_pipeline_config(args.mode)
    config.seed = args.seed
    config.device = args.device

    summary = run_experiment(
        images,
        name=args.name,
        config=config,
        multi=args.multi,
        use_collection_memory=args.collection,
        evaluate=not args.no_eval,
        save_artifacts_dir=args.output_dir,
    )

    summary_path = args.summary or str(Path(args.output_dir) / "experiments" / f"{args.name}.json")
    written = summary.write(summary_path)

    print(json.dumps(summary.to_dict(), indent=2))
    print(f"\nSummary written to {written}")
    return 0 if summary.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
