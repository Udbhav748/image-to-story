"""Command-line interface - the single entry point for the production system.

Usage:
    python -m image_story <paths> [options]
    image-story <paths> [options]        # after `pip install -e .`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config.settings import load_config_from_yaml, settings
from .domain.schemas import PipelineConfig
from .experiments.runners import find_images, run_experiment
from .observability.logging import configure_logging

DESCRIPTION = "Image -> Story: grounded creative storytelling from images"

MODES = ("fast", "standard", "full", "baseline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image-story", description=DESCRIPTION)
    parser.add_argument("paths", nargs="+", help="Image files or directories")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="standard",
        help="Pipeline mode preset (default: standard)",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="One story across all images, in filename order",
    )
    parser.add_argument(
        "--collection",
        action="store_true",
        help="Use hierarchical collection memory instead of flat retrieval",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--config",
        help="Path to a PipelineConfig YAML file (overrides --mode)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device (default: cpu)"
    )
    parser.add_argument(
        "--output-dir",
        default=settings.artifacts_dir,
        help="Directory for artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--name",
        default="run",
        help="Experiment name recorded in the summary",
    )
    parser.add_argument(
        "--summary",
        help="Write the aggregated JSON summary to this path",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> PipelineConfig:
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        from .config.settings import get_pipeline_config

        config = get_pipeline_config(args.mode)
    config.seed = args.seed
    config.device = args.device
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    settings.device = args.device
    settings.apply_env()

    images = find_images(args.paths)
    if not images:
        print("No images found")
        return 1

    if args.multi and len(images) < 2:
        print("--multi requires at least 2 images")
        return 1

    config = _resolve_config(args)
    print(f"Image -> Story | mode={config.mode} device={config.device} seed={config.seed}")
    print(f"Images: {len(images)}")

    summary = run_experiment(
        images,
        name=args.name,
        config=config,
        multi=args.multi,
        use_collection_memory=args.collection,
        evaluate=not args.no_eval,
        save_artifacts_dir=args.output_dir,
    )

    result = summary.to_dict()
    print(json.dumps(result["metrics"], indent=2))
    print(
        f"Successful: {result['successful']}/{result['total']}"
        f"  grounding_pass={result['grounding_pass']}"
    )
    for failure in result["failures"]:
        print(f"  FAILED {failure['image']}: {failure['error']}")

    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Summary written to {path}")

    return 0 if summary.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
