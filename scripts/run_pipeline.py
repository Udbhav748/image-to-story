#!/usr/bin/env python3
"""Run the pipeline over an image set and print the aggregate metrics.

Thin wrapper around `image_story.cli`. For the full option set:

    python -m image_story --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_story.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
