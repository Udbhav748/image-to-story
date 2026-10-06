#!/usr/bin/env python3
"""Download every model the Image -> Story pipeline needs, for offline use.

The Challenge V1 models (BLIP baseline) are not required by the production
pipeline; `challenges/image-story-v1/download_models.py` fetches those.
"""
import subprocess
import sys

from huggingface_hub import snapshot_download

MODELS = [
    # Perception
    "florence-community/Florence-2-base",           # vision perception
    "IDEA-Research/grounding-dino-base",            # open-vocabulary grounding
    # Memory and retrieval
    "sentence-transformers/all-MiniLM-L6-v2",       # embeddings for FAISS
    # Generation
    "Qwen/Qwen2.5-0.5B-Instruct",                   # story writer
    # Evaluation
    "openai/clip-vit-base-patch32",                 # image-vs-text similarity
    "cross-encoder/nli-MiniLM2-L6-H768",            # contradiction detection
    # OCR (only loaded in `full` mode)
    "microsoft/trocr-base-printed",                 # OCR recognition
    "facebook/detr-resnet-50",                      # OCR text detection
]

OPTIONAL_MODELS = {
    "grounding_dino_tiny": "IDEA-Research/grounding-dino-tiny",
    "embedding_mpnet": "sentence-transformers/all-mpnet-base-v2",
    "embedding_e5": "intfloat/e5-small-v2",
    "ocr_handwritten": "microsoft/trocr-base-handwritten",
    "qwen_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
}

SPACY_MODEL = "en_core_web_sm"


def download_model(model_id: str) -> str:
    """Download one model, returning its cache path ("" on failure)."""
    print(f"Downloading {model_id}...")
    try:
        path = snapshot_download(model_id)
        print(f"  -> {path}")
        return path
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"  ERROR: {e}")
        return ""


def main() -> int:
    interactive = "--yes" not in sys.argv

    print("Downloading models for Image -> Story...")
    print("=" * 50)

    for model in MODELS:
        download_model(model)

    if interactive:
        print("\nOptional models (press Enter to skip):")
        for name, model_id in OPTIONAL_MODELS.items():
            if input(f"Download {name} ({model_id})? [y/N]: ").strip().lower() == "y":
                download_model(model_id)

        print("\nspaCy model (needed for claim extraction):")
        if input(f"Download {SPACY_MODEL}? [y/N]: ").strip().lower() == "y":
            subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=False)
    else:
        print("\nSkipping optional models and spaCy (--yes).")

    print("\nDone. Models are cached for offline use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
