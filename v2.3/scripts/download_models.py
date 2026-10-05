#!/usr/bin/env python3
"""Download all models needed for Image -> Story V2 pipeline."""
from huggingface_hub import snapshot_download
import os

# Model list for V2
MODELS = [
    # Baseline/Original models
    "Salesforce/blip-image-captioning-base",        # baseline captioner
    "Qwen/Qwen2.5-0.5B-Instruct",                   # story model
    "florence-community/Florence-2-base",           # improved seeing stage
    "openai/clip-vit-base-patch32",                 # metric: image-vs-text similarity
    "cross-encoder/nli-MiniLM2-L6-H768",            # metric: contradiction check
    
    # V2 additions
    "IDEA-Research/grounding-dino-base",            # open-vocabulary object grounding
    "sentence-transformers/all-MiniLM-L6-v2",       # embeddings for FAISS
    "microsoft/trocr-base-printed",                 # OCR recognition
    "facebook/detr-resnet-50",                      # OCR text detection
    
    # Optional: spaCy for claim extraction
    # "en_core_web_sm",  # downloaded via: python -m spacy download en_core_web_sm
]

# Optional larger models
OPTIONAL_MODELS = {
    "grounding_dino_tiny": "IDEA-Research/grounding-dino-tiny",
    "embedding_mpnet": "sentence-transformers/all-mpnet-base-v2",
    "embedding_e5": "intfloat/e5-small-v2",
    "ocr_handwritten": "microsoft/trocr-base-handwritten",
    "qwen_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
}


def download_model(model_id: str) -> str:
    """Download a single model."""
    print(f"Downloading {model_id}...")
    try:
        path = snapshot_download(model_id)
        print(f"  -> {path}")
        return path
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""


def main():
    print("Downloading models for Image -> Story V2...")
    print("=" * 50)
    
    # Core models
    print("\nCore models:")
    for model in MODELS:
        download_model(model)
    
    # Ask about optional models
    print("\nOptional models (press Enter to skip):")
    for name, model_id in OPTIONAL_MODELS.items():
        response = input(f"Download {name} ({model_id})? [y/N]: ").strip().lower()
        if response == 'y':
            download_model(model_id)
    
    # spaCy model
    print("\nspaCy model (for claim extraction):")
    response = input("Download en_core_web_sm? [y/N]: ").strip().lower()
    if response == 'y':
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("\nDone! All models cached for offline use.")


if __name__ == "__main__":
    import sys
    main()