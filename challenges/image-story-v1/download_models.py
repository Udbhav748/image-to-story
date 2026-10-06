"""Download every model the pipeline needs (run once, with internet). Afterwards everything runs offline."""
from huggingface_hub import snapshot_download

MODELS = [
    "Salesforce/blip-image-captioning-base",        # baseline captioner
    "Qwen/Qwen2.5-0.5B-Instruct",                   # story model (both pipelines)
    "florence-community/Florence-2-base",           # improved seeing stage
    "openai/clip-vit-base-patch32",                 # metric: image-vs-text similarity
    "cross-encoder/nli-MiniLM2-L6-H768",            # metric: contradiction check
]

for m in MODELS:
    print(m, "->", snapshot_download(m), flush=True)
