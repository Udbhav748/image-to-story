"""Image hashing utilities (canonical `ImageHasher`)."""
import hashlib

import numpy as np
from PIL import Image

from ..domain.schemas import ImageRecord


class ImageHasher:
    """Computes content and perceptual hashes for images."""

    def __init__(self, perceptual_hash_size: int = 16):
        self.perceptual_hash_size = perceptual_hash_size

    def compute_content_hash(self, path: str) -> str:
        """Compute SHA256 hash of file content."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def compute_perceptual_hash(self, path: str) -> str:
        """Compute perceptual hash (pHash) for near-duplicate detection."""
        try:
            with Image.open(path) as img:
                # Use imagehash library for perceptual hash
                import imagehash

                phash = imagehash.phash(img, hash_size=self.perceptual_hash_size)
                return str(phash)
        except Exception:
            # Fallback: simple hash based on resized image
            return self._simple_perceptual_hash(path)

    def _simple_perceptual_hash(self, path: str) -> str:
        """Fallback perceptual hash without imagehash library."""
        try:
            with Image.open(path) as img:
                img = img.convert("L").resize((32, 32))
                pixels = np.array(img, dtype=np.float64).flatten().tolist()
                avg = sum(pixels) / len(pixels)
                bits = "".join("1" if p > avg else "0" for p in pixels)
                return hex(int(bits, 2))[2:].zfill(16)
        except Exception:
            return ""

    def compute_hashes(self, image_record: ImageRecord) -> tuple[str, str]:
        """Compute both content and perceptual hashes for an image record."""
        content_hash = self.compute_content_hash(image_record.path)
        perceptual_hash = self.compute_perceptual_hash(image_record.path)
        return content_hash, perceptual_hash
