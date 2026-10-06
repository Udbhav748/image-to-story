"""Image hashing utilities."""
import hashlib
from PIL import Image
import numpy as np
from typing import Any
import imagehash

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
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)
                bits = ''.join('1' if p > avg else '0' for p in pixels)
                return hex(int(bits, 2))[2:].zfill(16)
        except Exception:
            return ""
    
    def compute_hashes(self, image_record: "ImageRecord") -> tuple[str, str]:
        """Compute both content and perceptual hashes for an image record."""
        content_hash = self.compute_content_hash(image_record.path)
        perceptual_hash = self.compute_perceptual_hash(image_record.path)
        return content_hash, perceptual_hash


class Deduplicator:
    """Handles exact and near-duplicate detection."""
    
    def __init__(self, perceptual_threshold: int = 10):
        self.perceptual_threshold = perceptual_threshold  # Hamming distance threshold
        self._content_hash_map: dict[str, str] = {}  # content_hash -> image_id
        self._perceptual_hashes: dict[str, str] = {}  # image_id -> perceptual_hash
    
    def check_exact_duplicate(self, content_hash: str) -> str | None:
        """Check for exact duplicate by content hash."""
        return self._content_hash_map.get(content_hash)
    
    def check_near_duplicates(self, perceptual_hash: str, exclude_id: str = "") -> list[tuple[str, int]]:
        """Find near-duplicates using perceptual hash.
        Returns list of (image_id, hamming_distance) sorted by distance."""
        similar = []
        for img_id, phash in self._perceptual_hashes.items():
            if img_id == exclude_id:
                continue
            # Compute Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(perceptual_hash, phash))
            if distance <= self.perceptual_threshold:
                similar.append((img_id, distance))
        similar.sort(key=lambda x: x[1])
        return similar
    
    def register_image(self, image_id: str, content_hash: str, perceptual_hash: str) -> tuple[str | None, list[tuple[str, int]]]:
        """
        Register an image and check for duplicates.
        Returns (duplicate_of, similar_to).
        """
        # Check exact duplicate
        duplicate_of = self.check_exact_duplicate(content_hash)
        
        # Check near duplicates
        similar_to = self.check_near_duplicates(perceptual_hash, exclude_id="")
        
        # Register
        self._content_hash_map[content_hash] = ""
        self._perceptual_hashes[image_id] = perceptual_hash
        
        return duplicate_of, similar_to
    
    def register_image(self, image_id: str, content_hash: str, perceptual_hash: str) -> None:
        """Register an image after processing."""
        self._content_hash_map[content_hash] = image_id
        self._perceptual_hashes[image_id] = perceptual_hash
    
    def remove_image(self, image_id: str) -> None:
        """Remove an image from the deduplicator."""
        pass  # Implementation if needed