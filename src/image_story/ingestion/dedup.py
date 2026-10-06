"""Deduplication for ingested images (canonical `Deduplicator`).

The previous revision defined `register_image` twice in `ingestion/hashing.py`;
the second definition silently shadowed the first, so the duplicate-detection
result was discarded. There is now exactly one registration method, which
returns `(duplicate_of, similar_to)`.
"""
from __future__ import annotations


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

        Returns list of (image_id, hamming_distance) sorted by distance.
        """
        similar = []
        for img_id, phash in self._perceptual_hashes.items():
            if img_id == exclude_id:
                continue
            # Compute Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(perceptual_hash, phash, strict=False))
            if distance <= self.perceptual_threshold:
                similar.append((img_id, distance))
        similar.sort(key=lambda x: x[1])
        return similar

    def register_image(
        self, image_id: str, content_hash: str, perceptual_hash: str
    ) -> tuple[str | None, list[tuple[str, int]]]:
        """Register an image and report duplicates found before registration.

        Returns (duplicate_of, similar_to). The image is always registered so
        that subsequent images are compared against it.
        """
        duplicate_of = self.check_exact_duplicate(content_hash)
        similar_to = self.check_near_duplicates(perceptual_hash, exclude_id=image_id)

        self._content_hash_map[content_hash] = image_id
        self._perceptual_hashes[image_id] = perceptual_hash

        return duplicate_of, similar_to

    def remove_image(self, image_id: str) -> None:
        """Remove an image from the deduplicator."""
        self._perceptual_hashes.pop(image_id, None)
        for content_hash, mapped_id in list(self._content_hash_map.items()):
            if mapped_id == image_id:
                del self._content_hash_map[content_hash]
