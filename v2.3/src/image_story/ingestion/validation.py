"""Image validation utilities."""
from PIL import Image
from typing import Any
import os

from ..domain.schemas import ImageRecord
from ..domain.enums import ImageValidationStatus


class ImageValidator:
    """Validates images before processing."""
    
    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    MAX_DIMENSION = 4096
    MIN_DIMENSION = 32
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self, max_dimension: int = 4096, min_dimension: int = 32, max_file_size: int = 50 * 1024 * 1024):
        self.max_dimension = max_dimension
        self.min_dimension = min_dimension
        self.max_file_size = max_file_size
    
    def validate(self, image_record: ImageRecord) -> tuple[str, str]:
        """
        Validate an image file.
        Returns (status, error_message) where status is one of ImageValidationStatus.
        """
        path = image_record.path
        
        # Check file exists
        if not os.path.exists(path):
            return "invalid", f"File not found: {path}"
        
        # Check file size
        try:
            file_size = os.path.getsize(image_record.path)
            image_record.file_size = file_size
            if file_size == 0:
                return "invalid", "Zero-byte file"
            if file_size > self.max_file_size:
                return "invalid", f"File size {file_size} exceeds maximum {self.max_file_size}"
        except OSError as e:
            return "invalid", f"Cannot read file: {e}"
        
        # Check extension
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            return "invalid", f"Unsupported format: {ext}"
        
        # Try to open and validate image
        try:
            with Image.open(path) as img:
                img.verify()  # Verify it's a valid image
                image_record.width, image_record.height = img.size
                image_record.mime_type = Image.MIME.get(img.format, "")
                
                # Check dimensions
                width, height = img.size
                if width > self.max_dimension or height > self.max_dimension:
                    return "invalid", f"Image dimensions {width}x{height} exceed maximum {self.max_dimension}"
                if width < self.min_dimension or height < self.min_dimension:
                    return "invalid", f"Image dimensions {width}x{height} below minimum {self.min_dimension}"
                    
        except Exception as e:
            return "invalid", f"Corrupt or unreadable image: {e}"
        
        return "valid", ""


class ImageHasher:
    """Computes content and perceptual hashes for images."""
    
    def __init__(self, perceptual_hash_size: int = 16):
        self.perceptual_hash_size = perceptual_hash_size
    
    def compute_content_hash(self, path: str) -> str:
        """Compute SHA256 hash of file content."""
        import hashlib
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def compute_perceptual_hash(self, path: str) -> str:
        """Compute perceptual hash (pHash) for near-duplicate detection."""
        try:
            from PIL import Image
            with Image.open(path) as img:
                # Convert to grayscale and resize
                img = img.convert("L").resize((self.perceptual_hash_size * 2, self.perceptual_hash_size))
                # Compute DCT-based hash
                import numpy as np
                pixels = np.array(img, dtype=np.float32)
                # Simple DCT-based perceptual hash
                from scipy.fftpack import dct
                dct_coeffs = dct(dct(pixels.T, norm='ortho').T, norm='ortho')
                # Take top-left coefficients
                low_freq = dct_coeffs[:self.perceptual_hash_size, :self.perceptual_hash_size]
                median = np.median(low_freq)
                bits = (low_freq > median).flatten()
                # Convert to hex string
                hash_str = ''.join(['1' if b else '0' for b in bits])
                return hex(int(hash_str, 2))[2:].zfill(self.perceptual_hash_size * self.perceptual_hash_size // 4)
        except Exception:
            # Fallback: simple hash based on resized image
            return self._simple_perceptual_hash(path)
    
    def _simple_perceptual_hash(self, path: str) -> str:
        """Fallback perceptual hash without scipy."""
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
    
    def register(self, image_id: str, content_hash: str, perceptual_hash: str) -> tuple[str | None, list[tuple[str, int]]]:
        """
        Register an image and check for duplicates.
        Returns (duplicate_of, similar_to).
        """
        # Check exact duplicate
        duplicate_of = self.check_exact_duplicate(content_hash)
        
        # Check near duplicates
        similar_to = self.check_near_duplicates(perceptual_hash, exclude_id="")
        
        # Register
        self._content_hash_map[content_hash] = ""  # placeholder
        self._perceptual_hashes[""] = perceptual_hash
        
        return duplicate_of, similar_to
    
    def register_image(self, image_id: str, content_hash: str, perceptual_hash: str) -> None:
        """Register an image after processing."""
        self._content_hash_map[content_hash] = image_id
        self._perceptual_hashes[image_id] = perceptual_hash
    
    def remove_image(self, image_id: str) -> None:
        """Remove an image from the deduplicator."""
        pass  # Implementation if needed