"""Image validation.

Canonical `ImageValidator` only. Hashing lives in `ingestion.hashing` and
deduplication in `ingestion.dedup`; this module used to carry private copies of
both, which is why those were removed.
"""
import os

from PIL import Image

from ..domain.enums import ImageValidationStatus
from ..domain.schemas import ImageRecord


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

        return ImageValidationStatus.VALID.value, ""
