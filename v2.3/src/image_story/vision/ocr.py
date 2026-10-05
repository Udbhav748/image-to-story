"""OCR vision model adapter for text detection and recognition."""
import os
import time
import uuid
from typing import Any, Optional
from PIL import Image
import torch

from ..domain.schemas import (
    VisualObservations,
    EvidenceRecord,
    EvidenceType,
    SourceModel,
    BoundingBox,
    InformationClass,
)
from ..domain.exceptions import OCRError, ModelLoadError
from .base import VisionModel


class OCRModel(VisionModel):
    """OCR model for text detection and recognition in images."""
    
    def __init__(
        self,
        model_id: str = "microsoft/trocr-base-printed",
        device: str = "cpu",
        detection_model_id: str = "facebook/detr-resnet-50",
        enabled: bool = True,
    ):
        self._model_id = model_id
        self._detection_model_id = detection_model_id
        self._device = device
        self._detection_model = None
        self._detection_processor = None
        self._recognition_model = None
        self._recognition_processor = None
        self._load_time_s = 0.0
        self._enabled = enabled
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def is_loaded(self) -> bool:
        return self._recognition_model is not None
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False
    
    def load(self) -> None:
        if self.is_loaded:
            return
        
        if not self._enabled:
            raise OCRError("OCR is disabled")
        
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        
        t0 = time.perf_counter()
        try:
            from transformers import (
                AutoProcessor,
                AutoModelForObjectDetection,
                VisionEncoderDecoderModel,
                TrOCRProcessor,
            )
            
            self._detection_processor = AutoProcessor.from_pretrained(self._detection_model_id)
            self._detection_model = AutoModelForObjectDetection.from_pretrained(
                self._detection_model_id, torch_dtype=torch.float32
            )
            self._detection_model.eval()
            self._detection_model.to(self._device)
            
            self._recognition_processor = TrOCRProcessor.from_pretrained(self._model_id)
            self._recognition_model = VisionEncoderDecoderModel.from_pretrained(
                self._model_id, torch_dtype=torch.float32
            )
            self._recognition_model.eval()
            self._recognition_model.to(self._device)
            
            self._load_time_s = round(time.perf_counter() - t0, 2)
        except Exception as e:
            raise ModelLoadError(f"Failed to load OCR models: {e}")
    
    def unload(self) -> None:
        self._detection_model = None
        self._detection_processor = None
        self._recognition_model = None
        self._recognition_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def detect_text_regions(self, image: Image.Image) -> list[BoundingBox]:
        """Detect text regions in the image."""
        if not self.is_loaded:
            self.load()
        
        inputs = self._detection_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._detection_model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]
        
        boxes = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            if self._detection_model.config.id2label[int(label)].lower() in ["text", "text region"]:
                boxes.append(BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                ))
        
        return boxes
    
    def recognize_text(self, image: Image.Image, bbox: BoundingBox) -> tuple[str, float]:
        """Recognize text in a specific region."""
        if not self.is_loaded:
            self.load()
        
        cropped = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        pixel_values = self._recognition_processor(cropped, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)
        
        with torch.no_grad():
            generated_ids = self._recognition_model.generate(pixel_values)
        
        text = self._recognition_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip(), 0.9
    
    def extract_text(self, image: Image.Image, frame_id: int = 0) -> list[EvidenceRecord]:
        """Extract all text from an image with bounding boxes."""
        if not self._enabled:
            return []
        
        if not self.is_loaded:
            self.load()
        
        text_regions = self.detect_text_regions(image)
        evidence_records = []
        
        for i, bbox in enumerate(text_regions):
            text, conf = self.recognize_text(image, bbox)
            if text:
                evidence_records.append(EvidenceRecord(
                    entity=f"text_{i}",
                    type=EvidenceType.OCR,
                    frame_id=frame_id,
                    bbox=bbox,
                    confidence=conf,
                    source=SourceModel.OCR,
                    evidence_text=f"OCR text: {text}",
                    provenance={
                        "model": "trocr",
                        "detection_model": "detr-resnet-50",
                        "text": text,
                    },
                    sequence_position=frame_id,
                    information_class=InformationClass.HARD_FACT,
                ))
        
        return evidence_records
    
    def analyze(self, image: Image.Image, frame_id: int = 0) -> VisualObservations:
        """Extract text and return as VisualObservations."""
        evidence_records = self.extract_text(image, frame_id)
        ocr_text = " ".join([e.evidence_text.replace("OCR text: ", "") for e in evidence_records])
        
        return VisualObservations(
            image_id=f"ocr_{frame_id}",
            frame_id=frame_id,
            ocr_text=ocr_text,
            evidence_records=evidence_records,
            models_used=["ocr"] if self._enabled else [],
        )
    
    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "detection_model_id": self._detection_model_id,
            "load_time_s": self._load_time_s,
            "device": self._device,
            "is_loaded": self.is_loaded,
            "enabled": self._enabled,
        }


def create_ocr(
    model_variant: str = "base",
    device: str = "cpu",
    enabled: bool = True,
) -> OCRModel:
    """Factory function to create OCR model."""
    model_map = {
        "base": "microsoft/trocr-base-printed",
        "large": "microsoft/trocr-large-printed",
        "handwritten": "microsoft/trocr-base-handwritten",
    }
    model_id = model_map.get(model_variant, model_variant)
    return OCRModel(model_id=model_id, device=device, enabled=enabled)