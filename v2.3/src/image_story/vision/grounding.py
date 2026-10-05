"""GroundingDINO vision model adapter for open-vocabulary object grounding."""
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
from ..domain.exceptions import GroundingDINOError, ModelLoadError
from .base import VisionModel


class GroundingDINOModel(VisionModel):
    """GroundingDINO model for open-vocabulary object detection and grounding."""
    
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cpu",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        self._model_id = model_id
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._model = None
        self._processor = None
        self._load_time_s = 0.0
        self._enabled = True
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
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
            raise GroundingDINOError("GroundingDINO is disabled")
        
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        
        t0 = time.perf_counter()
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self._model_id, torch_dtype=torch.float32
            )
            self._model.eval()
            self._model.to(self._device)
            self._load_time_s = round(time.perf_counter() - t0, 2)
        except Exception as e:
            raise ModelLoadError(f"Failed to load GroundingDINO: {e}")
    
    def unload(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        frame_id: int = 0,
    ) -> list[EvidenceRecord]:
        """Run open-vocabulary detection with a text prompt."""
        if not self.is_loaded:
            self.load()
        
        if not self._enabled:
            return []
        
        inputs = self._processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        
        evidence_records = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            bbox = BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            )
            
            evidence_records.append(EvidenceRecord(
                entity=label,
                type=EvidenceType.OBJECT,
                frame_id=frame_id,
                bbox=bbox,
                confidence=float(score),
                source=SourceModel.GROUNDING_DINO,
                evidence_text=f"Grounded detection: {label} (confidence: {score:.3f})",
                provenance={
                    "model": "grounding_dino",
                    "text_prompt": text_prompt,
                    "box_threshold": self._box_threshold,
                    "text_threshold": self._text_threshold,
                },
                sequence_position=frame_id,
                information_class=InformationClass.HARD_FACT,
            ))
        
        return evidence_records
    
    def detect_with_phrases(
        self,
        image: Image.Image,
        phrases: list[str],
        frame_id: int = 0,
    ) -> list[EvidenceRecord]:
        """Run detection with multiple phrases combined."""
        if not phrases:
            return []
        
        text_prompt = " . ".join(phrases) + " ."
        return self.detect(image, text_prompt, frame_id)
    
    def analyze(self, image: Image.Image, frame_id: int = 0) -> VisualObservations:
        """GroundingDINO doesn't produce a full VisualObservations on its own.
        Use detect() or detect_with_phrases() instead."""
        return VisualObservations(
            image_id=f"grounding_dino_{frame_id}",
            frame_id=frame_id,
            evidence_records=[],
            models_used=["grounding_dino"] if self._enabled else [],
        )
    
    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "load_time_s": self._load_time_s,
            "device": self._device,
            "is_loaded": self.is_loaded,
            "enabled": self._enabled,
            "box_threshold": self._box_threshold,
            "text_threshold": self._text_threshold,
        }


def create_grounding_dino(
    model_variant: str = "base",
    device: str = "cpu",
    enabled: bool = True,
) -> GroundingDINOModel:
    """Factory function to create GroundingDINO model."""
    model_map = {
        "base": "IDEA-Research/grounding-dino-base",
        "tiny": "IDEA-Research/grounding-dino-tiny",
    }
    model_id = model_map.get(model_variant, model_variant)
    model = GroundingDINOModel(model_id=model_id, device=device)
    if not enabled:
        model.disable()
    return model