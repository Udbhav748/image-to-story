"""Base vision model interface."""
from abc import ABC, abstractmethod
from typing import Any
from PIL import Image

from ..domain.schemas import VisualObservations


class VisionModel(ABC):
    """Abstract base class for vision models."""
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free memory."""
        pass
    
    @abstractmethod
    def analyze(self, image: Image.Image, frame_id: int = 0) -> VisualObservations:
        """Analyze an image and return structured observations."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        pass


class VisionModelRegistry:
    """Registry for vision models."""
    
    _models: dict[str, VisionModel] = {}
    
    @classmethod
    def register(cls, name: str, model: VisionModel) -> None:
        cls._models[name] = model
    
    @classmethod
    def get(cls, name: str) -> VisionModel | None:
        return cls._models.get(name)
    
    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._models.keys())
    
    @classmethod
    def unload_all(cls) -> None:
        for model in cls._models.values():
            if model.is_loaded:
                model.unload()