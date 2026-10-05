"""Embedding model interface and implementations."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from ..domain.enums import EmbeddingModel


class EmbeddingModelInterface(ABC):
    """Abstract interface for embedding models."""
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass
    
    @abstractmethod
    def load(self) -> None:
        pass
    
    @abstractmethod
    def unload(self) -> None:
        pass
    
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        pass
    
    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to normalized embedding."""
        pass


class SentenceTransformerEmbedding(EmbeddingModelInterface):
    """SentenceTransformer-based embedding model."""
    
    def __init__(
        self,
        model_id: str = EmbeddingModel.MINILM_L6.value,
        device: str = "cpu",
    ):
        self._model_id = model_id
        self._device = device
        self._model = None
        self._embedding_dim = 384  # MiniLM-L6-v2 dimension
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def load(self) -> None:
        if self.is_loaded:
            return
        
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_id, device=self._device)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
    
    def unload(self) -> None:
        self._model = None
    
    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def create_embedding_model(
    model_variant: str = "minilm_l6",
    device: str = "cpu",
) -> EmbeddingModelInterface:
    """Factory function to create embedding model."""
    model_map = {
        "minilm_l6": EmbeddingModel.MINILM_L6.value,
        "mpnet_base": EmbeddingModel.MPNET_BASE.value,
        "e5_small": EmbeddingModel.E5_SMALL.value,
    }
    model_id = model_map.get(model_variant, model_variant)
    return SentenceTransformerEmbedding(model_id=model_id, device=device)