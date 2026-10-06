"""FAISS vector store for semantic memory and retrieval."""
import os
import pickle
import uuid
from typing import Any
import numpy as np
import faiss

from ..domain.schemas import EvidenceRecord, RetrievedEvidence
from ..domain.exceptions import FAISSIndexError, RetrievalError
from .embeddings import EmbeddingModelInterface
from .hierarchical import IndexedEvidenceRecord, RetrievalFilter, create_retrieval_filter, apply_retrieval_filter


class FAISSVectorStore:
    """FAISS-based vector store for evidence retrieval with hierarchical support."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModelInterface,
        index_type: str = "flat_ip",
        device: str = "cpu",
    ):
        self._embedding_model = embedding_model
        self._index_type = index_type
        self._device = device
        self._index: faiss.Index | None = None
        # Legacy maps for backward compatibility
        self._evidence_map: dict[int, EvidenceRecord] = {}
        self._id_to_index: dict[str, int] = {}
        # Enhanced maps with full provenance
        self._indexed_evidence: dict[int, IndexedEvidenceRecord] = {}
        self._scene_to_indices: dict[str, list[int]] = {}
        self._entity_to_indices: dict[str, list[int]] = {}
        self._image_to_indices: dict[str, list[int]] = {}
        self._next_index = 0
        self._dimension = embedding_model.embedding_dim
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def count(self) -> int:
        return self._next_index
    
    def _create_index(self) -> faiss.Index:
        if self._index_type == "flat_ip":
            index = faiss.IndexFlatIP(self._dimension)
        elif self._index_type == "flat_l2":
            index = faiss.IndexFlatL2(self._dimension)
        elif self._index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self._dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
        else:
            raise FAISSIndexError(f"Unknown index type: {self._index_type}")
        
        if self._device == "cuda" and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass
        
        return index
    
    def initialize(self) -> None:
        """Initialize the FAISS index."""
        if self._index is None:
            self._index = self._create_index()
    
    def add_evidence(self, evidence: EvidenceRecord, text: str) -> int:
        """Add evidence to the store with its embedding text."""
        self.initialize()
        
        embedding = self._embedding_model.encode_single(text)
        embedding = embedding.reshape(1, -1)
        
        index_id = self._next_index
        self._index.add(embedding)
        self._evidence_map[index_id] = evidence
        self._id_to_index[evidence.id] = index_id
        self._next_index += 1
        
        return index_id
    
    def add_evidence_with_provenance(
        self,
        evidence: EvidenceRecord,
        text: str,
        image_id: str,
        scene_id: str,
        frame_id: int,
        collection_id: str,
    ) -> int:
        """Add evidence with full provenance for hierarchical retrieval."""
        self.initialize()
        
        embedding = self._embedding_model.encode_single(text)
        embedding = embedding.reshape(1, -1)
        
        index_id = self._next_index
        self._index.add(embedding)
        
        # Create indexed evidence with provenance
        indexed = IndexedEvidenceRecord(
            evidence=evidence,
            image_id=image_id,
            scene_id=scene_id,
            frame_id=frame_id,
            collection_id=collection_id,
            index_position=index_id,
        )
        
        # Update all maps
        self._evidence_map[index_id] = evidence
        self._indexed_evidence[index_id] = indexed
        self._id_to_index[evidence.id] = index_id
        
        # Update scene/entity/image indices
        self._scene_to_indices.setdefault(scene_id, []).append(index_id)
        self._entity_to_indices.setdefault(evidence.entity, []).append(index_id)
        self._image_to_indices.setdefault(image_id, []).append(index_id)
        
        self._next_index += 1
        
        return index_id
    
    def add_batch(self, evidence_list: list[EvidenceRecord], texts: list[str]) -> list[int]:
        """Add multiple evidence records at once (legacy)."""
        self.initialize()
        
        if not evidence_list:
            return []
        
        embeddings = self._embedding_model.encode(texts)
        index_ids = list(range(self._next_index, self._next_index + len(evidence_list)))
        
        self._index.add(embeddings)
        for i, (evidence, idx) in enumerate(zip(evidence_list, index_ids)):
            self._evidence_map[idx] = evidence
            self._id_to_index[evidence.id] = idx
        
        self._next_index += len(evidence_list)
        return index_ids
    
    def add_batch_with_provenance(
        self,
        evidence_list: list[EvidenceRecord],
        texts: list[str],
        image_ids: list[str],
        scene_ids: list[str],
        frame_ids: list[int],
        collection_id: str,
    ) -> list[int]:
        """Add multiple evidence records with full provenance."""
        self.initialize()
        
        if not evidence_list:
            return []
        
        embeddings = self._embedding_model.encode(texts)
        index_ids = list(range(self._next_index, self._next_index + len(evidence_list)))
        
        self._index.add(embeddings)
        for i, (evidence, idx) in enumerate(zip(evidence_list, index_ids)):
            indexed = IndexedEvidenceRecord(
                evidence=evidence,
                image_id=image_ids[i],
                scene_id=scene_ids[i],
                frame_id=frame_ids[i],
                collection_id=collection_id,
                index_position=idx,
            )
            self._evidence_map[idx] = evidence
            self._indexed_evidence[idx] = indexed
            self._id_to_index[evidence.id] = idx
            
            # Update indices
            self._scene_to_indices.setdefault(scene_ids[i], []).append(idx)
            self._entity_to_indices.setdefault(evidence.entity, []).append(idx)
            self._image_to_indices.setdefault(image_ids[i], []).append(idx)
        
        self._next_index += len(evidence_list)
        return index_ids
    
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        filter_fn: callable = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[RetrievedEvidence]:
        """Search for similar evidence with optional hierarchical filtering."""
        if self._index is None or self._next_index == 0:
            return []
        
        query_embedding = self._embedding_model.encode_single(query_text)
        query_embedding = query_embedding.reshape(1, -1)
        
        actual_k = min(top_k * 3, self._next_index)
        scores, indices = self._index.search(query_embedding, actual_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= self._next_index:
                continue
            
            evidence = self._evidence_map.get(idx)
            if evidence is None:
                continue
            
            # Apply legacy filter
            if filter_fn and not filter_fn(evidence):
                continue
            
            # Apply structured retrieval filter
            if retrieval_filter and not apply_retrieval_filter(evidence, retrieval_filter):
                continue
            
            # For hierarchical filtering, we need to check the indexed evidence
            if retrieval_filter and "scene_ids" in retrieval_filter:
                indexed = self._indexed_evidence.get(idx)
                if indexed and indexed.scene_id not in retrieval_filter["scene_ids"]:
                    continue
            
            results.append(RetrievedEvidence(
                record=evidence,
                semantic_similarity=float(score),
                rank_score=0.0,
                rank_factors={"semantic_similarity": float(score)},
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_scene(
        self,
        scene_id: str,
        query_text: str,
        top_k: int = 10,
    ) -> list[RetrievedEvidence]:
        """Search evidence within a specific scene."""
        filter_dict = create_retrieval_filter(scene_ids=[scene_id])
        return self.search(query_text, top_k, retrieval_filter=filter_dict)
    
    def search_by_entity(
        self,
        entity_label: str,
        query_text: str = "",
        top_k: int = 10,
    ) -> list[RetrievedEvidence]:
        """Search evidence for a specific entity."""
        # First try direct entity index lookup
        indices = self._entity_to_indices.get(entity_label, [])
        if indices and not query_text:
            # Return all evidence for this entity
            results = []
            for idx in indices:
                evidence = self._evidence_map.get(idx)
                if evidence:
                    results.append(RetrievedEvidence(
                        record=evidence,
                        semantic_similarity=1.0,
                        rank_score=0.0,
                        rank_factors={"entity_match": 1.0},
                    ))
            return results[:top_k]
        
        # Otherwise search with entity filter
        filter_dict = create_retrieval_filter(entity_labels=[entity_label])
        return self.search(query_text or f"Entity: {entity_label}", top_k, retrieval_filter=filter_dict)
    
    def get_scene_evidence(self, scene_id: str) -> list[EvidenceRecord]:
        """Get all evidence records for a scene."""
        indices = self._scene_to_indices.get(scene_id, [])
        return [self._evidence_map[idx] for idx in indices if idx in self._evidence_map]
    
    def get_entity_evidence(self, entity_label: str) -> list[EvidenceRecord]:
        """Get all evidence records for an entity."""
        indices = self._entity_to_indices.get(entity_label, [])
        return [self._evidence_map[idx] for idx in indices if idx in self._evidence_map]
    
    def get_image_evidence(self, image_id: str) -> list[EvidenceRecord]:
        """Get all evidence records for an image."""
        indices = self._image_to_indices.get(image_id, [])
        return [self._evidence_map[idx] for idx in indices if idx in self._evidence_map]
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_fn: callable = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[RetrievedEvidence]:
        """Search using a pre-computed embedding with optional hierarchical filtering."""
        if self._index is None or self._next_index == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        actual_k = min(top_k * 3, self._next_index)
        scores, indices = self._index.search(query_embedding, actual_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= self._next_index:
                continue
            
            evidence = self._evidence_map.get(idx)
            if evidence is None:
                continue
            
            if filter_fn and not filter_fn(evidence):
                continue
            
            if retrieval_filter and not apply_retrieval_filter(evidence, retrieval_filter):
                continue
            
            if retrieval_filter and "scene_ids" in retrieval_filter:
                indexed = self._indexed_evidence.get(idx)
                if indexed and indexed.scene_id not in retrieval_filter["scene_ids"]:
                    continue
            
            results.append(RetrievedEvidence(
                record=evidence,
                semantic_similarity=float(score),
                rank_score=0.0,
                rank_factors={"semantic_similarity": float(score)},
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_evidence(self, index_id: int) -> EvidenceRecord | None:
        """Get evidence by internal index ID."""
        return self._evidence_map.get(index_id)
    
    def get_indexed_evidence(self, index_id: int | str) -> IndexedEvidenceRecord | None:
        """Get indexed evidence with full provenance by internal index ID or evidence ID."""
        if isinstance(index_id, str):
            return self.get_indexed_evidence_by_id(index_id)
        return self._indexed_evidence.get(index_id)
    
    def get_evidence_by_id(self, evidence_id: str) -> EvidenceRecord | None:
        """Get evidence by evidence ID."""
        idx = self._id_to_index.get(evidence_id)
        if idx is not None:
            return self._evidence_map.get(idx)
        return None
    
    def get_indexed_evidence_by_id(self, evidence_id: str) -> IndexedEvidenceRecord | None:
        """Get indexed evidence with provenance by evidence ID."""
        idx = self._id_to_index.get(evidence_id)
        if idx is not None:
            return self._indexed_evidence.get(idx)
        return None
    
    def remove_evidence(self, evidence_id: str) -> bool:
        """Remove evidence from the store (marks as deleted, doesn't shrink index)."""
        idx = self._id_to_index.pop(evidence_id, None)
        if idx is not None:
            evidence = self._evidence_map.pop(idx, None)
            self._indexed_evidence.pop(idx, None)
            
            # Clean up indices
            if evidence:
                entity_indices = self._entity_to_indices.get(evidence.entity, [])
                if idx in entity_indices:
                    entity_indices.remove(idx)
            
            return True
        return False
    
    def save(self, path: str) -> None:
        """Save the index and metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        index_path = f"{path}.index"
        meta_path = f"{path}.meta"
        
        if self._index is not None:
            faiss.write_index(self._index, index_path)
        
        with open(meta_path, "wb") as f:
            pickle.dump({
                "evidence_map": self._evidence_map,
                "indexed_evidence": self._indexed_evidence,
                "id_to_index": self._id_to_index,
                "scene_to_indices": self._scene_to_indices,
                "entity_to_indices": self._entity_to_indices,
                "image_to_indices": self._image_to_indices,
                "next_index": self._next_index,
                "dimension": self._dimension,
                "index_type": self._index_type,
            }, f)
    
    def load(self, path: str) -> None:
        """Load the index and metadata from disk."""
        index_path = f"{path}.index"
        meta_path = f"{path}.meta"
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FAISSIndexError(f"Index files not found at {path}")
        
        self._index = faiss.read_index(index_path)
        
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self._evidence_map = data["evidence_map"]
            self._indexed_evidence = data.get("indexed_evidence", {})
            self._id_to_index = data["id_to_index"]
            self._scene_to_indices = data.get("scene_to_indices", {})
            self._entity_to_indices = data.get("entity_to_indices", {})
            self._image_to_indices = data.get("image_to_indices", {})
            self._next_index = data["next_index"]
            self._dimension = data["dimension"]
            self._index_type = data["index_type"]
    
    def clear(self) -> None:
        """Clear the store."""
        self._index = self._create_index()
        self._evidence_map.clear()
        self._indexed_evidence.clear()
        self._id_to_index.clear()
        self._scene_to_indices.clear()
        self._entity_to_indices.clear()
        self._image_to_indices.clear()
        self._next_index = 0
    
    def get_stats(self) -> dict[str, Any]:
        return {
            "count": self._next_index,
            "dimension": self._dimension,
            "index_type": self._index_type,
            "device": self._device,
            "num_scenes": len(self._scene_to_indices),
            "num_entities": len(self._entity_to_indices),
            "num_images": len(self._image_to_indices),
        }
    
    def get_scene_ids(self) -> list[str]:
        """Get all scene IDs in the store."""
        return list(self._scene_to_indices.keys())
    
    def get_entity_labels(self) -> list[str]:
        """Get all entity labels in the store."""
        return list(self._entity_to_indices.keys())
    
    def get_image_ids(self) -> list[str]:
        """Get all image IDs in the store."""
        return list(self._image_to_indices.keys())