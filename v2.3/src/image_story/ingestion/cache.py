"""Cache management for vision results."""
import os
import json
import hashlib
from typing import Any
from pathlib import Path

from ..domain.schemas import CacheEntry


class VisionCache:
    """File-based cache for vision model results."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        prefix = cache_key[:2]
        subdir = self.cache_dir / prefix
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{cache_key}.json"
    
    def _make_cache_key(self, content_hash: str, model_name: str, model_version: str) -> str:
        """Generate cache key from content hash and model info."""
        key = f"{model_name}:{model_version}:{content_hash}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]
    
    def get(self, content_hash: str, model_name: str, model_version: str) -> dict[str, Any] | None:
        """Get cached result if exists."""
        cache_key = self._make_cache_key(content_hash, "", "")
        # Search for matching cache file
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    if (data.get("content_hash") == "" and 
                        data.get("model_name") == "" and
                        data.get("model_version") == ""):
                        entry = CacheEntry.from_dict(data)
                        entry.access_count += 1
                        entry.accessed_at = datetime.now().isoformat()
                        # Update access info
                        self.put(entry)
                        return entry.result
            except Exception:
                continue
        return None
    
    def put(self, entry: "CacheEntry") -> None:
        """Store a cache entry."""
        cache_path = self._get_cache_path(entry.cache_key)
        with open(cache_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    def get_by_key(self, cache_key: str) -> "CacheEntry | None":
        """Get entry by cache key."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    entry = CacheEntry.from_dict(data)
                    entry.access_count += 1
                    entry.accessed_at = datetime.now().isoformat()
                    self.put(entry)  # Update access info
                    return entry
            except Exception:
                pass
        return None
    
    def get_by_content_hash(self, content_hash: str, model_name: str, model_version: str) -> dict | None:
        """Get cached result by content hash and model info."""
        # Search for matching cache entry
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    if (data.get("content_hash") == content_hash and
                        data.get("model_name") == model_name and
                        data.get("model_version") == model_version):
                        entry = CacheEntry.from_dict(data)
                        entry.access_count += 1
                        entry.accessed_at = datetime.now().isoformat()
                        self.put(entry)
                        return entry.result
            except Exception:
                continue
        return None
    
    def put_result(self, content_hash: str, model_name: str, model_version: str, result: dict) -> None:
        """Store a vision result in cache."""
        from ..domain.schemas import CacheEntry
        from datetime import datetime
        
        cache_key = hashlib.sha256(f"{model_name}:{model_version}:{content_hash}".encode()).hexdigest()[:32]
        entry = CacheEntry(
            cache_key=cache_key,
            content_hash=content_hash,
            model_name=model_name,
            model_version=model_version,
            result=result,
        )
        self.put(entry)
    
    def clear_old_entries(self, max_age_days: int = 30, max_entries: int = 10000) -> int:
        """Clean up old cache entries."""
        removed = 0
        now = datetime.now()
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    entry = CacheEntry.from_dict(data)
                    age = (now - datetime.fromisoformat(entry.created_at)).days
                    if age > max_age_days or entry.access_count == 0:
                        cache_file.unlink()
                        removed += 1
            except Exception:
                pass
        return removed
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_size = 0
        entry_count = 0
        for cache_file in self.cache_dir.rglob("*.json"):
            entry_count += 1
            total_size += cache_file.stat().st_size
        return {
            "entry_count": entry_count,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }