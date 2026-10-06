"""Cache management for vision results."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
        """Get a cached vision result, or None on a miss.

        The cache key is derived from (model, version, content hash), so this is
        a direct file lookup. An earlier revision ignored its arguments and
        scanned every cache file comparing against empty strings, which could
        only ever match an entry written the same wrong way.
        """
        entry = self.get_by_key(self._make_cache_key(content_hash, model_name, model_version))
        return entry.result if entry else None

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
                with open(cache_path) as f:
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
        """Get a cached result by content hash and model info.

        Equivalent to `get()`; kept because it reads better at some call sites.
        """
        return self.get(content_hash, model_name, model_version)

    def put_result(self, content_hash: str, model_name: str, model_version: str, result: dict) -> None:
        """Store a vision result in the cache."""
        entry = CacheEntry(
            cache_key=self._make_cache_key(content_hash, model_name, model_version),
            content_hash=content_hash,
            model_name=model_name,
            model_version=model_version,
            result=result,
        )
        self.put(entry)

    def clear_old_entries(self, max_age_days: int = 30, max_entries: int = 10000) -> int:
        """Remove cache entries that are stale or have never been read.

        Candidate paths are collected before any unlink: deleting a file while
        iterating a `Path.rglob` generator fails on Windows (the generator holds a
        directory handle), and the bare `except` that used to guard this loop
        swallowed the error, so the function silently removed nothing.
        """
        now = datetime.now()
        stale: list[Path] = []
        for cache_file in sorted(self.cache_dir.rglob("*.json")):
            try:
                with open(cache_file, encoding="utf-8") as f:
                    entry = CacheEntry.from_dict(json.load(f))
                age = (now - datetime.fromisoformat(entry.created_at)).days
                if age > max_age_days or entry.access_count == 0:
                    stale.append(cache_file)
            except Exception:  # noqa: BLE001 - a corrupt entry must not stop cleanup
                stale.append(cache_file)

        removed = 0
        for cache_file in stale:
            try:
                cache_file.unlink()
                removed += 1
            except OSError:
                continue
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
