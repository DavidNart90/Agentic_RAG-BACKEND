"""
Caching system for embeddings to reduce API calls and improve performance.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingCache:
    """
    Cache for embeddings to reduce API calls and improve performance.

    This class provides an in-memory cache with optional disk persistence
    for storing and retrieving embeddings.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_seconds: int = 86400,  # 24 hours
        max_size: int = 10000,
        persist_to_disk: bool = True,
    ):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for cache entries in seconds
            max_size: Maximum number of entries in the cache
            persist_to_disk: Whether to persist the cache to disk
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir or os.path.join("cache", "embeddings")
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.persist_to_disk = persist_to_disk
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the cache and load from disk if enabled."""
        if self.persist_to_disk:
            os.makedirs(self.cache_dir, exist_ok=True)
            await self._load_from_disk()

    def _get_cache_key(self, text: str, model: str) -> str:
        """
        Generate a cache key for a text and model.

        Args:
            text: Text to generate a key for
            model: Model name

        Returns:
            Cache key
        """
        # Create a hash of the text and model
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model}_{text_hash}"

    def _get_cache_file_path(self, model: str) -> str:
        """
        Get the file path for a model's cache.

        Args:
            model: Model name

        Returns:
            Cache file path
        """
        # Sanitize model name for file path
        safe_model = model.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"{safe_model}_cache.pkl")

    async def get(self, text: str, model: str) -> Optional[Tuple[List[float], int]]:
        """
        Get an embedding and its token count from the cache.

        Args:
            text: Text to get the embedding for
            model: Model name

        Returns:
            A tuple containing the cached embedding and token count, or None if not found
        """
        key = self._get_cache_key(text, model)

        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check if the entry is expired
                if "timestamp" in entry:
                    timestamp = entry["timestamp"]
                    if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                        # Entry is expired
                        del self.cache[key]
                        return None

                return entry.get("embedding"), entry.get("token_count", 0)

        return None

    async def set(self, text: str, embedding: List[float], model: str, token_count: int) -> None:
        """
        Set an embedding and its token count in the cache.

        Args:
            text: Text the embedding is for
            embedding: Vector embedding
            model: Model name
            token_count: The number of tokens in the text
        """
        key = self._get_cache_key(text, model)

        async with self.lock:
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size:
                await self._evict_entries()

            # Add the entry to the cache
            self.cache[key] = {
                "embedding": embedding,
                "token_count": token_count,
                "timestamp": datetime.now(),
                "model": model,
            }

            # Persist to disk if enabled
            if self.persist_to_disk:
                await self._save_to_disk(model)

    async def _evict_entries(self) -> None:
        """Evict old entries from the cache."""
        # Sort entries by timestamp
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].get("timestamp", datetime.min))

        # Remove the oldest entries
        entries_to_remove = max(1, int(self.max_size * 0.2))  # Remove at least 20% of entries
        for key, _ in sorted_entries[:entries_to_remove]:
            del self.cache[key]

    async def _save_to_disk(self, model: str) -> None:
        """
        Save the cache to disk.

        Args:
            model: Model name to save cache for
        """
        try:
            # Get entries for this model
            model_entries = {key: entry for key, entry in self.cache.items() if entry.get("model") == model}

            if not model_entries:
                return

            # Save to disk
            file_path = self._get_cache_file_path(model)
            with open(file_path, "wb") as f:
                pickle.dump(model_entries, f)

            self.logger.debug(f"Saved {len(model_entries)} cache entries to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save cache to disk: {e}")

    async def _load_from_disk(self) -> None:
        """Load the cache from disk."""
        try:
            # Find all cache files
            if not os.path.exists(self.cache_dir):
                return

            cache_files = [
                os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith("_cache.pkl")
            ]

            for file_path in cache_files:
                try:
                    with open(file_path, "rb") as f:
                        entries = pickle.load(f)

                    # Add entries to the cache
                    self.cache.update(entries)

                    self.logger.debug(f"Loaded {len(entries)} cache entries from {file_path}")

                except Exception as e:
                    self.logger.error(f"Failed to load cache from {file_path}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to load cache from disk: {e}")

    async def clear(self) -> None:
        """Clear the cache."""
        async with self.lock:
            self.cache.clear()

            # Remove cache files if enabled
            if self.persist_to_disk:
                try:
                    for file_path in os.listdir(self.cache_dir):
                        if file_path.endswith("_cache.pkl"):
                            os.remove(os.path.join(self.cache_dir, file_path))
                except Exception as e:
                    self.logger.error(f"Failed to remove cache files: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        async with self.lock:
            # Count entries by model
            model_counts = {}
            for entry in self.cache.values():
                model = entry.get("model", "unknown")
                model_counts[model] = model_counts.get(model, 0) + 1

            # Calculate size in memory
            size_bytes = sum(len(pickle.dumps(entry)) for entry in self.cache.values())

            return {
                "total_entries": len(self.cache),
                "models": model_counts,
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "persist_to_disk": self.persist_to_disk,
            }
