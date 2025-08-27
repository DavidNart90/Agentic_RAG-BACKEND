"""
Embedder implementations for the RAG module.
"""

import hashlib
import logging
from typing import List

from openai import AsyncOpenAI

from ..core.config import settings

logger = logging.getLogger(__name__)


class DefaultEmbedder:
    """Default embedder using OpenAI with caching for performance."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.model = settings.EMBEDDING_MODEL
        self.initialized = False

        # Simple embedding cache to avoid repeat API calls
        self.cache = {}
        self.cache_order = []
        self.cache_size = 500
        self.cache_hits = 0
        self.cache_misses = 0

    async def initialize(self):
        """Initialize the OpenAI client."""
        if not self.initialized:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.initialized = True
            self.logger.info(f"Initialized OpenAI embedder with model: {self.model}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _evict_lru(self):
        """Remove least recently used item from cache."""
        if len(self.cache) >= self.cache_size and self.cache_order:
            oldest = self.cache_order.pop(0)
            if oldest in self.cache:
                del self.cache[oldest]

    def _update_access(self, key: str):
        """Update access order for LRU."""
        if key in self.cache_order:
            self.cache_order.remove(key)
        self.cache_order.append(key)

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query with caching."""
        if not self.initialized:
            await self.initialize()

        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            self._update_access(cache_key)
            self.cache_hits += 1
            return self.cache[cache_key]

        # Generate new embedding
        self.cache_misses += 1
        response = await self.client.embeddings.create(input=[text], model=self.model)
        embedding = response.data[0].embedding

        # Store in cache
        self._evict_lru()
        self.cache[cache_key] = embedding
        self._update_access(cache_key)

        return embedding

    @property
    def cache_hit_rate(self) -> float:
        """Get current cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not self.initialized:
            await self.initialize()

        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in response.data]
