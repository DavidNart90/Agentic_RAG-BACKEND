"""
OpenAI embedder implementation for generating vector embeddings.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from ...core.config import get_settings
from .cache import EmbeddingCache
from .embedder import Embedder

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIEmbedder(Embedder):
    """
    OpenAI embedder for generating vector embeddings.

    This class uses the OpenAI API to generate vector embeddings for text chunks.
    It supports batching, caching, and fallback mechanisms.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        use_cache: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the OpenAI embedder.

        Args:
            model: Name of the OpenAI embedding model to use
            dimensions: Dimensionality of the embeddings
            batch_size: Number of texts to process in a batch
            use_cache: Whether to use caching for embeddings
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        super().__init__(model, dimensions, batch_size, use_cache)
        self.api_key = api_key or settings.embedding_api_key or settings.llm_api_key
        self.base_url = base_url or settings.EMBEDDING_BASE_URL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.cache = EmbeddingCache() if use_cache else None

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            if AsyncOpenAI is None:
                self.logger.error("Failed to import OpenAI package. Please install it with 'pip install openai'")
                raise ImportError("Failed to import OpenAI package. Please install it with 'pip install openai'")

            if not self.api_key:
                self.logger.error("No API key found for OpenAI embedding generation")
                raise ValueError("No API key found for OpenAI embedding generation")

            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.logger.info(f"Initialized OpenAI embedder with model {self.model}")

            # Initialize cache if enabled
            if self.cache:
                await self.cache.initialize()

        except ImportError:
            self.logger.error("Failed to import OpenAI package. Please install it with 'pip install openai'")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Text to generate an embedding for

        Returns:
            Vector embedding as a list of floats
        """
        # Check cache first if enabled
        if self.cache:
            cached_result = await self.cache.get(text, self.model)
            if cached_result:
                cached_embedding, _ = cached_result
                self.logger.debug("Using cached embedding")
                return cached_embedding

        # Generate embedding
        embeddings, token_counts = await self.generate_embeddings_batch([text])

        # Cache the result if enabled
        if self.cache and embeddings:
            await self.cache.set(text, embeddings[0], self.model, token_counts[0])

        return embeddings[0] if embeddings else [0.0] * self.dimensions

    async def generate_embeddings_batch(self, texts: List[str]) -> Tuple[List[List[float]], List[int]]:
        """
        Generate embeddings for a batch of texts efficiently.
        """
        if not texts:
            return [], []

        # Initialize client if needed
        if not self.client:
            await self.initialize()

        start_time = time.time()
        total_texts = len(texts)
        self.logger.debug(f"Generating embeddings for {total_texts} texts")

        # Step 1: Check cache first if enabled
        if self.cache:
            cached_results = await asyncio.gather(*[self.cache.get(text, self.model) for text in texts])
            cached_embeddings = [res[0] if res else None for res in cached_results]
            cached_token_counts = [res[1] if res else 0 for res in cached_results]
            missing_indices = [i for i, result in enumerate(cached_embeddings) if result is None]

            cache_hit_count = total_texts - len(missing_indices)
            if cache_hit_count > 0:
                self.logger.debug(f"Cache hit: {cache_hit_count}/{total_texts} embeddings found in cache")

            if not missing_indices:
                self.logger.debug("Using all cached embeddings (100% cache hit)")
                return cached_embeddings, cached_token_counts

            texts_to_generate = [texts[i] for i in missing_indices]
            self.logger.debug(f"Generating: {len(texts_to_generate)} missing embeddings")
        else:
            texts_to_generate = texts
            missing_indices = list(range(len(texts)))
            cached_embeddings = [None] * len(texts)
            cached_token_counts = [0] * len(texts)

        # Step 2: Generate embeddings for missing texts in optimal batches
        generated_embeddings = []
        generated_token_counts = []
        total_to_generate = len(texts_to_generate)
        processed_count = 0

        # Process in batches
        for i in range(0, total_to_generate, self.batch_size):
            batch = texts_to_generate[i: i + self.batch_size]
            batch_size = len(batch)

            self.logger.debug(f"Processing batch {i // self.batch_size + 1}: {batch_size} texts")
            batch_start_time = time.time()

            # Generate embeddings for this batch with retry logic
            batch_embeddings, batch_token_counts = await self._generate_batch_with_retry(batch)

            # Check if we got the expected number of embeddings
            if len(batch_embeddings) != batch_size:
                self.logger.warning(
                    f"Batch returned {len(batch_embeddings)} embeddings, expected {batch_size}. "
                    f"Using zero vectors for missing embeddings."
                )
                # Fill in missing embeddings with zero vectors
                while len(batch_embeddings) < batch_size:
                    batch_embeddings.append([0.0] * self.dimensions)
                    batch_token_counts.append(0)

            generated_embeddings.extend(batch_embeddings)
            generated_token_counts.extend(batch_token_counts)
            processed_count += batch_size

            batch_duration = time.time() - batch_start_time
            if total_to_generate > self.batch_size:
                progress = processed_count / total_to_generate * 100
                self.logger.debug(
                    f"Progress: {processed_count}/{total_to_generate} ({progress:.1f}%) - "
                    f"Batch took {batch_duration:.2f}s"
                )

        # Step 3: Cache the generated embeddings if enabled
        if self.cache:
            cache_tasks = []
            for i, (embedding, token_count) in enumerate(zip(generated_embeddings, generated_token_counts)):
                if any(embedding):  # Only cache non-zero embeddings
                    original_text_index = missing_indices[i]
                    cache_tasks.append(self.cache.set(texts[original_text_index], embedding, self.model, token_count))

            if cache_tasks:
                self.logger.debug(f"Caching {len(cache_tasks)} new embeddings")
                await asyncio.gather(*cache_tasks, return_exceptions=True)

        # Step 4: Combine cached and generated embeddings
        if self.cache:
            result_embeddings = list(cached_embeddings)
            result_token_counts = list(cached_token_counts)
            for i, (embedding, token_count) in enumerate(zip(generated_embeddings, generated_token_counts)):
                result_embeddings[missing_indices[i]] = embedding
                result_token_counts[missing_indices[i]] = token_count
        else:
            result_embeddings = generated_embeddings
            result_token_counts = generated_token_counts

        total_duration = time.time() - start_time
        self.logger.debug(f"Generated {len(generated_embeddings)} embeddings in {total_duration:.2f}s")

        return result_embeddings, result_token_counts

    async def _generate_batch_with_retry(self, texts: List[str]) -> Tuple[List[List[float]], List[int]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                # Pre-calculate token counts
                try:
                    encoding = tiktoken.encoding_for_model(self.model)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")

                token_counts = [len(encoding.encode(text)) for text in texts]

                # Make API call
                response = await self.client.embeddings.create(model=self.model, input=texts)

                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]

                # Ensure we have the right number of embeddings
                if len(embeddings) != len(texts):
                    self.logger.warning(f"API returned {len(embeddings)} embeddings for {len(texts)} texts")
                    # Pad with zero vectors if needed
                    while len(embeddings) < len(texts):
                        embeddings.append([0.0] * self.dimensions)

                    # Adjust token counts to match
                    while len(token_counts) < len(embeddings):
                        token_counts.append(0)

                # Normalize dimensions for pgvector compatibility
                if embeddings and len(embeddings[0]) != self.dimensions:
                    self.logger.warning(
                        f"Embedding dimensions mismatch: expected {self.dimensions}, "
                        f"got {len(embeddings[0])}. Normalizing dimensions."
                    )
                    embeddings = [self.normalize_dimensions(embedding) for embedding in embeddings]

                return embeddings, token_counts

            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    self.logger.error(f"Failed to generate embeddings after {self.max_retries} retries: {e}")
                    # Return zero vectors as fallback
                    return [[0.0] * self.dimensions for _ in texts], [0] * len(texts)

                self.logger.warning(f"Embedding generation failed (attempt {retries}/{self.max_retries}): {e}")
                await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff

        # This should never be reached, but just in case
        return [[0.0] * self.dimensions for _ in texts], [0] * len(texts)

    def normalize_dimensions(self, embedding: List[float]) -> List[float]:
        """
        Normalize embedding dimensions to match expected dimensions.

        Args:
            embedding: Input embedding vector

        Returns:
            Normalized embedding vector with correct dimensions
        """
        if len(embedding) == self.dimensions:
            return embedding
        elif len(embedding) > self.dimensions:
            # Truncate if too long
            return embedding[: self.dimensions]
        else:
            # Pad with zeros if too short
            return embedding + [0.0] * (self.dimensions - len(embedding))
