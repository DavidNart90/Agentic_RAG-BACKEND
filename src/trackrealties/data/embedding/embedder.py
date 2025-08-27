"""
Base Embedder class for generating vector embeddings.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Embedder(ABC):
    """
    Abstract base class for embedding generators.

    This class defines the interface for embedding generators, which are responsible
    for converting text into vector embeddings for semantic search.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        use_cache: bool = True,
    ):
        """
        Initialize the Embedder.

        Args:
            model: Name of the embedding model to use
            dimensions: Dimensionality of the embeddings
            batch_size: Number of texts to process in a batch
            use_cache: Whether to use caching for embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = batch_size
        self.use_cache = use_cache

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding client."""
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Text to generate an embedding for

        Returns:
            Vector embedding as a list of floats
        """
        pass

    @abstractmethod
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of vector embeddings
        """
        pass

    async def get_embedding_with_fallback(self, text: str) -> List[float]:
        """
        Get an embedding with fallback to a zero vector if generation fails.

        Args:
            text: Text to generate an embedding for

        Returns:
            Vector embedding as a list of floats
        """
        try:
            embedding = await self.generate_embedding(text)
            return self.normalize_dimensions(embedding)
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.dimensions

    def normalize_dimensions(self, embedding: List[float]) -> List[float]:
        """
        Normalize embedding dimensions to match the expected dimensions for pgvector.

        This method ensures that embeddings have the correct dimensions for pgvector
        by either truncating or padding the vector as needed.

        Args:
            embedding: The embedding vector to normalize

        Returns:
            Normalized embedding vector with the correct dimensions
        """
        current_dim = len(embedding)

        # If dimensions match, return as is
        if current_dim == self.dimensions:
            return embedding

        # If embedding is too long, truncate
        if current_dim > self.dimensions:
            self.logger.warning(f"Truncating embedding from {current_dim} to {self.dimensions} dimensions")
            return embedding[: self.dimensions]

        # If embedding is too short, pad with zeros
        if current_dim < self.dimensions:
            self.logger.warning(f"Padding embedding from {current_dim} to {self.dimensions} dimensions")
            return embedding + [0.0] * (self.dimensions - current_dim)
