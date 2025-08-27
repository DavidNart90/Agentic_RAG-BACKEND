"""
Embedding System for TrackRealties AI Platform.

This module provides functionality to generate vector embeddings for text chunks,
enabling semantic search capabilities.
"""

from .cache import EmbeddingCache
from .embedder import Embedder
from .openai_embedder import OpenAIEmbedder

__all__ = ["Embedder", "OpenAIEmbedder", "EmbeddingCache"]
