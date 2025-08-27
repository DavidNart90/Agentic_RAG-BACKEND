"""
JSON Chunking System for TrackRealties AI Platform.

This module provides functionality to chunk JSON data semantically based on its structure,
ensuring that retrieval is more accurate and contextually relevant.
"""

from .chunk import Chunk
from .json_chunker import JSONChunker

__all__ = ["Chunk", "JSONChunker"]
