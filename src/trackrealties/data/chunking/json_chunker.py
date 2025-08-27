"""
JSON Chunker implementation for the TrackRealties AI Platform.
"""

import logging
from typing import Any, Dict, List, Optional

from ...core.config import get_settings
from .chunk import Chunk
from .enhanced_semantic_chunker import EnhancedSemanticChunker
from .utils import generate_chunk_id

logger = logging.getLogger(__name__)
settings = get_settings()


class JSONChunker:
    """
    Chunks JSON data semantically based on its structure.

    This class provides methods to break down JSON data into semantically meaningful chunks
    based on the structure of the data, with special handling for property listings and
    market data.
    """

    def __init__(self, max_chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """
        Initialize the JSONChunker.

        Args:
            max_chunk_size: Maximum size of a chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.enhanced_chunker = EnhancedSemanticChunker(
            max_chunk_size or settings.max_chunk_size, chunk_overlap or settings.chunk_overlap
        )

        self.logger = logging.getLogger(__name__)
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Chunking methods
    def chunk_json_enhanced(self, data: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
        """Use enhanced semantic chunking"""
        return self.enhanced_chunker.chunk_with_semantic_awareness(data, data_type)

    def chunk_json(self, data: Dict[str, Any], data_type: str) -> List[Chunk]:
        """Chunk JSON data using enhanced semantic chunking."""
        self.logger.info(f"Chunking JSON data of type {data_type}")

        # Always use enhanced chunking
        enhanced_chunks = self.enhanced_chunker.chunk_with_semantic_awareness(data, data_type)

        # Convert to Chunk objects
        chunks = []
        for chunk_data in enhanced_chunks:
            chunk = Chunk(
                chunk_id=generate_chunk_id(chunk_data["content"], data_type),
                content=chunk_data["content"],
                metadata=chunk_data["metadata"],
            )
            chunks.append(chunk)

        self.logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
