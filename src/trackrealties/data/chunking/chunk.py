"""
Chunk data model for the JSON Chunking System.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """Represents a semantic chunk of data with metadata."""

    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    parent_id: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "embedding": self.embedding,
        }
