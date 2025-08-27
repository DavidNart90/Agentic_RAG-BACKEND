"""
Utility functions for the JSON Chunking System.
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict


def generate_chunk_id(content: str, prefix: str = "chunk") -> str:
    """
    Generate a unique chunk ID based on content.

    Args:
        content: Content to hash
        prefix: Prefix for the chunk ID

    Returns:
        Unique chunk ID
    """
    # Create a hash of the content
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]

    # Create a timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    # Combine prefix, timestamp, and hash
    chunk_id = f"{prefix}_{timestamp}_{content_hash}"

    return chunk_id


def enrich_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich metadata with common fields.

    Args:
        metadata: Metadata dictionary

    Returns:
        Enriched metadata dictionary
    """
    # Add timestamp
    metadata["chunk_created_at"] = datetime.now(timezone.utc).isoformat()

    return metadata
