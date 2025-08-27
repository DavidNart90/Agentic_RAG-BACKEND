# RAG Module for TrackRealties AI Platform

This module implements the Retrieval-Augmented Generation (RAG) system for the TrackRealties AI platform, providing intelligent search and context-aware responses for real estate data.

## Overview

The RAG module combines vector search, graph search, and hybrid search capabilities to retrieve relevant information from the platform's databases and provide accurate, context-aware responses to user queries.

## Components

- **Vector Search**: Semantic similarity search using pgvector
- **Graph Search**: Relationship-based search using Neo4j
- **Hybrid Search**: Combined approach leveraging both vector and graph search
- **Hallucination Detection**: Domain-specific validation to ensure accuracy
- **External Fallback**: Web search integration when internal data is insufficient

## Usage

```python
from trackrealties.rag.search import HybridSearch
from trackrealties.core.config import get_settings

# Initialize search
settings = get_settings()
search = HybridSearch()
await search.initialize()

# Perform search
results = await search.search(
    query="investment properties in Seattle with positive cash flow",
    limit=5,
    filters={"property_type": "multi_family", "min_price": 500000}
)

# Process results
for result in results:
    print(f"Score: {result.score}, Content: {result.content}")
```

## Architecture

The RAG module follows a layered architecture:

1. **Query Processing**: Analyzes and enhances user queries
2. **Search Execution**: Performs vector, graph, or hybrid search
3. **Result Ranking**: Combines and ranks results from different sources
4. **Context Assembly**: Builds a comprehensive context for the LLM
5. **Response Generation**: Generates accurate, role-specific responses

## Integration

The RAG module integrates with:

- **Agent System**: Provides context for agent reasoning
- **Database Layer**: Retrieves data from PostgreSQL and Neo4j
- **Embedding System**: Utilizes embeddings for semantic search
