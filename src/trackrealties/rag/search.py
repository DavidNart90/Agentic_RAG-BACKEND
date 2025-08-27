"""
Search implementations for the RAG module.

This module provides vector search, graph search, and hybrid search capabilities
for the TrackRealties AI Platform.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.config import get_settings
from ..core.database import db_pool
from ..core.graph import graph_manager
from ..models.search import SearchFilters, SearchQuery, SearchResult
from ..rag.embedders import DefaultEmbedder
from ..rag.llm_entity_extraction import \
    LLMEntityExtractor as RealEstateEntityExtractor

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorSearch:
    """
    Vector-based semantic search using pgvector.

    This class provides methods to search for relevant content based on
    semantic similarity using vector embeddings stored in PostgreSQL.
    """

    def __init__(self):
        """Initialize the VectorSearch."""
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.embedder = DefaultEmbedder()

    async def initialize(self):
        """Initialize the vector search client."""
        await db_pool.initialize()
        await self.embedder.initialize()
        self.initialized = True
        self.logger.info("Vector search initialized")

    async def search(
        self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None, threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        Search for relevant content using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional filters to apply to the search
            threshold: Minimum similarity threshold

        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()

        query_embedding = await self.embedder.embed_query(query)
        query_embedding_str = str(query_embedding)

        # Build the filter query
        filter_clauses = []
        filter_values = []
        if filters:
            for key, value in filters.items():
                filter_clauses.append(f"metadata->>'{key}' = ${len(filter_values) + 3}")
                filter_values.append(value)

        filter_query = " AND ".join(filter_clauses)
        if filter_query:
            filter_query = "AND " + filter_query

        async with db_pool.acquire() as conn:
            # The l2_distance operator is <->
            # The inner_product operator is <#>
            # The cosine_distance operator is <=>
            # Query both property_chunks_enhanced and market_chunks_enhanced and combine them
            property_results = await conn.fetch(
                f"""
                SELECT
                    id,
                    content,
                    1 - (embedding <=> $1) AS similarity
                FROM
                    property_chunks_enhanced
                WHERE
                    1 - (embedding <=> $1) > $2
                    {filter_query}
                ORDER BY
                    similarity DESC
                LIMIT $3
                """,
                query_embedding_str,
                threshold,
                limit,
                *filter_values,
            )

            market_results = await conn.fetch(
                f"""
                SELECT
                    id,
                    content,
                    1 - (embedding <=> $1) AS similarity
                FROM
                    market_chunks_enhanced
                WHERE
                    1 - (embedding <=> $1) > $2
                    {filter_query}
                ORDER BY
                    similarity DESC
                LIMIT $3
                """,
                query_embedding_str,
                threshold,
                limit,
                *filter_values,
            )

        results = property_results + market_results
        # Sort the combined results by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        # Trim to the limit
        results = results[:limit]

        return [
            SearchResult(
                result_id=str(row["id"]),
                content=row["content"],
                similarity_score=row["similarity"],
                relevance_score=row["similarity"],  # Add required field
                result_type="property_listing",  # Add required field
                title="Search Result",  # Add required field
                source="Vector Database",  # Add required field
            )
            for row in results
        ]


class GraphSearch:
    """
    Graph-based search using Neo4j.

    This class provides methods to search for relevant content based on
    graph relationships stored in Neo4j.
    """

    def __init__(self):
        """Initialize the GraphSearch."""
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.entity_extractor = RealEstateEntityExtractor()

    async def initialize(self):
        """Initialize the graph search client."""
        await graph_manager.initialize()
        self.initialized = True
        self.logger.info("Graph search initialized")

    async def search(
        self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None, max_depth: int = 3
    ) -> List[SearchResult]:
        """
        Search for relevant content using graph relationships.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional filters to apply to the search
            max_depth: Maximum depth for graph traversal

        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()

        entities = await self.entity_extractor.extract_entities(query)

        if not entities:
            return []

        location_entities = [e["name"] for e in entities if e["type"] == "LOCATION"]

        if not location_entities:
            return []

        # This query is more specific. It looks for Market nodes connected to the given locations.
        cypher_query = """
        MATCH (l:Location)<-[:LOCATED_IN]-(m:Market)
        WHERE l.name IN $locations
        RETURN 
            elementId(m) AS id,
            m.summary AS content,
            "graph_fact" as result_type,
            l.name + " Market Report" as title,
            m.source AS source,
            1.0 AS score
        LIMIT $limit
        """

        async with graph_manager._driver.session() as session:
            result = await session.run(cypher_query, locations=location_entities, limit=limit)
            return [
                SearchResult(
                    result_id=record["id"],
                    content=record["content"],
                    relevance_score=record["score"],
                    result_type=record["result_type"],
                    title=record["title"],
                    source=record["source"],
                )
                for record in await result.data()
            ]


class HybridSearchEngine:
    """
    A search engine that combines vector, graph, and keyword search.
    """

    def __init__(self):
        """Initialize the HybridSearchEngine."""
        self.logger = logging.getLogger(__name__)
        self.vector_search = VectorSearch()
        self.graph_search = GraphSearch()
        self.initialized = False

    async def initialize(self):
        """Initialize the hybrid search client."""
        await asyncio.gather(self.vector_search.initialize(), self.graph_search.initialize())
        self.initialized = True
        self.logger.info("Hybrid search initialized")

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
    ) -> List[SearchResult]:
        """
        Search for relevant content using both vector and graph search.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional filters to apply to the search
            vector_weight: Weight for vector search results
            graph_weight: Weight for graph search results

        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()

        # Run both searches in parallel
        vector_results, graph_results = await asyncio.gather(
            self.vector_search.search(query, limit=limit, filters=filters),
            self.graph_search.search(query, limit=limit, filters=filters),
        )

        # Combine and rank results
        combined_results = self._combine_and_rank(vector_results, graph_results, vector_weight, graph_weight)

        self.logger.info(f"Hybrid search for: {query}")

        return combined_results[:limit]

    def _combine_and_rank(
        self,
        vector_results: List[SearchResult],
        graph_results: List[SearchResult],
        vector_weight: float,
        graph_weight: float,
    ) -> List[SearchResult]:
        """
        Combine and rank results from vector and graph search.
        """
        combined = {}

        # Add vector results
        for result in vector_results:
            combined[result.result_id] = {"result": result, "score": result.similarity_score * vector_weight}

        # Add graph results
        for result in graph_results:
            if result.result_id in combined:
                combined[result.result_id]["score"] += result.relevance_score * graph_weight
            else:
                combined[result.result_id] = {"result": result, "score": result.relevance_score * graph_weight}

        # Sort by score
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

        return [item["result"] for item in sorted_results]
