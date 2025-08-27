"""
Performance-Optimized RAG Pipeline
Implementing critical optimizations for retrieval speed improvements
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track retrieval performance metrics"""

    query_times: List[float]
    embedding_times: List[float]
    db_query_times: List[float]
    result_counts: List[int]
    cache_hits: int = 0
    cache_misses: int = 0

    def add_query_time(self, time_ms: float, result_count: int):
        self.query_times.append(time_ms)
        self.result_counts.append(result_count)

    def add_embedding_time(self, time_ms: float):
        self.embedding_times.append(time_ms)

    def add_db_time(self, time_ms: float):
        self.db_query_times.append(time_ms)

    def cache_hit(self):
        self.cache_hits += 1

    def cache_miss(self):
        self.cache_misses += 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_query_time(self) -> float:
        return np.mean(self.query_times) if self.query_times else 0.0

    @property
    def p95_query_time(self) -> float:
        return np.percentile(self.query_times, 95) if self.query_times else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "avg_query_time_ms": self.avg_query_time,
            "p95_query_time_ms": self.p95_query_time,
            "avg_embedding_time_ms": np.mean(self.embedding_times) if self.embedding_times else 0.0,
            "avg_db_time_ms": np.mean(self.db_query_times) if self.db_query_times else 0.0,
            "cache_hit_rate": self.cache_hit_rate,
            "total_queries": len(self.query_times),
            "avg_results_per_query": np.mean(self.result_counts) if self.result_counts else 0.0,
        }


class CachedEmbedder:
    """High-performance embedding cache with LRU eviction"""

    def __init__(self, base_embedder, cache_size: int = 1000):
        self.base_embedder = base_embedder
        self.cache = {}
        self.cache_order = []  # For LRU tracking
        self.cache_size = cache_size
        self.metrics = PerformanceMetrics([], [], [], [])

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _evict_lru(self):
        """Remove least recently used item from cache"""
        if len(self.cache) >= self.cache_size and self.cache_order:
            lru_key = self.cache_order.pop(0)
            if lru_key in self.cache:
                del self.cache[lru_key]

    def _update_lru(self, key: str):
        """Update LRU order for accessed key"""
        if key in self.cache_order:
            self.cache_order.remove(key)
        self.cache_order.append(key)

    async def embed_query(self, query: str) -> List[float]:
        """Get embedding with caching"""
        start_time = time.time()
        cache_key = self._get_cache_key(query)

        # Check cache first
        if cache_key in self.cache:
            self._update_lru(cache_key)
            self.metrics.cache_hit()
            embedding_time = (time.time() - start_time) * 1000
            self.metrics.add_embedding_time(embedding_time)
            return self.cache[cache_key]

        # Generate embedding
        self.metrics.cache_miss()
        embedding = await self.base_embedder.embed_query(query)

        # Store in cache
        self._evict_lru()
        self.cache[cache_key] = embedding
        self._update_lru(cache_key)

        embedding_time = (time.time() - start_time) * 1000
        self.metrics.add_embedding_time(embedding_time)

        return embedding


class OptimizedVectorSearchV2:
    """Performance-optimized vector search with batching and caching"""

    def __init__(self, embedder):
        self.embedder = CachedEmbedder(embedder, cache_size=1000)
        self.metrics = PerformanceMetrics([], [], [], [])
        self.initialized = False

    async def initialize(self):
        """Initialize with performance optimizations"""
        from ..core.database import db_pool

        await db_pool.initialize()
        await self.embedder.base_embedder.initialize()

        # Create optimized indexes
        await self._create_performance_indexes()
        self.initialized = True
        logger.info("Optimized vector search initialized with performance enhancements")

    async def _create_performance_indexes(self):
        """Create performance-optimized database indexes"""
        from ..core.database import db_pool

        optimization_queries = [
            # Enhanced HNSW indexes
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_embedding_hnsw_perf
            ON property_chunks_enhanced USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 32, ef_construction = 200)
            WHERE embedding IS NOT NULL;
            """,
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_embedding_hnsw_perf
            ON market_chunks_enhanced USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 32, ef_construction = 200)
            WHERE embedding IS NOT NULL;
            """,
            # Covering indexes for common queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_search_covering
            ON property_chunks_enhanced (property_listing_id, chunk_type, semantic_score)
            INCLUDE (content, metadata)
            WHERE embedding IS NOT NULL AND semantic_score > 0.5;
            """,
            # Partial indexes for high-quality content
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_high_quality
            ON property_chunks_enhanced USING hnsw (embedding vector_cosine_ops)
            WHERE semantic_score > 0.7 AND embedding IS NOT NULL;
            """,
            # Computed price columns for faster filtering
            """
            ALTER TABLE property_chunks_enhanced 
            ADD COLUMN IF NOT EXISTS extracted_list_price NUMERIC;
            """,
            """
            UPDATE property_chunks_enhanced 
            SET extracted_list_price = regexp_replace(
                substring(content FROM 'List Price: \\$([0-9,]+)'), ',', '', 'g'
            )::numeric
            WHERE chunk_type = 'financial_analysis' 
            AND content ~ 'List Price: \\$([0-9,]+)'
            AND extracted_list_price IS NULL;
            """,
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_price_range
            ON property_chunks_enhanced (extracted_list_price)
            WHERE extracted_list_price IS NOT NULL;
            """,
        ]

        async with db_pool.acquire() as conn:
            for query in optimization_queries:
                try:
                    await conn.execute(query)
                    logger.info("Successfully executed optimization query")
                except Exception as e:
                    logger.warning(f"Optimization query failed (may already exist): {e}")

    async def search_optimized(
        self, query: str, limit: int = 15, filters: Optional[Dict[str, Any]] = None, threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Optimized search with performance improvements"""

        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Parallel embedding generation and query analysis
            embedding_task = asyncio.create_task(self.embedder.embed_query(query))
            analysis_task = asyncio.create_task(self._analyze_query(query))

            embedding, query_analysis = await asyncio.gather(embedding_task, analysis_task)

            # Execute optimized database query based on analysis
            db_start = time.time()
            results = await self._execute_optimized_query(embedding, query_analysis, limit, threshold, filters)
            db_time = (time.time() - db_start) * 1000
            self.metrics.add_db_time(db_time)

            total_time = (time.time() - start_time) * 1000
            self.metrics.add_query_time(total_time, len(results))

            logger.info(f"Optimized search completed in {total_time:.2f}ms, {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Optimized search failed: {e}")
            return []

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Fast query analysis for optimization decisions"""
        query_lower = query.lower()

        return {
            "is_property_search": any(
                kw in query_lower for kw in ["property", "house", "home", "listing", "bedroom", "bathroom"]
            ),
            "is_market_search": any(kw in query_lower for kw in ["market", "median price", "trends", "analysis"]),
            "is_investment_query": any(kw in query_lower for kw in ["invest", "budget", "have $", "spend $"]),
            "has_location": any(kw in query_lower for kw in ["texas", "austin", "dallas", "houston", "san antonio"]),
            "has_price_filter": any(kw in query_lower for kw in ["under", "over", "between", "$"]),
        }

    async def _execute_optimized_query(
        self,
        embedding: List[float],
        query_analysis: Dict[str, Any],
        limit: int,
        threshold: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute optimized database query based on analysis"""

        from ..core.database import db_pool

        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        # Build filter conditions
        filter_clauses = []
        filter_values = []
        if filters:
            for key, value in filters.items():
                filter_clauses.append(f"metadata->>'{key}' = ${len(filter_values) + 4}")
                filter_values.append(value)

        filter_query = " AND " + " AND ".join(filter_clauses) if filter_clauses else ""

        async with db_pool.acquire() as conn:
            if query_analysis["is_property_search"] and not query_analysis["is_market_search"]:
                # Optimized property-only search with batching
                query = f"""
                WITH top_properties AS (
                    SELECT DISTINCT 
                        property_listing_id,
                        MAX(1 - (embedding <=> $1)) as max_similarity
                    FROM property_chunks_enhanced 
                    WHERE (1 - (embedding <=> $1)) > $2 
                    AND embedding IS NOT NULL
                    {filter_query}
                    GROUP BY property_listing_id
                    ORDER BY max_similarity DESC
                    LIMIT $3
                )
                SELECT 
                    'property_' || pc.id::text as result_id,
                    pc.content,
                    'property_listing' as result_type,
                    (1 - (pc.embedding <=> $1)) as similarity,
                    COALESCE(pc.metadata->>'address', pc.metadata->>'title', 'Property Listing') as title,
                    'Property Database' as source,
                    pc.metadata,
                    pc.chunk_type,
                    pc.property_listing_id,
                    tp.max_similarity
                FROM property_chunks_enhanced pc
                INNER JOIN top_properties tp ON pc.property_listing_id = tp.property_listing_id
                ORDER BY tp.max_similarity DESC, similarity DESC
                LIMIT $3;
                """

                results = await conn.fetch(query, embedding_str, threshold, limit, *filter_values)

            elif query_analysis["is_market_search"] and not query_analysis["is_property_search"]:
                # Optimized market-only search
                query = f"""
                SELECT DISTINCT
                    'market_' || id::text as result_id,
                    content,
                    'market_data' as result_type,
                    1 - (embedding <=> $1) AS similarity,
                    COALESCE(metadata->>'region_name', metadata->>'title', 'Market Data') as title,
                    'Market Database' as source,
                    metadata
                FROM market_chunks_enhanced
                WHERE (1 - (embedding <=> $1)) > $2 
                AND embedding IS NOT NULL
                {filter_query}
                ORDER BY similarity DESC
                LIMIT $3;
                """

                results = await conn.fetch(query, embedding_str, threshold, limit, *filter_values)

            else:
                # Unified dual-table search for mixed or investment queries
                query = f"""
                SELECT * FROM (
                    (SELECT 
                        'property_' || id::text as result_id,
                        content,
                        'property_listing' as result_type,
                        1 - (embedding <=> $1) AS similarity,
                        COALESCE(metadata->>'address', metadata->>'title', 'Property Listing') as title,
                        'Property Database' as source,
                        metadata
                    FROM property_chunks_enhanced
                    WHERE (1 - (embedding <=> $1)) > $2 
                    AND embedding IS NOT NULL
                    {filter_query}
                    ORDER BY similarity DESC
                    LIMIT $3)

                    UNION ALL

                    (SELECT 
                        'market_' || id::text as result_id,
                        content,
                        'market_data' as result_type,
                        1 - (embedding <=> $1) AS similarity,
                        COALESCE(metadata->>'region_name', metadata->>'title', 'Market Data') as title,
                        'Market Database' as source,
                        metadata
                    FROM market_chunks_enhanced
                    WHERE (1 - (embedding <=> $1)) > $2 
                    AND embedding IS NOT NULL
                    {filter_query}
                    ORDER BY similarity DESC
                    LIMIT $3)
                ) combined_results
                ORDER BY similarity DESC
                LIMIT $3;
                """

                results = await conn.fetch(query, embedding_str, threshold, limit // 2, *filter_values)

            return [dict(row) for row in results]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "search_metrics": self.metrics.get_stats(),
            "embedding_metrics": self.embedder.metrics.get_stats(),
            "cache_size": len(self.embedder.cache),
            "cache_hit_rate": self.embedder.metrics.cache_hit_rate,
        }


class ParallelSearchExecutor:
    """Execute multiple search strategies in parallel"""

    def __init__(self, vector_search, graph_search=None):
        self.vector_search = vector_search
        self.graph_search = graph_search
        self.metrics = PerformanceMetrics([], [], [], [])

    async def parallel_search(
        self, query: str, limit: int = 15, vector_weight: float = 0.7, graph_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Execute vector and graph searches in parallel"""

        start_time = time.time()

        # Prepare search tasks
        tasks = []

        # Always include vector search
        vector_task = asyncio.create_task(self.vector_search.search_optimized(query, limit=limit))
        tasks.append(("vector", vector_task))

        # Include graph search if available
        if self.graph_search:
            graph_task = asyncio.create_task(self.graph_search.search(query, limit=limit // 2))
            tasks.append(("graph", graph_task))

        # Execute all searches in parallel
        results = {}
        for search_type, task in tasks:
            try:
                results[search_type] = await task
            except Exception as e:
                logger.error(f"{search_type} search failed: {e}")
                results[search_type] = []

        # Combine results with weighted scoring
        combined_results = self._combine_results(
            results.get("vector", []), results.get("graph", []), vector_weight, graph_weight, limit
        )

        total_time = (time.time() - start_time) * 1000
        self.metrics.add_query_time(total_time, len(combined_results))

        logger.info(f"Parallel search completed in {total_time:.2f}ms")
        return combined_results

    def _combine_results(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
        vector_weight: float,
        graph_weight: float,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Combine and rank results from multiple search strategies"""

        combined = []

        # Add vector results with weighted scores
        for result in vector_results:
            combined.append(
                {**result, "combined_score": result.get("similarity", 0.5) * vector_weight, "source_strategy": "vector"}
            )

        # Add graph results with weighted scores
        for result in graph_results:
            # Convert graph result format if needed
            if hasattr(result, "__dict__"):
                result_dict = {
                    "result_id": str(result.result_id),
                    "content": result.content,
                    "result_type": result.result_type,
                    "similarity": result.relevance_score,
                    "title": result.title,
                    "source": result.source,
                    "combined_score": result.relevance_score * graph_weight,
                    "source_strategy": "graph",
                }
            else:
                result_dict = {
                    **result,
                    "combined_score": result.get("similarity", 0.5) * graph_weight,
                    "source_strategy": "graph",
                }
            combined.append(result_dict)

        # Sort by combined score and limit results
        combined.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined[:limit]


# Usage Example and Benchmarking
async def benchmark_optimizations():
    """Benchmark the performance optimizations"""

    from ..rag.embedders import DefaultEmbedder
    from ..rag.optimized_pipeline import OptimizedVectorSearch

    # Initialize components
    embedder = DefaultEmbedder()
    original_search = OptimizedVectorSearch()
    optimized_search = OptimizedVectorSearchV2(embedder)

    await original_search.initialize()
    await optimized_search.initialize()

    test_queries = [
        "properties for sale in Austin Texas under $300000",
        "luxury homes with pools in Dallas",
        "market trends for investment properties",
        "affordable starter homes for first-time buyers",
        "commercial real estate opportunities",
    ]

    # Benchmark original implementation
    print("🔥 Benchmarking Original Implementation...")
    original_times = []
    for query in test_queries:
        start = time.time()
        results = await original_search.search(query, limit=15)
        elapsed = (time.time() - start) * 1000
        original_times.append(elapsed)
        print(f"  Query: {elapsed:.2f}ms ({len(results)} results)")

    # Benchmark optimized implementation
    print("\\n⚡ Benchmarking Optimized Implementation...")
    optimized_times = []
    for query in test_queries:
        start = time.time()
        results = await optimized_search.search_optimized(query, limit=15)
        elapsed = (time.time() - start) * 1000
        optimized_times.append(elapsed)
        print(f"  Query: {elapsed:.2f}ms ({len(results)} results)")

    # Performance comparison
    avg_original = np.mean(original_times)
    avg_optimized = np.mean(optimized_times)
    improvement = ((avg_original - avg_optimized) / avg_original) * 100

    print("\\n📊 Performance Summary:")
    print(f"  Original Average:  {avg_original:.2f}ms")
    print(f"  Optimized Average: {avg_optimized:.2f}ms")
    print(f"  Improvement:       {improvement:.1f}% faster")

    # Get detailed metrics
    print("\\n📈 Detailed Metrics:")
    stats = optimized_search.get_performance_stats()
    print(f"  Cache Hit Rate: {stats['embedding_metrics']['cache_hit_rate']:.2%}")
    print(f"  Avg DB Time: {stats['search_metrics']['avg_db_time_ms']:.2f}ms")


if __name__ == "__main__":
    asyncio.run(benchmark_optimizations())
