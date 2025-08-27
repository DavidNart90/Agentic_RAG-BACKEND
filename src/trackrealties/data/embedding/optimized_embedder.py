"""
Optimized Embedding Pipeline for NeonDB with Advanced Caching and Batch Processing
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg
import numpy as np
import redis
import tiktoken

from src.trackrealties.core.config import get_settings
# Import your existing components
from src.trackrealties.data.embedding.openai_embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class EmbeddingMetrics:
    """Track embedding generation metrics"""

    total_requests: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    error_count: int = 0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def average_time_per_request(self) -> float:
        return self.total_time / self.total_requests if self.total_requests > 0 else 0.0


class AdvancedEmbeddingCache:
    """
    Advanced caching system with Redis backend and intelligent cache warming
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 86400 * 7):
        self.redis_url = str("redis://localhost:6379")
        self.ttl = ttl  # 7 days default
        self.redis_client = None
        self.local_cache = {}  # In-memory fallback
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0}


async def initialize(self):
    """Initialize Redis connection"""
    try:
        # Use synchronous Redis client (it's fine for caching)
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        # Test connection
        self.redis_client.ping()
        logger.info("Redis cache initialized successfully")
    except ImportError:
        logger.warning("redis not installed, using in-memory cache only")
        self.redis_client = None
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}, using in-memory cache")
        self.redis_client = None

    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate consistent cache key"""
        content = f"{model}:{text}"
        return f"embedding:{hashlib.sha256(content.encode()).hexdigest()}"

    async def get(self, text: str, model: str) -> Optional[Tuple[List[float], int]]:
        """Get embedding from cache"""
        cache_key = self._generate_cache_key(text, model)

        # Try Redis first
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    self.cache_stats["hits"] += 1
                    return data["embedding"], data["token_count"]
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}")

        # Fallback to local cache
        if cache_key in self.local_cache:
            self.cache_stats["hits"] += 1
            return self.local_cache[cache_key]

        self.cache_stats["misses"] += 1
        return None

    async def set(self, text: str, model: str, embedding: List[float], token_count: int):
        """Set embedding in cache"""
        cache_key = self._generate_cache_key(text, model)
        cache_data = {"embedding": embedding, "token_count": token_count, "timestamp": time.time()}

        # Store in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(cache_key, self.ttl, json.dumps(cache_data))
                self.cache_stats["sets"] += 1
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")

        # Store in local cache (with size limit)
        if len(self.local_cache) < 10000:  # Limit local cache size
            self.local_cache[cache_key] = (embedding, token_count)

    async def get_batch(self, texts: List[str], model: str) -> List[Optional[Tuple[List[float], int]]]:
        """Get multiple embeddings from cache efficiently"""
        cache_keys = [self._generate_cache_key(text, model) for text in texts]
        results = [None] * len(texts)

        # Try Redis pipeline for efficiency
        if self.redis_client:
            try:
                pipeline = self.redis_client.pipeline()
                for key in cache_keys:
                    pipeline.get(key)
                cached_results = await pipeline.execute()

                for i, cached_data in enumerate(cached_results):
                    if cached_data:
                        try:
                            data = json.loads(cached_data)
                            results[i] = (data["embedding"], data["token_count"])
                            self.cache_stats["hits"] += 1
                        except json.JSONDecodeError:
                            self.cache_stats["misses"] += 1
                    else:
                        self.cache_stats["misses"] += 1

                return results
            except Exception as e:
                logger.warning(f"Redis batch get error: {e}")

        # Fallback to individual gets
        for i, text in enumerate(texts):
            results[i] = await self.get(text, model)

        return results


class OptimizedEmbeddingPipeline:
    """
    Production-ready embedding pipeline with advanced optimizations
    """

    def __init__(
        self,
        batch_size: int = 50,
        max_concurrent_batches: int = 3,
        enable_cache_warming: bool = True,
        cache_ttl: int = 86400 * 7,
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_cache_warming = enable_cache_warming

        # Initialize components
        self.embedder = OpenAIEmbedder(batch_size=batch_size)
        self.cache = AdvancedEmbeddingCache(ttl=cache_ttl)
        self.metrics = EmbeddingMetrics()

        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(max_concurrent_batches)
        self.last_api_call = 0
        self.min_api_interval = 0.1  # Minimum seconds between API calls

        # Tokenizer for preprocessing
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Deduplication tracking
        self.processed_hashes: Set[str] = set()

    async def initialize(self):
        """Initialize all components"""
        await self.embedder.initialize()
        logger.info("Optimized embedding pipeline initialized")

    async def generate_embeddings_optimized(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None, deduplicate: bool = True
    ) -> Tuple[List[List[float]], List[int], EmbeddingMetrics]:
        """
        Generate embeddings with all optimizations enabled
        """
        start_time = time.time()
        self.metrics.total_requests += len(texts)

        # Step 1: Preprocessing and deduplication
        if deduplicate:
            texts, dedup_map = self._deduplicate_texts(texts)
            logger.debug(f"Deduplicated {len(texts)} unique texts from {len(dedup_map)} total")

        # Step 2: Filter by token count (avoid texts that are too long)
        valid_texts, valid_indices = self._filter_by_token_count(texts)
        if len(valid_texts) < len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} texts due to token limit")

        # Step 3: Batch cache lookup
        cached_results = await self.cache.get_batch(valid_texts, self.embedder.model)

        # Step 4: Identify texts that need embedding generation
        texts_to_generate = []
        missing_indices = []
        embeddings_result = [None] * len(valid_texts)
        token_counts_result = [0] * len(valid_texts)

        for i, (text, cached_result) in enumerate(zip(valid_texts, cached_results)):
            if cached_result:
                embeddings_result[i], token_counts_result[i] = cached_result
                self.metrics.cache_hits += 1
            else:
                texts_to_generate.append(text)
                missing_indices.append(i)

        # Step 5: Generate missing embeddings in optimized batches
        if texts_to_generate:
            logger.debug(f"Generating {len(texts_to_generate)} new embeddings")
            new_embeddings, new_token_counts = await self._generate_batch_with_rate_limiting(texts_to_generate)

            # Store new embeddings in cache
            cache_tasks = []
            for text, embedding, token_count in zip(texts_to_generate, new_embeddings, new_token_counts):
                cache_tasks.append(self.cache.set(text, self.embedder.model, embedding, token_count))

            if cache_tasks:
                await asyncio.gather(*cache_tasks, return_exceptions=True)

            # Insert new embeddings into results
            for i, missing_idx in enumerate(missing_indices):
                embeddings_result[missing_idx] = new_embeddings[i]
                token_counts_result[missing_idx] = new_token_counts[i]

            self.metrics.api_calls += len(texts_to_generate)
            self.metrics.total_tokens += sum(new_token_counts)

        # Step 6: Reconstruct full results if deduplication was used
        if deduplicate:
            full_embeddings = []
            full_token_counts = []
            for original_idx in dedup_map:
                full_embeddings.append(embeddings_result[original_idx])
                full_token_counts.append(token_counts_result[original_idx])
            embeddings_result = full_embeddings
            token_counts_result = full_token_counts

        # Step 7: Handle any invalid texts
        final_embeddings = []
        final_token_counts = []
        for i, text in enumerate(texts):
            if i < len(embeddings_result) and embeddings_result[i] is not None:
                final_embeddings.append(embeddings_result[i])
                final_token_counts.append(token_counts_result[i])
            else:
                # Fallback to zero vector
                final_embeddings.append([0.0] * self.embedder.dimensions)
                final_token_counts.append(0)
                self.metrics.error_count += 1

        # Update metrics
        self.metrics.total_time += time.time() - start_time

        return final_embeddings, final_token_counts, self.metrics

    def _deduplicate_texts(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """Remove duplicate texts and return mapping"""
        seen = {}
        unique_texts = []
        dedup_map = []

        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()

            if text_hash not in seen:
                seen[text_hash] = len(unique_texts)
                unique_texts.append(text)

            dedup_map.append(seen[text_hash])

        return unique_texts, dedup_map

    def _filter_by_token_count(self, texts: List[str], max_tokens: int = 8000) -> Tuple[List[str], List[int]]:
        """Filter texts that exceed token limit"""
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            token_count = len(self.tokenizer.encode(text))
            if token_count <= max_tokens:
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                logger.warning(f"Text {i} exceeds token limit ({token_count} > {max_tokens})")

        return valid_texts, valid_indices

    async def _generate_batch_with_rate_limiting(self, texts: List[str]) -> Tuple[List[List[float]], List[int]]:
        """Generate embeddings with proper rate limiting"""
        async with self.rate_limiter:
            # Ensure minimum interval between API calls
            time_since_last_call = time.time() - self.last_api_call
            if time_since_last_call < self.min_api_interval:
                await asyncio.sleep(self.min_api_interval - time_since_last_call)

            # Split into optimal batch sizes
            all_embeddings = []
            all_token_counts = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i: i + self.batch_size]

                try:
                    self.last_api_call = time.time()
                    embeddings, token_counts = await self.embedder.generate_embeddings_batch_optimized(batch)
                    all_embeddings.extend(embeddings)
                    all_token_counts.extend(token_counts)

                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch: {e}")
                    # Add fallback embeddings for failed batch
                    fallback_embeddings = [[0.0] * self.embedder.dimensions] * len(batch)
                    fallback_token_counts = [0] * len(batch)
                    all_embeddings.extend(fallback_embeddings)
                    all_token_counts.extend(fallback_token_counts)
                    self.metrics.error_count += len(batch)

            return all_embeddings, all_token_counts

    async def warm_cache_for_common_queries(self, common_texts: List[str]):
        """Pre-warm cache with commonly used texts"""
        if not self.enable_cache_warming:
            return

        logger.info(f"Warming cache with {len(common_texts)} common texts")

        # Check which texts are not cached
        cached_results = await self.cache.get_batch(common_texts, self.embedder.model)
        texts_to_warm = [text for text, cached in zip(common_texts, cached_results) if cached is None]

        if texts_to_warm:
            logger.info(f"Pre-generating {len(texts_to_warm)} embeddings for cache warming")
            await self.generate_embeddings_optimized(texts_to_warm)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "embedding_metrics": asdict(self.metrics),
            "cache_stats": self.cache.cache_stats,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "average_time_per_request": self.metrics.average_time_per_request,
            "processed_unique_hashes": len(self.processed_hashes),
        }


class NeonDBEmbeddingManager:
    """
    Specialized manager for NeonDB vector operations with optimized batch inserts
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(self.connection_string, min_size=5, max_size=20, command_timeout=60)
        logger.info("NeonDB connection pool initialized")

    async def batch_upsert_embeddings(
        self, chunks_data: List[Dict[str, Any]], table_name: str = "property_chunks_enhanced"
    ) -> List[str]:
        """
        Efficiently batch upsert chunks with embeddings to NeonDB
        """
        if not chunks_data:
            return []

        async with self.pool.acquire() as conn:
            # Prepare batch insert query with ON CONFLICT handling
            if table_name == "property_chunks_enhanced":
                query = """
                INSERT INTO property_chunks_enhanced (
                    property_listing_id, content, chunk_index, token_count, 
                    embedding, metadata, chunk_type, semantic_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (property_listing_id, chunk_index) 
                DO UPDATE SET
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    chunk_type = EXCLUDED.chunk_type,
                    semantic_score = EXCLUDED.semantic_score,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """
            else:  # market_chunks_enhanced
                query = """
                INSERT INTO market_chunks_enhanced (
                    market_data_id, content, chunk_index, token_count,
                    embedding, metadata, chunk_type, semantic_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (market_data_id, chunk_index)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    chunk_type = EXCLUDED.chunk_type,
                    semantic_score = EXCLUDED.semantic_score,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """

            # Prepare batch data
            batch_data = []
            for chunk_data in chunks_data:
                embedding_vector = chunk_data["embedding"]
                if isinstance(embedding_vector, list):
                    # Convert to PostgreSQL array format for pgvector
                    embedding_str = "[" + ",".join(map(str, embedding_vector)) + "]"
                else:
                    embedding_str = str(embedding_vector)

                batch_data.append(
                    (
                        chunk_data.get("parent_id"),
                        chunk_data["content"],
                        chunk_data.get("chunk_index", 0),
                        chunk_data.get("token_count", 0),
                        embedding_str,
                        json.dumps(chunk_data.get("metadata", {})),
                        chunk_data.get("chunk_type", "unknown"),
                        chunk_data.get("semantic_score", 0.0),
                    )
                )

            # Execute batch insert
            try:
                result_ids = await conn.fetch(query, *zip(*batch_data))
                inserted_ids = [str(row["id"]) for row in result_ids]
                logger.info(f"Successfully upserted {len(inserted_ids)} chunks to {table_name}")
                return inserted_ids

            except Exception as e:
                logger.error(f"Failed to batch upsert to {table_name}: {e}")
                raise

    async def optimize_vector_search_performance(self):
        """Optimize NeonDB for vector search performance"""
        async with self.pool.acquire() as conn:
            # Create optimized indexes if they don't exist
            optimization_queries = [
                # HNSW index for vector similarity (most important)
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_enhanced_embedding_hnsw 
                ON property_chunks_enhanced USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
                """,
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_enhanced_embedding_hnsw 
                ON market_chunks_enhanced USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
                """,
                # Composite indexes for filtered searches
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_enhanced_type_score 
                ON property_chunks_enhanced (chunk_type, semantic_score DESC) 
                WHERE embedding IS NOT NULL;
                """,
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_enhanced_type_score 
                ON market_chunks_enhanced (chunk_type, semantic_score DESC) 
                WHERE embedding IS NOT NULL;
                """,
                # Optimize for metadata filtering
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_enhanced_metadata_gin 
                ON property_chunks_enhanced USING gin (metadata);
                """,
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_enhanced_metadata_gin 
                ON market_chunks_enhanced USING gin (metadata);
                """,
            ]

            for query in optimization_queries:
                try:
                    await conn.execute(query)
                    logger.info("Successfully executed optimization query")
                except Exception as e:
                    logger.warning(f"Optimization query failed (may already exist): {e}")

    async def advanced_vector_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        boost_semantic_score: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Advanced vector search with multiple optimizations
        """
        async with self.pool.acquire() as conn:
            # Convert embedding to PostgreSQL format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Build dynamic filter conditions
            filter_conditions = []
            filter_params = [embedding_str, similarity_threshold, limit]
            param_count = 3

            if filters:
                for key, value in filters.items():
                    param_count += 1
                    if key == "chunk_type":
                        filter_conditions.append(f"chunk_type = ${param_count}")
                    elif key == "min_semantic_score":
                        filter_conditions.append(f"semantic_score >= ${param_count}")
                    else:
                        filter_conditions.append(f"metadata->>'{key}' = ${param_count}")
                    filter_params.append(value)

            filter_clause = " AND " + " AND ".join(filter_conditions) if filter_conditions else ""

            # Advanced search query with semantic score boosting
            if boost_semantic_score:
                # Combine cosine similarity with semantic score
                search_query = f"""
                WITH combined_results AS (
                    SELECT 
                        'property_' || id::text as result_id,
                        content,
                        'property' as result_type,
                        1 - (embedding <=> $1::vector) AS similarity,
                        semantic_score,
                        -- Combined score: 70% similarity + 30% semantic score
                        (0.7 * (1 - (embedding <=> $1::vector)) + 0.3 * semantic_score) as combined_score,
                        metadata,
                        chunk_type
                    FROM property_chunks_enhanced
                    WHERE 
                        embedding IS NOT NULL
                        AND (1 - (embedding <=> $1::vector)) >= $2
                        {filter_clause}

                    UNION ALL

                    SELECT 
                        'market_' || id::text as result_id,
                        content,
                        'market_data' as result_type,
                        1 - (embedding <=> $1::vector) AS similarity,
                        semantic_score,
                        (0.7 * (1 - (embedding <=> $1::vector)) + 0.3 * semantic_score) as combined_score,
                        metadata,
                        chunk_type
                    FROM market_chunks_enhanced
                    WHERE 
                        embedding IS NOT NULL
                        AND (1 - (embedding <=> $1::vector)) >= $2
                        {filter_clause}
                )
                SELECT 
                    result_id,
                    content,
                    result_type,
                    similarity,
                    semantic_score,
                    combined_score,
                    metadata,
                    chunk_type
                FROM combined_results
                ORDER BY combined_score DESC
                LIMIT $3
                """
            else:
                # Standard similarity search
                search_query = f"""
                SELECT 
                    'property_' || id::text as result_id,
                    content,
                    'property' as result_type,
                    1 - (embedding <=> $1::vector) AS similarity,
                    semantic_score,
                    metadata,
                    chunk_type
                FROM property_chunks_enhanced
                WHERE 
                    embedding IS NOT NULL
                    AND (1 - (embedding <=> $1::vector)) >= $2
                    {filter_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """

            try:
                results = await conn.fetch(search_query, *filter_params)

                return [
                    {
                        "id": row["result_id"],
                        "content": row["content"],
                        "type": row["result_type"],
                        "similarity": float(row["similarity"]),
                        "semantic_score": float(row["semantic_score"]),
                        "combined_score": float(row.get("combined_score", row["similarity"])),
                        "metadata": row["metadata"],
                        "chunk_type": row["chunk_type"],
                    }
                    for row in results
                ]

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []


# Usage Example and Integration
async def example_usage():
    """Example of how to use the optimized pipeline"""

    # Initialize components
    pipeline = OptimizedEmbeddingPipeline(batch_size=50, max_concurrent_batches=3, enable_cache_warming=True)
    await pipeline.initialize()

    # Initialize NeonDB manager
    neon_manager = NeonDBEmbeddingManager("postgresql://user:pass@host:5432/db")
    await neon_manager.initialize()

    # Sample property data
    sample_texts = [
        "Beautiful 3BR/2BA home with granite countertops and hardwood floors",
        "Luxury condo in downtown area with city views and modern amenities",
        "Family-friendly neighborhood with excellent schools and parks nearby",
    ]

    # Generate optimized embeddings
    embeddings, token_counts, metrics = await pipeline.generate_embeddings_optimized(sample_texts, deduplicate=True)

    print(f"Generated embeddings with {metrics.cache_hit_rate:.2%} cache hit rate")
    print(f"Total time: {metrics.total_time:.2f}s, API calls: {metrics.api_calls}")

    # Prepare chunk data for database
    chunks_data = []
    for i, (text, embedding, token_count) in enumerate(zip(sample_texts, embeddings, token_counts)):
        chunks_data.append(
            {
                "parent_id": f"property_{i}",
                "content": text,
                "chunk_index": 0,
                "token_count": token_count,
                "embedding": embedding,
                "metadata": {"type": "property_overview"},
                "chunk_type": "property_core",
                "semantic_score": 0.85,
            }
        )

    # Batch insert to NeonDB
    inserted_ids = await neon_manager.batch_upsert_embeddings(chunks_data, "property_chunks_enhanced")
    print(f"Inserted {len(inserted_ids)} chunks to database")

    # Optimize database for vector search
    await neon_manager.optimize_vector_search_performance()

    # Perform advanced search
    query_text = "3 bedroom house with modern kitchen"
    query_embeddings, _, _ = await pipeline.generate_embeddings_optimized([query_text])

    search_results = await neon_manager.advanced_vector_search(
        query_embeddings[0],
        limit=5,
        similarity_threshold=0.6,
        filters={"chunk_type": "property_core"},
        boost_semantic_score=True,
    )

    print(f"Found {len(search_results)} relevant results")
    for result in search_results:
        print(f"- {result['content'][:100]}... (similarity: {result['similarity']:.3f})")


if __name__ == "__main__":
    asyncio.run(example_usage())
