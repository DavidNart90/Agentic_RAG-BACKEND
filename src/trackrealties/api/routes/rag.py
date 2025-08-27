"""
API endpoints for RAG search functionality.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from ...models.search import (QueryRequest, SearchRequest, SearchResponse,
                              SearchResult)
from ...rag.llm_entity_extraction import \
    LLMEntityExtractor as RealEstateEntityExtractor
from ...rag.optimized_pipeline import EnhancedRAGPipeline
from ...rag.router import QueryRouter
from ...rag.synthesizer import ResponseSynthesizer

logger = logging.getLogger(__name__)
router = APIRouter()

pipeline = EnhancedRAGPipeline()


async def get_search_engine() -> EnhancedRAGPipeline:
    """Dependency to get an initialized EnhancedRAGPipeline instance."""
    if not pipeline.initialized:
        await pipeline.initialize()
    return pipeline


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_engine: EnhancedRAGPipeline = Depends(get_search_engine),
):
    """
    Perform a search using the RAG engine.
    """
    start_time = time.time()
    try:
        logger.info(f"Performing search for query: {request.query}")

        results: list[SearchResult] = await search_engine.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters,
        )

        end_time = time.time()
        search_time_ms = int((end_time - start_time) * 1000)

        return SearchResponse(
            query=request.query,
            search_type=request.search_type,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms,
            filters_applied=request.filters,
            sources_searched=["vector_db", "graph_db"],  # Placeholder
        )
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the search.") from e


@router.post("/query", response_model=SearchResponse)
async def intelligent_query(
    request: QueryRequest,
    search_engine: EnhancedRAGPipeline = Depends(get_search_engine),
):
    """
    Perform an intelligent search using a query router.
    """
    start_time = time.time()
    try:
        logger.info(f"Performing intelligent query: {request.query}")

        # Instantiate components
        query_router = QueryRouter()
        response_synthesizer = ResponseSynthesizer()

        # Route query to determine search strategy
        search_strategy = query_router.route_query(request.query)
        logger.info(f"Routed query to strategy: {search_strategy}")

        # Perform search using the chosen strategy
        search_results: list[SearchResult] = await search_engine.search(query=request.query)

        # Synthesize the response
        # Extracting 'content' from each SearchResult
        result_content = [result.content for result in search_results]
        synthesized_text = response_synthesizer.synthesize_response(query=request.query, search_results=result_content)

        # Create a single SearchResult for the synthesized response
        synthesized_result = SearchResult(
            result_id="synthesized-response",
            result_type="document",
            title="Synthesized Answer",
            content=synthesized_text,
            relevance_score=1.0,  # Max relevance for synthesized response
            source="RAG Engine",
        )

        end_time = time.time()
        search_time_ms = int((end_time - start_time) * 1000)

        return SearchResponse(
            query=request.query,
            search_type=search_strategy,
            results=[synthesized_result],
            total_results=1,
            search_time_ms=search_time_ms,
        )
    except Exception as e:
        logger.error(f"Error during intelligent query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the query.") from e


@router.post("/query-router")
async def query_router_diagnostics(request: QueryRequest):
    """Return router diagnostics for a query."""
    try:
        router = QueryRouter()
        extractor = RealEstateEntityExtractor()
        strategy = router.route_query(request.query)
        entities = await extractor.extract_entities(request.query)
        return {"strategy": strategy, "entities": entities}
    except Exception as e:
        logger.error(f"Query router diagnostics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to analyze query") from e
