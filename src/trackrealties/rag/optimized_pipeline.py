"""
RAG Pipeline Integration with Smart Search Router
Replace existing search.py implementation with optimized routing
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.trackrealties.analytics.search import (SearchAnalytics,
                                                search_analytics)
from src.trackrealties.core.database import db_pool
from src.trackrealties.core.graph import graph_manager
from src.trackrealties.models.agent import ValidationResult
from src.trackrealties.models.search import SearchResult
from src.trackrealties.rag.embedders import DefaultEmbedder
from src.trackrealties.rag.synthesizer import ResponseSynthesizer
from src.trackrealties.validation.hallucination import \
    RealEstateHallucinationDetector

from .smart_search import (FixedGraphSearch, RealEstateEntityExtractor,
                           SmartSearchRouter)

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG pipeline processing - expected by agents."""

    search_results: List[Any]
    market_context: Dict[str, Any]
    tools_used: List[str]
    confidence_score: float
    response_content: str
    search_strategy: str = "enhanced_rag"
    session_id: str = ""
    validation: Optional[Dict[str, Any]] = None


class GreetingDetector:
    """
    Sophisticated greeting detection that handles various greeting patterns
    """

    # Comprehensive greeting patterns
    GREETING_PATTERNS = [
        # Basic greetings
        r"\b(hi|hey|hello|greetings|good\s+(morning|afternoon|evening|day))\b",
        # How are you variations
        r"\bhow\s+(are|r)\s+(you|u|ya)\b",
        r"\bhow\'s\s+it\s+going\b",
        r"\bwhat\'s\s+up\b",
        r"\bwhats\s+up\b",
        # Agent specific greetings
        r"\b(hi|hello|hey)\s+agent\b",
        r"\bagent\s*,?\s*(hi|hello|hey)\b",
        # Polite greetings
        r"\bgood\s+to\s+(see|meet)\s+you\b",
        r"\bnice\s+to\s+meet\s+you\b",
        # Start of conversation
        r"^(well\s+)?(hi|hello|hey)\s*[,!.]?\s*$",
        r"^greetings\s*[,!.]?\s*$",
        # Question greetings
        r"\bhow\s+do\s+you\s+do\b",
        r"\bhow\s+are\s+things\b",
        # Simple standalone greetings
        r"^(yo|sup|hiya|howdy)\s*[,!.]?\s*$",
        # Exit Greetings Thank You, Bye, Goodbye, See you later,see ya
        r"\b(thank\s+you|thanks|bye|goodbye|see\s+you\s+later|see\s+ya)\b",
        # Casual greetings
        r"\b(what\'s\s+good|what\'s\s+new|what\'s\s+crackin|what\'s\s+shakin)\b",
    ]

    # Compile patterns for efficiency
    COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in GREETING_PATTERNS]

    # Context words that might appear with greetings but don't change the intent
    GREETING_CONTEXT_WORDS = {
        "there",
        "mate",
        "friend",
        "pal",
        "buddy",
        "folks",
        "everyone",
        "team",
        "all",
        "guys",
        "hope",
        "doing",
        "well",
        "today",
        "morning",
        "afternoon",
        "evening",
        "night",
        "thanks",
        "appreciate",
        "welcome",
        "pleasure",
        "bye",
        "goodbye",
    }

    @classmethod
    def is_greeting(cls, query: str) -> bool:
        """
        Check if the query is a greeting.
        """
        query = query.strip().lower()

        # Check against compiled patterns
        for pattern in cls.COMPILED_PATTERNS:
            if pattern.search(query):
                return True

        # Check if query consists mostly of greeting context words
        words = set(query.split())
        greeting_words = words.intersection(cls.GREETING_CONTEXT_WORDS)
        if len(greeting_words) / len(words) > 0.5:  # If more than half are greeting words
            return True

        return False

    @classmethod
    def extract_greeting_intent(cls, query: str) -> Dict[str, Any]:
        """
        Extract greeting intent and metadata.
        """
        is_greeting = cls.is_greeting(query)

        if is_greeting:
            return {
                "is_greeting": True,
                "result": GreetingSearchResult(),
                "metadata": {"intent": "greeting", "confidence": 1.0, "skip_search": True},
            }

        return {
            "is_greeting": False,
            "result": None,
            "metadata": {"intent": "other", "confidence": 0.0, "skip_search": False},
        }


class GreetingSearchResult(SearchResult):
    """
    Special search result for greetings that signals the agent to use greeting prompt
    """

    def __init__(self):
        super().__init__(
            result_id="greeting_response",
            content="GREETING_DETECTED",
            result_type="document",  # Changed from "greeting" to valid type
            relevance_score=1.0,
            title="Greeting Response Required",
            source="system",
            metadata={
                "instruction": "Use GREETINGS_PROMPT to generate a welcoming response",
                "skip_search": True,
                "response_type": "greeting",
            },
        )


class OptimizedVectorSearch:
    """
    Optimized vector search with better error handling and performance
    """

    def __init__(self):
        self.embedder = DefaultEmbedder()
        self.initialized = False

    async def initialize(self):
        """Initialize the vector search client."""
        await db_pool.initialize()
        await self.embedder.initialize()
        self.initialized = True
        logger.info("Optimized vector search initialized")

    async def search(
        self,
        query: str,
        limit: int = 60,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.2,  # Lowered from 0.3 to capture more property results
    ) -> List[SearchResult]:
        """
        Search for relevant content using vector similarity with optimizations
        """
        if not self.initialized:
            await self.initialize()

        try:
            query_embedding = await self.embedder.embed_query(query)
            # Convert embedding to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Build filter query more efficiently
            filter_clauses = []
            filter_values = []
            if filters:
                for key, value in filters.items():
                    filter_clauses.append(f"metadata->>'{key}' = ${len(filter_values) + 4}")
                    filter_values.append(value)

            filter_query = " AND " + " AND ".join(filter_clauses) if filter_clauses else ""

            async with db_pool.acquire() as conn:
                # Determine search strategy based on query content
                query_lower = query.lower()
                is_property_search = any(
                    keyword in query_lower
                    for keyword in [
                        "property",
                        "properties",
                        "house",
                        "houses",
                        "home",
                        "homes",
                        "listing",
                        "listings",
                        "bedroom",
                        "bathroom",
                        "sqft",
                        "square feet",
                        "garage",
                        "yard",
                        "pool",
                    ]
                )
                is_rental_search = any(
                    keyword in query_lower
                    for keyword in ["rental", "rent", "lease", "monthly", "apartment", "apartments", "unit", "units"]
                )
                is_market_search = any(
                    keyword in query_lower
                    for keyword in [
                        "market",
                        "median price",
                        "average price",
                        "sales volume",
                        "market trends",
                        "price analysis",
                        "market data",
                        "market overview",
                    ]
                )

                # Expand property search to include rental searches
                is_property_search = is_property_search or is_rental_search

                # NEW: Detect investment queries for dual-table search
                is_investment_query = any(
                    keyword in query_lower for keyword in ["invest", "investment", "budget", "have $", "spend $"]
                ) and any(keyword in query_lower for keyword in ["property", "properties", "real estate"])

                # Extract budget for investment queries
                investment_budget = None
                if is_investment_query:
                    budget_pattern = r"(?:have|invest|budget|spend).*?\$?(\d+(?:,\d{3})*|\d{1,3}(?:,\d{3})*)(?:k|m)?"
                    budget_match = re.search(budget_pattern, query_lower)
                    if budget_match:
                        investment_budget = int(budget_match.group(1).replace(",", ""))
                        # Check for k/m ONLY immediately after the number (in the captured group + next char)
                        full_match = budget_match.group(0)
                        number_part = budget_match.group(1)
                        # Find where the number ends in the full match and check the next character
                        number_start_pos = full_match.find(number_part)
                        number_end_pos = number_start_pos + len(number_part)
                        next_char = (
                            full_match[number_end_pos: number_end_pos + 1] if number_end_pos < len(full_match) else ""
                        )

                        if next_char.lower() == "k":
                            investment_budget *= 1000
                        elif next_char.lower() == "m":
                            investment_budget *= 1000000
                    # Note: If no budget match found, investment_budget remains None

                # Build price filter based on query
                price_filter = self._build_price_filter(query, is_property_search)

                # DUAL-TABLE INVESTMENT SEARCH: Query both property_chunks_enhanced AND market_chunks_enhanced
                if is_investment_query and investment_budget:
                    logger.info(f"Dual-table investment search for budget: ${investment_budget:,}")

                    # Search investment properties (limit//2 results)
                    min_price = int(investment_budget * 0.9)
                    max_price = int(investment_budget * 1.1)

                    property_investment_results = await conn.fetch(
                        f"""
                        SELECT DISTINCT 
                            property_listing_id,
                            MAX(1 - (embedding <=> $1)) as max_similarity
                        FROM property_chunks_enhanced 
                        WHERE (1 - (embedding <=> $1)) > $2 
                            AND (
                                (chunk_type = 'financial_analysis' 
                                 AND content ~ 'List Price: \\$([0-9,]+)'
                                 AND regexp_replace(
                                     substring(content FROM 'List Price: \\$([0-9,]+)'),
                                     ',', '', 'g'
                                 )::numeric BETWEEN {min_price} AND {max_price})
                            )
                            {filter_query}
                        GROUP BY property_listing_id
                        ORDER BY max_similarity DESC
                        LIMIT $3
                        """,
                        embedding_str,
                        threshold,
                        limit // 2,
                        *filter_values,
                    )

                    # Get all chunks for matching investment properties
                    investment_property_results = []
                    for prop_match in property_investment_results:
                        property_chunks = await conn.fetch(
                            """
                            SELECT 
                                'investment_property_' || id::text as result_id,
                                content,
                                'investment_property' as result_type,
                                (1 - (embedding <=> $1)) as similarity,
                                'Investment Property - ' || chunk_type as title,
                                'Investment Property Database' as source,
                                jsonb_build_object(
                                    'property_id', property_listing_id::text,
                                    'chunk_type', chunk_type,
                                    'budget_match', $3::text,
                                    'table_source', 'property_chunks_enhanced'
                                ) as metadata
                            FROM property_chunks_enhanced 
                            WHERE property_listing_id = $2
                            ORDER BY similarity DESC
                            """,
                            embedding_str,
                            prop_match["property_listing_id"],
                            f"${investment_budget:,}",
                        )
                        investment_property_results.extend(property_chunks)

                    # Search market analysis data (limit//2 results) - Texas specific
                    texas_filter = (
                        "AND (content ILIKE '%texas%' OR content ILIKE '%tx%' OR content ILIKE '%san antonio%' OR content ILIKE '%dallas%' OR content ILIKE '%houston%' OR content ILIKE '%austin%')"
                        if "texas" in query_lower
                        else ""
                    )

                    market_investment_results = await conn.fetch(
                        f"""
                        SELECT 
                            'investment_market_' || id::text as result_id,
                            content,
                            'market_analysis' as result_type,
                            (1 - (embedding <=> $1)) as similarity,
                            'Market Analysis - ' || chunk_type as title,
                            'Market Intelligence Database' as source,
                            jsonb_build_object(
                                'market_data_id', market_data_id::text,
                                'chunk_type', chunk_type,
                                'budget_context', $3::text,
                                'table_source', 'market_chunks_enhanced'
                            ) as metadata
                        FROM market_chunks_enhanced 
                        WHERE (1 - (embedding <=> $1)) > $2 {texas_filter}
                        ORDER BY similarity DESC
                        LIMIT $4
                        """,
                        embedding_str,
                        threshold * 0.6,  # Even lower threshold for market data
                        f"${investment_budget:,}",
                        10,  # Get more market chunks
                        *filter_values,
                    )

                    # Combine investment results and apply final limit
                    results = investment_property_results + market_investment_results
                    results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)[:limit]
                    logger.info(
                        f"Dual-table investment search: {len(investment_property_results)} properties + {len(market_investment_results)} market chunks = {len(results)} total (limited to {limit})"
                    )

                elif is_property_search and not is_market_search:
                    # Property-focused search: COMPREHENSIVE APPROACH - get all chunk types per property
                    # Step 1: Find top matching properties across all chunk types
                    logger.info(
                        f"Comprehensive property search - threshold: {threshold}, price_filter: '{price_filter}'"
                    )

                    property_matches = await conn.fetch(
                        f"""
                        SELECT DISTINCT 
                            property_listing_id,
                            MAX(1 - (embedding <=> $1)) as max_similarity,
                            COUNT(*) as chunk_count
                        FROM property_chunks_enhanced 
                        WHERE (1 - (embedding <=> $1)) > $2 {filter_query} {price_filter}
                        GROUP BY property_listing_id
                        ORDER BY max_similarity DESC
                        LIMIT $3
                        """,
                        embedding_str,
                        threshold,
                        limit,
                        *filter_values,
                    )

                    logger.info(f"Found {len(property_matches)} property matches")

                    # Step 2: Get ALL chunks for matching properties in ONE OPTIMIZED QUERY
                    logger.info(f"Getting chunks for {len(property_matches)} properties in batched query")

                    # Create a single query with all property IDs to eliminate N+1 queries
                    property_ids = [str(match["property_listing_id"]) for match in property_matches]
                    property_ids_str = "'" + "','".join(property_ids) + "'"

                    comprehensive_results = await conn.fetch(
                        f"""
                        WITH ranked_chunks AS (
                            SELECT 
                                'property_' || id::text as result_id,
                                content,
                                'property_listing' as result_type,
                                (1 - (embedding <=> $1)) as similarity,
                                COALESCE(metadata->>'address', metadata->>'title', 'Property Listing') as title,
                                'Property Database' as source,
                                metadata,
                                chunk_type,
                                property_listing_id,
                                ROW_NUMBER() OVER (PARTITION BY property_listing_id ORDER BY (1 - (embedding <=> $1)) DESC) as chunk_rank
                            FROM property_chunks_enhanced 
                            WHERE property_listing_id::text IN ({property_ids_str})
                            AND embedding IS NOT NULL
                        )
                        SELECT 
                            result_id, content, result_type, similarity, title, source, metadata, chunk_type, property_listing_id
                        FROM ranked_chunks
                        WHERE chunk_rank <= 6  -- Get top 6 chunks per property
                        ORDER BY similarity DESC
                        """,
                        embedding_str,
                    )
                    logger.info(
                        f"Retrieved {len(comprehensive_results)} chunks for {len(property_matches)} properties in single query"
                    )

                    # Apply final limit to comprehensive results - ensure we don't exceed the requested limit
                    results = sorted(comprehensive_results, key=lambda x: x.get("similarity", 0), reverse=True)[:limit]
                    logger.info(
                        f"Total comprehensive results: {
                            len(comprehensive_results)} -> limited to {len(results)} (limit: {limit})"
                    )
                elif is_market_search and not is_property_search:
                    # Market-focused search: prioritize market_chunks_enhanced
                    results = await conn.fetch(
                        f"""
                        SELECT DISTINCT
                            'market_' || id::text as result_id,
                            content,
                            'market_data' as result_type,
                            1 - (embedding <=> $1) AS similarity,
                            COALESCE(metadata->>'region_name', metadata->>'title', 'Market Data') as title,
                            'Market Database' as source,
                            metadata
                        FROM market_chunks_enhanced
                        WHERE (1 - (embedding <=> $1)) > $2 {filter_query} {price_filter}
                        ORDER BY similarity DESC
                        LIMIT $3
                        """,
                        embedding_str,
                        threshold,
                        limit,
                        *filter_values,
                    )
                else:
                    # General search or mixed query: search both tables but prioritize by relevance
                    # For mixed queries, apply appropriate price filter based on which table we're searching
                    property_price_filter = (
                        self._build_price_filter(query, True)
                        if any(keyword in query_lower for keyword in ["under", "below", "over", "above", "$"])
                        else ""
                    )
                    market_price_filter = (
                        self._build_price_filter(query, False)
                        if any(keyword in query_lower for keyword in ["under", "below", "over", "above", "$"])
                        else ""
                    )

                    results = await conn.fetch(
                        f"""
                        (
                            SELECT DISTINCT
                                'property_' || id::text as result_id,
                                content,
                                'property_listing' as result_type,
                                1 - (embedding <=> $1) AS similarity,
                                COALESCE(metadata->>'address', metadata->>'title', 'Property Listing') as title,
                                'Property Database' as source,
                                metadata
                            FROM property_chunks_enhanced
                            WHERE (1 - (embedding <=> $1)) > $2 {filter_query} {property_price_filter}
                            ORDER BY similarity DESC
                            LIMIT $3
                        )
                        UNION ALL
                        (
                            SELECT DISTINCT
                                'market_' || id::text as result_id,
                                content,
                                'market_data' as result_type,
                                1 - (embedding <=> $1) AS similarity,
                                COALESCE(metadata->>'region_name', metadata->>'title', 'Market Data') as title,
                                'Market Database' as source,
                                metadata
                            FROM market_chunks_enhanced
                            WHERE (1 - (embedding <=> $1)) > $2 {filter_query} {market_price_filter}
                            ORDER BY similarity DESC
                            LIMIT $3
                        )
                        ORDER BY similarity DESC
                        LIMIT $3
                        """,
                        embedding_str,
                        threshold,
                        limit,
                        *filter_values,
                    )

            return [
                SearchResult(
                    result_id=row["result_id"],
                    content=row["content"],
                    result_type=(
                        row["result_type"]
                        if row["result_type"] in ["market_data", "property_listing", "document", "graph_fact"]
                        else "property_listing"
                    ),
                    relevance_score=row["similarity"],
                    title=row.get("title") or "No Title",
                    source=row.get("source", "No Source"),
                    metadata={
                        # Parse JSON metadata from database
                        **(
                            json.loads(row.get("metadata"))
                            if row.get("metadata") and isinstance(row.get("metadata"), str)
                            else row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                        ),
                        "chunk_type": row.get("chunk_type"),
                        "property_listing_id": row.get("property_listing_id"),
                    },
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _extract_price_range(self, query: str) -> Optional[Tuple[int, int]]:
        """Extract price range from query for filtering"""
        query_lower = query.lower()

        # Handle "under $X" or "below $X"
        under_pattern = r"(?:under|below|less\s+than)\s*\$?(\d+(?:,\d{3})*|\d+)(?:k|m)?"
        under_match = re.search(under_pattern, query_lower)
        if under_match:
            price_str = under_match.group(1).replace(",", "")
            price = int(price_str)
            # Check if 'k' or 'm' appears after the number in the original match
            full_match = under_match.group(0)
            if "k" in full_match and len(price_str) <= 3:  # Only apply k multiplier to small numbers
                price *= 1000
            elif "m" in full_match and len(price_str) <= 3:  # Only apply m multiplier to small numbers
                price *= 1000000
            return (0, price)

        # Handle "over $X" or "above $X"
        over_pattern = r"(?:over|above|more\s+than)\s*\$?(\d+(?:,\d{3})*|\d+)(?:k|m)?"
        over_match = re.search(over_pattern, query_lower)
        if over_match:
            price_str = over_match.group(1).replace(",", "")
            price = int(price_str)
            # Check if 'k' or 'm' appears after the number in the original match
            full_match = over_match.group(0)
            if "k" in full_match and len(price_str) <= 3:  # Only apply k multiplier to small numbers
                price *= 1000
            elif "m" in full_match and len(price_str) <= 3:  # Only apply m multiplier to small numbers
                price *= 1000000
            return (price, 10000000)  # Set upper limit to 10M

        # Handle "between $X and $Y" or "between $X-$Y"
        between_pattern = r"between\s*\$?(\d+(?:,\d{3})*|\d+)(?:k|m)?\s*(?:and|to|-)\s*\$?(\d+(?:,\d{3})*|\d+)(?:k|m)?"
        between_match = re.search(between_pattern, query_lower)
        if between_match:
            price1_str = between_match.group(1).replace(",", "")
            price2_str = between_match.group(2).replace(",", "")
            price1 = int(price1_str)
            price2 = int(price2_str)
            full_match = between_match.group(0)
            # Apply k or m multiplier only to numbers that appear with 'k' or 'm'
            if "k" in full_match:
                if len(price1_str) <= 3:
                    price1 *= 1000
                if len(price2_str) <= 3:
                    price2 *= 1000
            elif "m" in full_match:
                if len(price1_str) <= 3:
                    price1 *= 1000000
                if len(price2_str) <= 3:
                    price2 *= 1000000
            return (min(price1, price2), max(price1, price2))

        # Handle specific amount like "$300,000" or "I have $300,000" - create range around it
        investment_pattern = r"(?:have|invest|budget|spend).*?\$(\d+(?:,\d{3})*|\d+)(?:k|m)?"
        investment_match = re.search(investment_pattern, query_lower)
        if investment_match:
            price_str = investment_match.group(1).replace(",", "")
            price = int(price_str)
            full_match = investment_match.group(0)
            if "k" in full_match and len(price_str) <= 3:
                price *= 1000
            elif "m" in full_match and len(price_str) <= 3:
                price *= 1000000
            # For investment queries, search for properties within budget (±10%)
            margin = price * 0.1
            return (int(price - margin), int(price))

        # Handle general price mentions like "$300,000"
        amount_pattern = r"\$(\d+(?:,\d{3})*|\d+)(?:k|m)?"
        amount_match = re.search(amount_pattern, query_lower)
        if amount_match:
            price_str = amount_match.group(1).replace(",", "")
            price = int(price_str)
            full_match = amount_match.group(0)
            if "k" in full_match and len(price_str) <= 3:
                price *= 1000
            elif "m" in full_match and len(price_str) <= 3:
                price *= 1000000
            # Create range ±20% around the specified amount
            margin = price * 0.2
            return (int(price - margin), int(price + margin))

        return None

    def _build_price_filter(self, query: str, is_property_search: bool) -> str:
        """Build price filter SQL clause based on query and search type"""
        price_range = self._extract_price_range(query)
        if not price_range:
            return ""

        min_price, max_price = price_range

        # Check if this is a rental query
        query_lower = query.lower()
        is_rental_query = any(
            keyword in query_lower for keyword in ["rental", "rent", "lease", "monthly", "per month", "month"]
        )

        if is_property_search:
            if is_rental_query:
                # Filter for rental properties with rental history events
                if min_price > 0 and max_price < 10000000:
                    # Range filter for rentals
                    return f"""
                    AND (
                        (chunk_type = 'general' 
                         AND (content ~ 'Event: Rental Listing' OR content ~ 'Event: Sale Listing')
                         AND content ~ 'Price: \\$([0-9,]+\\.00)' 
                         AND regexp_replace(
                             substring(content FROM 'Price: \\$([0-9,]+)\\.00'), 
                             ',', '', 'g'
                         )::numeric BETWEEN {min_price} AND {max_price})
                    )
                    """
                elif max_price < 10000000:
                    # Under X filter for rentals
                    return f"""
                    AND (
                        (chunk_type = 'general' 
                         AND (content ~ 'Event: Rental Listing' OR content ~ 'Event: Sale Listing')
                         AND content ~ 'Price: \\$([0-9,]+\\.00)' 
                         AND regexp_replace(
                             substring(content FROM 'Price: \\$([0-9,]+)\\.00'), 
                             ',', '', 'g'
                         )::numeric <= {max_price})
                    )
                    """
                else:
                    # Over X filter for rentals
                    return f"""
                    AND (
                        (chunk_type = 'general' 
                         AND (content ~ 'Event: Rental Listing' OR content ~ 'Event: Sale Listing')
                         AND content ~ 'Price: \\$([0-9,]+\\.00)' 
                         AND regexp_replace(
                             substring(content FROM 'Price: \\$([0-9,]+)\\.00'), 
                             ',', '', 'g'
                         )::numeric >= {min_price})
                    )
                    """
            else:
                # Original property price filtering for sales
                if min_price > 0 and max_price < 10000000:
                    # Range filter
                    return f"""
                    AND (
                        (metadata->'extracted_entities'->>'price' IS NOT NULL 
                         AND (metadata->'extracted_entities'->>'price')::numeric BETWEEN {min_price} AND {max_price})
                        OR 
                        (chunk_type = 'financial_analysis' 
                         AND content ~ 'List Price: \\$\\$?([0-9,]+)' 
                         AND regexp_replace(
                             substring(content FROM 'List Price: \\$([0-9,]+)'), 
                             ',', '', 'g'
                         )::numeric BETWEEN {min_price} AND {max_price})
                    )
                    """
                elif max_price < 10000000:
                    # Under X filter
                    return f"""
                    AND (
                        (metadata->'extracted_entities'->>'price' IS NOT NULL 
                         AND (metadata->'extracted_entities'->>'price')::numeric <= {max_price})
                        OR 
                        (chunk_type = 'financial_analysis' 
                         AND content ~ 'List Price: \\$\\$?([0-9,]+)' 
                         AND regexp_replace(
                             substring(content FROM 'List Price: \\$([0-9,]+)'), 
                             ',', '', 'g'
                         )::numeric <= {max_price})
                    )
                    """
                else:
                    # Over X filter
                    return f"""
                    AND (
                        (metadata->'extracted_entities'->>'price' IS NOT NULL 
                         AND (metadata->'extracted_entities'->>'price')::numeric >= {min_price})
                        OR 
                        (chunk_type = 'financial_analysis' 
                         AND content ~ 'List Price: \\$\\$?([0-9,]+)' 
                         AND regexp_replace(
                             substring(content FROM 'List Price: \\$([0-9,]+)'), 
                             ',', '', 'g'
                         )::numeric >= {min_price})
                    )
                    """
        else:
            # Filter market data by median price
            if min_price > 0 and max_price < 10000000:
                # Range filter
                return f"""
                AND (
                    (metadata->'extracted_entities'->>'median_price' IS NOT NULL 
                     AND (metadata->'extracted_entities'->>'median_price')::numeric BETWEEN {min_price} AND {max_price})
                    OR 
                    (content ~ 'Median Price: \\$\\$?([0-9,]+)' 
                     AND regexp_replace(
                         substring(content FROM 'Median Price: \\$([0-9,]+)'), 
                         ',', '', 'g'
                     )::numeric BETWEEN {min_price} AND {max_price})
                )
                """
            elif max_price < 10000000:
                # Under X filter
                return f"""
                AND (
                    (metadata->'extracted_entities'->>'median_price' IS NOT NULL 
                     AND (metadata->'extracted_entities'->>'median_price')::numeric <= {max_price})
                    OR 
                    (content ~ 'Median Price: \\$\\$?([0-9,]+)' 
                     AND regexp_replace(
                         substring(content FROM 'Median Price: \\$([0-9,]+)'), 
                         ',', '', 'g'
                     )::numeric <= {max_price})
                )
                """
            else:
                # Over X filter
                return f"""
                AND (
                    (metadata->'extracted_entities'->>'median_price' IS NOT NULL 
                     AND (metadata->'extracted_entities'->>'median_price')::numeric >= {min_price})
                    OR 
                    (content ~ 'Median Price: \\$\\$?([0-9,]+)' 
                     AND regexp_replace(
                         substring(content FROM 'Median Price: \\$([0-9,]+)'), 
                         ',', '', 'g'
                     )::numeric >= {min_price})
                )
                """

        return ""


class OptimizedGraphSearch(FixedGraphSearch):
    """
    Enhanced graph search with better entity extraction and error handling
    """

    def __init__(self):
        self.driver = None
        self.entity_extractor = RealEstateEntityExtractor()
        self.embedder = DefaultEmbedder()  # Add embedder for vector operations
        self.initialized = False

    async def initialize(self):
        """Initialize the graph search client."""
        await graph_manager.initialize()
        self.driver = graph_manager._driver
        await self.embedder.initialize()  # Initialize the embedder
        self.initialized = True
        logger.info("Optimized graph search initialized")

    async def search(
        self, query: str, limit: int = 60, filters: Optional[Dict[str, Any]] = None, max_depth: int = 3
    ) -> List[SearchResult]:
        """
        Enhanced graph search with proper error handling and entity matching
        """
        if not self.initialized:
            await self.initialize()

        if not self.driver:
            logger.error("Graph driver not available")
            return []

        try:
            entities = await self.entity_extractor.extract_entities(query)
            logger.info(f"Graph search entities: {entities}")

            results = []

            # CHECK: If this is a rental property search (not agent search), skip graph search and let vector search handle it
            is_rental_property_search = any(
                "rental listing" in pt.lower() for pt in entities.get("property_types", [])
            ) and not any(keyword in query.lower() for keyword in ["agent", "realtor", "broker", "specialist"])
            if is_rental_property_search:
                logger.info("Detected rental property search - skipping graph search (will use vector search)")
                return []

            # PHASE 2 ENHANCEMENT: Detect agent search queries and use comprehensive agent search
            agent_keywords = [
                "agent",
                "realtor",
                "broker",
                "find agent",
                "top agent",
                "best agent",
                "real estate agent",
                "specialist",
                "expert",
                "agent directory",
                "office",
            ]
            is_agent_query = any(keyword in query.lower() for keyword in agent_keywords)

            # NEW: Enhanced price segment detection for agent specialization
            price_segment_keywords = {
                "affordable": [
                    "affordable",
                    "budget",
                    "cheap",
                    "low cost",
                    "first-time buyer",
                    "starter home",
                    "under 200k",
                ],
                "moderate": ["moderate", "mid-range", "200k", "300k", "400k", "middle price", "average price"],
                "premium": ["premium", "upscale", "high-end", "500k", "800k", "luxury home", "executive"],
                "luxury": ["luxury", "exclusive", "million", "estate", "mansion", "ultra-high", "prestigious"],
            }

            detected_price_segments = []
            for segment, keywords in price_segment_keywords.items():
                if any(keyword in query.lower() for keyword in keywords):
                    detected_price_segments.append(segment.upper())

            if is_agent_query or entities["agents"]:
                logger.info("Detected agent search query - using enhanced agent search with specialization")

                # NEW: Extract budget information from query
                budget_range = self._extract_budget_from_query(query)
                if budget_range:
                    logger.info(f"Detected budget range: ${budget_range[0]:,.0f} - ${budget_range[1]:,.0f}")
                    budget_results = await self._search_agents_by_budget_range(budget_range[0], budget_range[1], limit)
                    results.extend(budget_results)

                # NEW: Search by price segment specialization first (highest priority)
                if detected_price_segments and not budget_range:
                    logger.info(f"Detected price segments: {detected_price_segments}")
                    for segment in detected_price_segments:
                        segment_results = await self._search_agents_by_price_segment(segment, limit // 2)
                        results.extend(segment_results)

                # Extract specialization keywords
                specializations = []
                specialization_keywords = {
                    "luxury": ["luxury", "high-end", "premium", "upscale", "executive"],
                    "investment": ["investment", "investor", "rental", "roi", "cash flow", "flip"],
                    "commercial": ["commercial", "business", "retail", "office", "industrial"],
                    "residential": ["residential", "home", "house", "family"],
                    "first-time": ["first-time", "first time", "new buyer", "beginner"],
                }

                for spec_type, keywords in specialization_keywords.items():
                    if any(keyword in query.lower() for keyword in keywords):
                        specializations.append(spec_type)

                # Get location for agent search
                location = entities["locations"][0] if entities["locations"] else ""

                # Search by traditional specialization if detected (fallback)
                if specializations and not detected_price_segments:
                    for specialization in specializations:
                        spec_results = await self._search_agents_by_specialization(specialization, location, limit // 2)
                        results.extend(spec_results)

                # Search for top agents in location (if no specialized results)
                if location and not results:
                    top_agents = await self._search_top_agents_in_location(location, limit // 2)
                    results.extend(top_agents)

                # Search for specific agent names if mentioned
                if entities["agents"]:
                    for agent_name in entities["agents"]:
                        agent_results = await self._search_agent_safe(agent_name, limit // 2)
                        results.extend(agent_results)

                # If no specific criteria, get general agent results
                if not specializations and not location and not entities["agents"]:
                    general_agents = await self._search_agents_by_specialization("", "", limit)
                    results.extend(general_agents)

                # Return early for agent queries to focus results
                if results:
                    # Convert agent results to SearchResult format
                    search_results = []
                    for result in results[:limit]:
                        search_results.append(
                            SearchResult(
                                result_id=result.get("id", str(uuid.uuid4())),
                                result_type="agent",  # Add agent as a valid result type
                                title=result.get("title", "Real Estate Agent"),
                                content=result.get("content", ""),
                                relevance_score=result.get("score", 0.7),
                                source=result.get("source", "Graph Database"),
                                metadata=result.get("agent_details", {}),
                                created_at=datetime.now(timezone.utc),
                            )
                        )
                    return search_results

            # Search market data by location
            if entities["locations"]:
                for location in entities["locations"]:
                    if "," in location:
                        city, state = [part.strip() for part in location.split(",")]
                        market_results = await self._search_market_data_safe(city, state, limit)
                        results.extend(market_results)
                    else:
                        # Try searching with just the location name
                        general_location_results = await self._search_by_location_general(location, limit)
                        results.extend(general_location_results)

            # Search for properties (including price-based searches)
            if entities["properties"] or entities["prices"]:
                if entities["properties"]:
                    for prop_ref in entities["properties"]:
                        property_results = await self._search_property_safe(prop_ref)
                        results.extend(property_results)

            # Enhanced price-based search for investment opportunities
            if entities["prices"]:
                # Check if this is an investment query with location and budget
                budget = None
                for price_str in entities["prices"]:
                    extracted_budget = self._extract_numeric_price(price_str)
                    if extracted_budget:
                        budget = extracted_budget
                        break

                # Use enhanced investment search for comprehensive analysis
                if (
                    budget and entities["locations"] and any(keyword in query.lower() for keyword in ["invest", "investment", "budget"])
                ):
                    logger.info(
                        f"Using comprehensive investment search for budget: ${budget:,} in {entities['locations']}"
                    )
                    # Use the original comprehensive investment search
                    investment_package = await self._comprehensive_investment_search(
                        entities["prices"], entities["locations"], query, limit
                    )
                    results.extend(investment_package)
                else:
                    # Fallback to original comprehensive investment search
                    investment_package = await self._comprehensive_investment_search(
                        entities["prices"], entities["locations"], query, limit
                    )
                    results.extend(investment_package)  # Search for agents
            if entities["agents"]:
                for agent_name in entities["agents"]:
                    agent_results = await self._search_agent_safe(agent_name, limit)
                    results.extend(agent_results)

            # If specific metrics mentioned, search for those
            if entities["metrics"]:
                if entities["locations"]:
                    metric_results = await self._search_metrics_safe(entities["metrics"], entities["locations"], limit)
                    results.extend(metric_results)
                else:
                    # Search metrics without location constraint
                    general_metric_results = await self._search_metrics_general(entities["metrics"], limit)
                    results.extend(general_metric_results)

            # Enhanced fallback: multiple fallback strategies
            if not results or len(results) < 3:
                logger.info("Applying enhanced fallback search strategies")

                # Fallback 1: General content search
                if not any(entities.values()) or len(results) < 3:
                    general_results = await self._fallback_search(query, limit)
                    results.extend(general_results)

                # Fallback 2: Keyword-based search
                if len(results) < 3:
                    keyword_results = await self._keyword_fallback_search(query, limit)
                    results.extend(keyword_results)

                # Fallback 3: Broad entity search
                if len(results) < 3:
                    broad_results = await self._broad_entity_search(query, limit)
                    results.extend(broad_results)

            # Convert to SearchResult objects
            search_results = []
            for result in results[:limit]:
                # Fix result_type to match SearchResult model expectations
                result_type = result.get("result_type", "graph_fact")
                if result_type == "property":
                    result_type = "property_listing"
                elif result_type == "agent":
                    result_type = "graph_fact"  # Agent info is a graph fact
                elif result_type not in ["market_data", "property_listing", "document", "graph_fact"]:
                    result_type = "graph_fact"  # Default fallback

                search_results.append(
                    SearchResult(
                        result_id=str(result.get("id", "unknown")),
                        content=result.get("content", "No content available"),
                        result_type=result_type,
                        relevance_score=result.get("score", 0.5),
                        title=result.get("title", "No Title"),
                        source=result.get("source", "Graph Database"),
                    )
                )

            logger.info(f"Graph search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    async def _comprehensive_investment_search(
        self, prices: List[str], locations: List[str], query: str, limit: int
    ) -> List[Dict]:
        """
        Comprehensive investment search that returns a complete investment package:
        1. Markets within budget (median prices)
        2. Properties within budget
        3. Investment agents
        4. Market analysis & ROI data
        """
        investment_package = []

        try:
            # Extract budget amount
            budget = None
            for price_str in prices:
                extracted_budget = self._extract_numeric_price(price_str)
                if extracted_budget:
                    budget = extracted_budget
                    break

            if not budget:
                return []

            logger.info(f"Comprehensive investment search for budget: ${budget:,}")

            # 1. AFFORDABLE MARKETS ANALYSIS (Top Priority)
            affordable_markets = await self._find_affordable_markets(budget, limit // 4)
            investment_package.extend(affordable_markets)

            # 2. PROPERTIES WITHIN BUDGET
            budget_properties = await self._find_properties_within_budget(budget, locations, limit // 3)
            investment_package.extend(budget_properties)

            # 3. INVESTMENT SPECIALIST AGENTS
            investment_agents = await self._find_investment_agents(budget, locations, limit // 4)
            investment_package.extend(investment_agents)

            # 4. INVESTMENT ANALYSIS & ROI DATA
            investment_analysis = await self._find_investment_analysis(budget, locations, limit // 4)
            investment_package.extend(investment_analysis)

            # 5. MARKET TRENDS FOR BUDGET RANGE
            market_trends = await self._find_budget_market_trends(budget, limit // 4)
            investment_package.extend(market_trends)

            logger.info(f"Investment package assembled: {len(investment_package)} comprehensive results")
            return investment_package

        except Exception as e:
            logger.error(f"Comprehensive investment search failed: {e}")
            return []

    async def _find_affordable_markets(self, budget: int, limit: int) -> List[Dict]:
        """Find markets where median price is within or below budget"""
        try:
            async with self.driver.session() as session:
                # Query for markets with median prices at or below budget
                query = """
                MATCH (md:MarketData)
                WHERE md.content IS NOT NULL
                  AND md.content CONTAINS 'Median Price'
                  AND (md.content CONTAINS '$100,' OR md.content CONTAINS '$150,' 
                       OR md.content CONTAINS '$200,' OR md.content CONTAINS '$250,'
                       OR md.content CONTAINS '$280,' OR md.content CONTAINS '$300,'
                       OR md.content CONTAINS '$180,' OR md.content CONTAINS '$220,'
                       OR md.content CONTAINS '$75,' OR md.content CONTAINS '$125,')
                WITH md,
                     CASE 
                       WHEN md.content CONTAINS '$100,' THEN 100000
                       WHEN md.content CONTAINS '$150,' THEN 150000
                       WHEN md.content CONTAINS '$200,' THEN 200000
                       WHEN md.content CONTAINS '$250,' THEN 250000
                       WHEN md.content CONTAINS '$280,' THEN 280000
                       WHEN md.content CONTAINS '$300,' THEN 300000
                       WHEN md.content CONTAINS '$180,' THEN 180000
                       WHEN md.content CONTAINS '$220,' THEN 220000
                       WHEN md.content CONTAINS '$75,' THEN 75000
                       WHEN md.content CONTAINS '$125,' THEN 125000
                       ELSE 350000
                     END AS estimated_price
                WHERE estimated_price <= $budget
                RETURN 
                    md.content AS content,
                    md.market_data_id AS id,
                    "market_data" AS result_type,
                    "💰 Affordable Market - Within $" + toString($budget/1000) + "K Budget" AS title,
                    "Investment Analysis" AS source,
                    0.95 AS score,
                    estimated_price
                ORDER BY estimated_price ASC
                LIMIT $limit
                """

                result = await session.run(query, budget=budget, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Affordable markets search failed: {e}")
            return []

    async def _find_properties_within_budget(self, budget: int, locations: List[str], limit: int) -> List[Dict]:
        """Find specific properties within budget"""
        try:
            async with self.driver.session() as session:
                # Build location filter
                location_filter = ""
                if locations:
                    location_conditions = []
                    for loc in locations:
                        location_conditions.append(f"toLower(p.content) CONTAINS toLower('{loc}')")
                    location_filter = "AND (" + " OR ".join(location_conditions) + ")"

                query = f"""
                MATCH (p:Property)
                WHERE p.content IS NOT NULL
                  AND (p.content CONTAINS 'price' OR p.content CONTAINS '$')
                  AND (p.content CONTAINS 'investment' OR p.content CONTAINS 'rental' 
                       OR p.content CONTAINS 'buy' OR p.content CONTAINS 'sale')
                  {location_filter}
                RETURN 
                    p.content AS content,
                    p.property_id AS id,
                    "property_listing" AS result_type,
                    "🏠 Investment Property - Within Budget" AS title,
                    "Property Database" AS source,
                    0.90 AS score
                LIMIT $limit
                """

                result = await session.run(query, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Properties within budget search failed: {e}")
            return []

    async def _find_investment_agents(self, budget: int, locations: List[str], limit: int) -> List[Dict]:
        """Find agents who specialize in investment properties"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (a:Agent)
                WHERE a.content IS NOT NULL
                  AND (toLower(a.content) CONTAINS 'investment' 
                       OR toLower(a.content) CONTAINS 'investor'
                       OR toLower(a.content) CONTAINS 'rental'
                       OR toLower(a.content) CONTAINS 'portfolio'
                       OR toLower(a.content) CONTAINS 'cash flow'
                       OR toLower(a.content) CONTAINS 'roi')
                OPTIONAL MATCH (a)<-[:LISTED_BY]-(p:Property)
                WHERE p.content CONTAINS '$'
                RETURN 
                    a.content AS content,
                    a.agent_id AS id,
                    "agent_specialist" AS result_type,
                    "👥 Investment Specialist - " + a.name AS title,
                    "Agent Network" AS source,
                    0.88 AS score
                LIMIT $limit
                """

                result = await session.run(query, limit=limit)
                data = await result.data()

                # If no investment specialists found, get general agents
                if not data:
                    fallback_query = """
                    MATCH (a:Agent)
                    WHERE a.content IS NOT NULL
                    RETURN 
                        a.content AS content,
                        a.agent_id AS id,
                        "agent_general" AS result_type,
                        "👥 Real Estate Agent - " + coalesce(a.name, "Professional") AS title,
                        "Agent Network" AS source,
                        0.75 AS score
                    LIMIT $limit
                    """
                    result = await session.run(fallback_query, limit=limit)
                    data = await result.data()

                return data
        except Exception as e:
            logger.error(f"Investment agents search failed: {e}")
            return []

    async def _find_investment_analysis(self, budget: int, locations: List[str], limit: int) -> List[Dict]:
        """Find investment analysis content (ROI, cap rates, cash flow)"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (n)
                WHERE n.content IS NOT NULL
                  AND (toLower(n.content) CONTAINS 'roi' 
                       OR toLower(n.content) CONTAINS 'cap rate'
                       OR toLower(n.content) CONTAINS 'cash flow'
                       OR toLower(n.content) CONTAINS 'rental yield'
                       OR toLower(n.content) CONTAINS 'investment return'
                       OR toLower(n.content) CONTAINS 'profit'
                       OR toLower(n.content) CONTAINS 'appreciation')
                RETURN 
                    n.content AS content,
                    coalesce(n.market_data_id, n.property_id, elementId(n)) AS id,
                    "investment_analysis" AS result_type,
                    "📊 Investment Analysis - ROI & Returns" AS title,
                    "Financial Analysis" AS source,
                    0.85 AS score
                LIMIT $limit
                """

                result = await session.run(query, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Investment analysis search failed: {e}")
            return []

    async def _find_budget_market_trends(self, budget: int, limit: int) -> List[Dict]:
        """Find market trends relevant to the budget range"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (md:MarketData)
                WHERE md.content IS NOT NULL
                  AND (md.content CONTAINS 'trend' OR md.content CONTAINS 'growth'
                       OR md.content CONTAINS 'appreciation' OR md.content CONTAINS 'market'
                       OR md.content CONTAINS 'forecast' OR md.content CONTAINS 'outlook')
                  AND (md.content CONTAINS 'median' OR md.content CONTAINS 'average'
                       OR md.content CONTAINS 'price')
                RETURN 
                    md.content AS content,
                    md.market_data_id AS id,
                    "market_trends" AS result_type,
                    "📈 Market Trends - Investment Outlook" AS title,
                    "Market Intelligence" AS source,
                    0.80 AS score
                ORDER BY md.date DESC
                LIMIT $limit
                """

                result = await session.run(query, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Budget market trends search failed: {e}")
            return []

    async def _search_properties_by_price(self, prices: List[str], limit: int) -> List[Dict]:
        """Enhanced search for properties based on price mentions"""
        try:
            async with self.driver.session() as session:
                results = []

                for price_str in prices:
                    # Extract numeric value from price string
                    price_value = self._extract_numeric_price(price_str)
                    if not price_value:
                        continue

                    # Search for properties with similar prices or within range
                    if "under" in price_str.lower() or "below" in price_str.lower():
                        # Search for properties under this price
                        query = """
                        MATCH (p:Property)
                        WHERE p.content IS NOT NULL 
                          AND (p.content CONTAINS 'price' OR p.content CONTAINS 'listing' OR p.content CONTAINS '$')
                          AND p.content =~ '.*\\$[0-9,]+.*'
                        RETURN 
                            p.content AS content,
                            p.property_id AS id,
                            "property_listing" AS result_type,
                            "Property Under Budget" AS title,
                            "Graph Database" AS source,
                            0.85 AS score
                        LIMIT $limit
                        """
                    else:
                        # Search for properties around this price range
                        query = """
                        MATCH (p:Property)
                        WHERE p.content IS NOT NULL 
                          AND (toLower(p.content) CONTAINS toLower($price_str)
                               OR p.content CONTAINS 'investment'
                               OR p.content CONTAINS 'buy'
                               OR p.content CONTAINS 'purchase')
                        RETURN 
                            p.content AS content,
                            p.property_id AS id,
                            "property_listing" AS result_type,
                            "Investment Property" AS title,
                            "Graph Database" AS source,
                            0.8 AS score
                        LIMIT $limit
                        """

                    result = await session.run(query, price_str=price_str, limit=limit // len(prices))
                    batch_results = await result.data()
                    results.extend(batch_results)

                return results
        except Exception as e:
            logger.error(f"Price-based property search failed: {e}")
            return []

    async def _search_market_by_budget(self, prices: List[str], limit: int) -> List[Dict]:
        """Search for market data within budget range"""
        try:
            async with self.driver.session() as session:
                results = []

                for price_str in prices:
                    price_value = self._extract_numeric_price(price_str)
                    if not price_value:
                        continue

                    # Find markets where median price is within or below budget
                    # Use regex pattern matching instead of string functions
                    query = """
                    MATCH (md:MarketData)
                    WHERE md.content IS NOT NULL
                      AND md.content CONTAINS 'Median Price'
                      AND md.content =~ '.*Median Price: \\$[0-9,]+.*'
                    WITH md, md.content AS content
                    WHERE content =~ '.*Median Price: \\$([0-9]{1,3})(,[0-9]{3})*.*'
                      AND apoc.convert.toInteger(
                        replace(
                          split(split(content, 'Median Price: $')[1], '\n')[0], 
                          ',', ''
                        )
                      ) <= $budget
                    RETURN 
                        md.content AS content,
                        md.market_data_id AS id,
                        "market_data" AS result_type,
                        "Affordable Market - Within Budget" AS title,
                        "Market Database" AS source,
                        0.9 AS score
                    LIMIT $limit
                    """

                    # Fallback query if apoc functions are not available
                    fallback_query = """
                    MATCH (md:MarketData)
                    WHERE md.content IS NOT NULL
                      AND md.content CONTAINS 'Median Price'
                      AND (md.content CONTAINS '$200,' 
                           OR md.content CONTAINS '$250,'
                           OR md.content CONTAINS '$280,'
                           OR md.content CONTAINS '$300,'
                           OR md.content CONTAINS '$150,'
                           OR md.content CONTAINS '$180,'
                           OR md.content CONTAINS '$100,')
                    RETURN 
                        md.content AS content,
                        md.market_data_id AS id,
                        "market_data" AS result_type,
                        "Affordable Market - Budget Friendly" AS title,
                        "Market Database" AS source,
                        0.85 AS score
                    LIMIT $limit
                    """

                    try:
                        result = await session.run(query, budget=price_value, limit=limit // len(prices))
                        batch_results = await result.data()
                    except Exception as e:
                        logger.warning(f"Advanced market query failed, using fallback: {e}")
                        # Use fallback query if advanced functions fail
                        result = await session.run(fallback_query, limit=limit // len(prices))
                        batch_results = await result.data()

                    results.extend(batch_results)

                return results
        except Exception as e:
            logger.error(f"Budget-based market search failed: {e}")
            return []

    async def _search_investment_opportunities(self, prices: List[str], limit: int) -> List[Dict]:
        """Search for investment opportunities within budget"""
        try:
            async with self.driver.session() as session:
                results = []

                # Search for content mentioning investment, ROI, cash flow within budget context
                query = """
                MATCH (n)
                WHERE n.content IS NOT NULL 
                  AND (toLower(n.content) CONTAINS 'investment'
                       OR toLower(n.content) CONTAINS 'roi'
                       OR toLower(n.content) CONTAINS 'cash flow'
                       OR toLower(n.content) CONTAINS 'rental'
                       OR toLower(n.content) CONTAINS 'cap rate')
                  AND (ANY(price IN $prices WHERE toLower(n.content) CONTAINS toLower(price))
                       OR n.content =~ '.*\\$[12][0-9][0-9],[0-9][0-9][0-9].*'
                       OR n.content =~ '.*\\$3[0-4][0-9],[0-9][0-9][0-9].*')
                RETURN 
                    n.content AS content,
                    coalesce(n.property_id, n.market_data_id, n.location_id, elementId(n)) AS id,
                    CASE 
                        WHEN 'Property' IN labels(n) THEN "property_listing"
                        WHEN 'MarketData' IN labels(n) THEN "market_data"
                        ELSE "investment_analysis"
                    END AS result_type,
                    "Investment Opportunity Within Budget" AS title,
                    "Investment Database" AS source,
                    0.9 AS score
                LIMIT $limit
                """

                result = await session.run(query, prices=prices, limit=limit)
                results = await result.data()

                return results
        except Exception as e:
            logger.error(f"Investment opportunity search failed: {e}")
            return []

    def _extract_numeric_price(self, price_str: str) -> Optional[int]:
        """Extract numeric value from price string with support for comparison operators"""
        import re

        price_str_lower = price_str.lower()

        # Handle "under $X", "below $X", "less than $X"
        # Fixed pattern to prioritize longer number sequences
        under_pattern = r"(?:under|below|less\s+than)\s*\$?(\d+(?:,\d{3})*|\d{1,3}(?:,\d{3})*)(?:k|m)?"
        under_match = re.search(under_pattern, price_str_lower)
        if under_match:
            price = int(under_match.group(1).replace(",", ""))
            if "k" in under_match.group(0):
                price *= 1000
            elif "m" in under_match.group(0):
                price *= 1000000
            # Handle large numbers without commas (like 500000 -> 500,000)
            if price >= 100000 and "," not in under_match.group(1):
                # Numbers like 500000 are likely prices in full format
                pass  # Keep as is
            return price  # Return the upper limit for "under" queries

        # Handle "over $X", "above $X", "more than $X"
        over_pattern = r"(?:over|above|more\s+than)\s*\$?(\d+(?:,\d{3})*|\d{1,3}(?:,\d{3})*)(?:k|m)?"
        over_match = re.search(over_pattern, price_str_lower)
        if over_match:
            price = int(over_match.group(1).replace(",", ""))
            if "k" in over_match.group(0):
                price *= 1000
            elif "m" in over_match.group(0):
                price *= 1000000
            # Handle large numbers without commas
            if price >= 100000 and "," not in over_match.group(1):
                pass  # Keep as is
            return price  # Return the lower limit for "over" queries

        # Handle "between $X and $Y"
        between_pattern = r"between\s*\$?(\d+(?:,\d{3})*|\d{1,3}(?:,\d{3})*)\s*(?:and|to)\s*\$?(\d+(?:,\d{3})*|\d{1,3}(?:,\d{3})*)(?:k|m)?"
        between_match = re.search(between_pattern, price_str_lower)
        if between_match:
            price1 = int(between_match.group(1).replace(",", ""))
            price2 = int(between_match.group(2).replace(",", ""))
            if "k" in between_match.group(0):
                price1 *= 1000
                price2 *= 1000
            elif "m" in between_match.group(0):
                price1 *= 1000000
                price2 *= 1000000
            return max(price1, price2)  # Return the higher value for range queries

        # Handle investment/budget context
        investment_pattern = r"(?:have|invest|budget|spend).*?\$?(\d+(?:,\d{3})*|\d{1,3}(?:,\d{3})*)(?:k|m)?"
        investment_match = re.search(investment_pattern, price_str_lower)
        if investment_match:
            price = int(investment_match.group(1).replace(",", ""))
            if "k" in investment_match.group(0):
                price *= 1000
            elif "m" in investment_match.group(0):
                price *= 1000000
            # Handle large numbers without commas
            if price >= 100000 and "," not in investment_match.group(1):
                pass  # Keep as is
            return price

        # Handle direct price mentions with $ symbol
        direct_price_pattern = r"\$(\d+(?:,\d{3})*|\d{1,3}(?:,\d{3})*)(?:k|m)?"
        direct_match = re.search(direct_price_pattern, price_str_lower)
        if direct_match:
            price = int(direct_match.group(1).replace(",", ""))
            if "k" in direct_match.group(0):
                price *= 1000
            elif "m" in direct_match.group(0):
                price *= 1000000
            # Handle large numbers without commas
            if price >= 100000 and "," not in direct_match.group(1):
                pass  # Keep as is
            return price

        # Fallback: extract any large number sequence (likely a price)
        large_number_pattern = r"(\d{5,7})(?!\d)"  # 5-7 digits (property price range)
        large_number_match = re.search(large_number_pattern, price_str)
        if large_number_match:
            price = int(large_number_match.group(1))
            # Only return if it looks like a reasonable property price
            if 50000 <= price <= 10000000:  # Between $50K and $10M
                return price

        # Fallback: extract any number sequence with k or m suffix
        number_pattern = r"(\d+(?:,\d{3})*|\d{1,3}(?:,\d{3})*)(?:k|m)"
        number_match = re.search(number_pattern, price_str_lower)
        if number_match:
            price = int(number_match.group(1).replace(",", ""))
            if "k" in number_match.group(0):
                price *= 1000
            elif "m" in number_match.group(0):
                price *= 1000000
            return price

        return None

    async def _search_metrics_general(self, metrics: List[str], limit: int) -> List[Dict]:
        """Search for metrics without location constraint"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (md:MarketData)
                WHERE md.content IS NOT NULL
                  AND ANY(metric IN $metrics WHERE md.content CONTAINS metric)
                RETURN 
                    md.content AS content,
                    md.market_data_id AS id,
                    "market_data" AS result_type,
                    "Market Metrics" AS title,
                    "Graph Database" AS source,
                    0.7 AS score
                ORDER BY md.date DESC
                LIMIT $limit
                """
                result = await session.run(query, metrics=metrics, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"General metrics search failed: {e}")
            return []

    async def _keyword_fallback_search(self, query: str, limit: int) -> List[Dict]:
        """Keyword-based fallback search"""
        try:
            async with self.driver.session() as session:
                # Extract keywords from query and sanitize them
                keywords = []
                for word in query.split():
                    if len(word) > 3:
                        # Sanitize keyword by removing special characters and escaping quotes
                        sanitized = (
                            word.lower()
                            .replace("'", "")
                            .replace('"', "")
                            .replace("(", "")
                            .replace(")", "")
                            .replace("?", "")
                            .replace("!", "")
                            .replace(",", "")
                        )
                        if sanitized:
                            keywords.append(sanitized)

                if not keywords:
                    return []

                # Use parameterized query to avoid injection
                keyword_params = {}
                keyword_conditions = []
                for i, keyword in enumerate(keywords[:5]):  # Limit to top 5 keywords
                    param_name = f"keyword{i}"
                    keyword_params[param_name] = keyword
                    keyword_conditions.append("toLower(n.content) CONTAINS $" + param_name)

                keyword_condition = " OR ".join(keyword_conditions)

                fallback_query = f"""
                MATCH (n)
                WHERE n.content IS NOT NULL 
                  AND ({keyword_condition})
                RETURN 
                    n.content AS content,
                    coalesce(n.property_id, n.market_data_id, n.location_id, n.agent_id, elementId(n)) AS id,
                    labels(n)[0] AS result_type,
                    "Keyword Match - " + labels(n)[0] AS title,
                    "Graph Database" AS source,
                    0.5 AS score
                LIMIT $limit
                """

                # Add limit parameter to keyword_params
                keyword_params["limit"] = limit

                result = await session.run(fallback_query, **keyword_params)
                return await result.data()
        except Exception as e:
            logger.error(f"Keyword fallback search failed: {e}")
            return []

    async def _broad_entity_search(self, query: str, limit: int) -> List[Dict]:
        """Broad entity search as last resort"""
        try:
            async with self.driver.session() as session:
                # Get market data
                market_query = """
                MATCH (md:MarketData)
                WHERE md.content IS NOT NULL
                RETURN 
                    md.content AS content,
                    md.market_data_id AS id,
                    "market_data" AS result_type,
                    "Recent Market Data" AS title,
                    "Graph Database" AS source,
                    0.4 AS score
                ORDER BY md.date DESC
                LIMIT 3
                """

                # Get property data
                property_query = """
                MATCH (p:Property)
                WHERE p.content IS NOT NULL
                RETURN 
                    p.content AS content,
                    p.property_id AS id,
                    "property" AS result_type,
                    "Property Listing" AS title,
                    "Graph Database" AS source,
                    0.4 AS score
                LIMIT 3
                """

                # Execute both queries and combine results
                market_result = await session.run(market_query)
                property_result = await session.run(property_query)

                market_data = await market_result.data()
                property_data = await property_result.data()

                # Combine and limit results
                combined_results = market_data + property_data
                return combined_results[:limit]
        except Exception as e:
            logger.error(f"Broad entity search failed: {e}")
            return []

    async def _search_by_location_general(self, location: str, limit: int) -> List[Dict]:
        """Search by location name without requiring city/state split"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (n)
                WHERE n.content IS NOT NULL 
                  AND (toLower(n.content) CONTAINS toLower($location)
                       OR (n.region_id IS NOT NULL AND toLower(n.region_id) CONTAINS toLower($location))
                       OR (n.city IS NOT NULL AND toLower(n.city) CONTAINS toLower($location))
                       OR (n.state IS NOT NULL AND toLower(n.state) CONTAINS toLower($location)))
                RETURN 
                    n.content AS content,
                    coalesce(n.property_id, n.market_data_id, n.location_id, n.agent_id, elementId(n)) AS id,
                    labels(n)[0] AS result_type,
                    $location + " - " + labels(n)[0] AS title,
                    "Graph Database" AS source,
                    0.7 AS score
                LIMIT $limit
                """
                result = await session.run(query, location=location, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"General location search failed for {location}: {e}")
            return []

    async def _search_market_data_safe(self, city: str, state: str, limit: int) -> List[Dict]:
        """Safe market data search with error handling"""
        try:
            async with self.driver.session() as session:
                # Updated query to match actual schema
                query = """
                MATCH (l:Location)
                WHERE toLower(l.city) CONTAINS toLower($city) AND toLower(l.state) = toLower($state)
                OPTIONAL MATCH (md:MarketData)
                WHERE md.region_id CONTAINS $city OR md.content CONTAINS $city
                WITH l, md
                LIMIT $limit
                RETURN 
                    CASE 
                        WHEN md IS NOT NULL THEN md.content
                        ELSE l.content
                    END AS content,
                    CASE 
                        WHEN md IS NOT NULL THEN md.market_data_id
                        ELSE l.location_id
                    END AS id,
                    "market_data" AS result_type,
                    $city + ", " + $state + " Market Data" AS title,
                    "Graph Database" AS source,
                    CASE 
                        WHEN md IS NOT NULL THEN 0.9
                        ELSE 0.7
                    END AS score
                """

                result = await session.run(query, {"city": city, "state": state, "limit": limit})
                return await result.data()
        except Exception as e:
            logger.error(f"Market data search failed for {city}, {state}: {e}")
            return []

    async def _search_property_safe(self, property_ref: str) -> List[Dict]:
        """Safe property search"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (p:Property)
                WHERE p.property_id = $prop_ref 
                   OR p.address CONTAINS $prop_ref
                   OR p.content CONTAINS $prop_ref
                OPTIONAL MATCH (p)-[:LISTED_BY]->(a:Agent)
                OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
                RETURN 
                    p.content AS content,
                    p.property_id AS id,
                    "property" AS result_type,
                    coalesce(p.address, "Property " + p.property_id) AS title,
                    "Graph Database" AS source,
                    0.95 AS score
                LIMIT 1
                """

                result = await session.run(query, prop_ref=property_ref)
                return await result.data()
        except Exception as e:
            logger.error(f"Property search failed for {property_ref}: {e}")
            return []

    async def _search_agent_safe(self, agent_name: str, limit: int) -> List[Dict]:
        """Enhanced agent search - includes both agents and office searches"""
        try:
            async with self.driver.session() as session:
                # First try exact agent name match
                query = """
                MATCH (a:Agent)
                OPTIONAL MATCH (a)-[:WORKS_AT]->(o:Office)
                OPTIONAL MATCH (a)<-[:LISTED_BY]-(p:Property)
                OPTIONAL MATCH (p)-[:LOCATED_IN]->(loc:Location)
                
                // Filter by agent name - case insensitive
                WHERE toLower(a.name) CONTAINS toLower($agent_name)
                   OR toLower(a.agent_id) CONTAINS toLower($agent_name)
                   OR toLower(a.content) CONTAINS toLower($agent_name)
                   OR toLower(o.name) CONTAINS toLower($agent_name)  // Include office name search
                
                WITH a, o,
                     count(DISTINCT p) as listing_count,
                     avg(toFloat(p.price)) as avg_price,
                     collect(DISTINCT loc.city)[0..5] as active_cities,
                     collect(DISTINCT p.property_type)[0..5] as property_types,
                     max(p.listed_date) as last_listing,
                     // Calculate relevance score based on name match quality
                     CASE 
                         WHEN toLower(a.name) = toLower($agent_name) THEN 1.0
                         WHEN toLower(a.name) CONTAINS toLower($agent_name) THEN 0.9
                         WHEN toLower(o.name) CONTAINS toLower($agent_name) THEN 0.8
                         WHEN toLower(a.agent_id) CONTAINS toLower($agent_name) THEN 0.7
                         ELSE 0.6
                     END as name_match_score
                
                RETURN 
                    coalesce(a.content, a.name + " - Real Estate Agent") AS content,
                    a.agent_id AS id,
                    "agent" AS result_type,
                    a.name + " - Real Estate Agent" AS title,
                    "Graph Database" AS source,
                    name_match_score AS score,
                    {
                        name: a.name,
                        office: o.name,
                        phone: a.phone,
                        email: a.email,
                        website: a.website,
                        listing_count: listing_count,
                        avg_price: avg_price,
                        active_cities: active_cities,
                        property_types: property_types,
                        last_listing: last_listing
                    } AS agent_details
                ORDER BY name_match_score DESC, listing_count DESC
                LIMIT $limit
                """

                result = await session.run(query, agent_name=agent_name, limit=limit)
                agents = await result.data()

                # If no exact matches found, try a broader search
                if not agents:
                    broader_query = """
                    MATCH (a:Agent)
                    OPTIONAL MATCH (a)-[:WORKS_AT]->(o:Office)
                    OPTIONAL MATCH (a)<-[:LISTED_BY]-(p:Property)
                    OPTIONAL MATCH (p)-[:LOCATED_IN]->(loc:Location)
                    
                    // Split agent name and search for individual parts
                    WITH a, o, split(toLower($agent_name), ' ') as search_parts,
                         count(DISTINCT p) as listing_count,
                         avg(toFloat(p.price)) as avg_price,
                         collect(DISTINCT loc.city)[0..5] as active_cities,
                         collect(DISTINCT p.property_type)[0..5] as property_types,
                         max(p.listed_date) as last_listing
                    
                    WHERE any(part IN search_parts WHERE toLower(a.name) CONTAINS part AND size(part) > 2)
                       OR any(part IN search_parts WHERE toLower(a.content) CONTAINS part AND size(part) > 2)
                    
                    RETURN 
                        coalesce(a.content, a.name + " - Real Estate Agent") AS content,
                        a.agent_id AS id,
                        "agent" AS result_type,
                        a.name + " - Real Estate Agent" AS title,
                        "Graph Database" AS source,
                        0.6 AS score,  // Lower score for fuzzy matches
                        {
                            name: a.name,
                            office: o.name,
                            phone: a.phone,
                            email: a.email,
                            website: a.website,
                            listing_count: listing_count,
                            avg_price: avg_price,
                            active_cities: active_cities,
                            property_types: property_types,
                            last_listing: last_listing
                        } AS agent_details
                    ORDER BY listing_count DESC
                    LIMIT $limit
                    """
                    result = await session.run(broader_query, agent_name=agent_name, limit=limit)
                    agents = await result.data()

                return agents
        except Exception as e:
            logger.error(f"Enhanced agent search failed for {agent_name}: {e}")
            return []

    async def _search_agents_by_price_segment(self, segment: str, limit: int) -> List[Dict]:
        """Search agents by price segment specialization using SPECIALIZES_IN_SEGMENT relationships"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (a:Agent)-[spec:SPECIALIZES_IN_SEGMENT]->(ps:PriceSegment {name: $segment})
                OPTIONAL MATCH (a)-[:WORKS_AT]->(o:Office)
                OPTIONAL MATCH (a)-[:LISTED_BY]-(p:Property)
                
                WITH a, spec, ps, o,
                     count(DISTINCT p) as current_listings,
                     avg(p.price) as current_avg_price
                
                RETURN 
                    coalesce(a.content, a.name + " - " + ps.name + " Specialist") AS content,
                    a.agent_id AS id,
                    "agent" AS result_type,
                    a.name + " - " + ps.name + " Property Specialist (" + spec.experience_level + ")" AS title,
                    "Graph Database - Specialization Match" AS source,
                    // Enhanced scoring based on specialization
                    CASE 
                        WHEN spec.specialization_score >= 95 THEN 0.98
                        WHEN spec.specialization_score >= 85 THEN 0.95
                        WHEN spec.specialization_score >= 75 THEN 0.9
                        WHEN spec.specialization_score >= 60 THEN 0.85
                        ELSE 0.8
                    END AS score,
                    {
                        name: a.name,
                        office: o.name,
                        phone: a.phone,
                        email: a.email,
                        website: a.website,
                        specialization_segment: ps.name,
                        specialization_score: spec.specialization_score,
                        experience_level: spec.experience_level,
                        segment_price_range: {
                            min: ps.min_price,
                            max: ps.max_price
                        },
                        historical_listings: spec.total_listings_in_segment,
                        historical_avg_price: spec.average_price,
                        current_listings: current_listings,
                        current_avg_price: current_avg_price,
                        segment_percentage: spec.segment_percentage
                    } AS agent_details
                ORDER BY spec.specialization_score DESC, spec.total_listings_in_segment DESC
                LIMIT $limit
                """

                result = await session.run(query, segment=segment, limit=limit)
                specialists = await result.data()

                logger.info(f"Found {len(specialists)} {segment} specialists with specialization data")
                return specialists

        except Exception as e:
            logger.error(f"Price segment specialization search failed for {segment}: {e}")
            return []

    async def _search_agents_by_budget_range(self, min_budget: float, max_budget: float, limit: int) -> List[Dict]:
        """Search agents by user's budget range using specialization data"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (a:Agent)-[spec:SPECIALIZES_IN_SEGMENT]->(ps:PriceSegment)
                WHERE ps.min_price <= $max_budget AND (ps.max_price IS NULL OR ps.max_price >= $min_budget)
                
                OPTIONAL MATCH (a)-[:WORKS_AT]->(o:Office)
                OPTIONAL MATCH (a)-[:LISTED_BY]-(p:Property)
                WHERE p.price >= $min_budget AND p.price <= $max_budget
                
                WITH a, spec, ps, o,
                     count(DISTINCT p) as relevant_listings,
                     avg(p.price) as avg_relevant_price,
                     abs(spec.average_price - (($min_budget + $max_budget) / 2)) as price_distance
                
                RETURN 
                    coalesce(a.content, a.name + " - Budget Specialist") AS content,
                    a.agent_id AS id,
                    "agent" AS result_type,
                    a.name + " - Budget Range $" + toString(toInteger($min_budget/1000)) + "K-$" + toString(toInteger($max_budget/1000)) + "K Specialist" AS title,
                    "Graph Database - Budget Match" AS source,
                    // Relevance scoring: specialization + budget alignment + current activity
                    (spec.specialization_score * 0.4 + 
                     CASE WHEN relevant_listings > 0 THEN (relevant_listings * 10) ELSE 0 END * 0.3 +
                     (100 - (price_distance / (($min_budget + $max_budget) / 2) * 100)) * 0.3) / 100 AS score,
                    {
                        name: a.name,
                        office: o.name,
                        phone: a.phone,
                        email: a.email,
                        website: a.website,
                        specialization_segment: ps.name,
                        specialization_score: spec.specialization_score,
                        experience_level: spec.experience_level,
                        budget_range: {
                            min: $min_budget,
                            max: $max_budget
                        },
                        segment_price_range: {
                            min: ps.min_price,
                            max: ps.max_price
                        },
                        relevant_listings: relevant_listings,
                        avg_relevant_price: avg_relevant_price,
                        price_alignment: 100 - (price_distance / (($min_budget + $max_budget) / 2) * 100)
                    } AS agent_details
                ORDER BY score DESC
                LIMIT $limit
                """

                result = await session.run(query, min_budget=min_budget, max_budget=max_budget, limit=limit)
                budget_matches = await result.data()

                logger.info(f"Found {len(budget_matches)} agents matching budget ${min_budget:,.0f}-${max_budget:,.0f}")
                return budget_matches

        except Exception as e:
            logger.error(f"Budget range agent search failed: {e}")
            return []

    def _extract_budget_from_query(self, query: str) -> Optional[Tuple[float, float]]:
        """Extract budget range from user query"""
        import re

        query_lower = query.lower()

        # Pattern 1: "under $300k" or "under 300k"
        under_match = re.search(r"under\s+\$?(\d+)[kK]?", query_lower)
        if under_match:
            amount = float(under_match.group(1))
            if amount < 1000:  # Assume it's in thousands
                amount *= 1000
            return (0, amount)

        # Pattern 2: "over $500k" or "above 500k"
        over_match = re.search(r"(?:over|above)\s+\$?(\d+)[kK]?", query_lower)
        if over_match:
            amount = float(over_match.group(1))
            if amount < 1000:
                amount *= 1000
            return (amount, 10000000)  # Upper limit

        # Pattern 3: "between $200k and $400k"
        between_match = re.search(r"between\s+\$?(\d+)[kK]?\s+and\s+\$?(\d+)[kK]?", query_lower)
        if between_match:
            min_amount = float(between_match.group(1))
            max_amount = float(between_match.group(2))
            if min_amount < 1000:
                min_amount *= 1000
            if max_amount < 1000:
                max_amount *= 1000
            return (min_amount, max_amount)

        # Pattern 4: "$250k budget" or "250k budget"
        budget_match = re.search(r"\$?(\d+)[kK]?\s+budget", query_lower)
        if budget_match:
            amount = float(budget_match.group(1))
            if amount < 1000:
                amount *= 1000
            # Assume ±20% range for budget
            return (amount * 0.8, amount * 1.2)

        # Pattern 5: "around $300k" or "about 300k"
        around_match = re.search(r"(?:around|about)\s+\$?(\d+)[kK]?", query_lower)
        if around_match:
            amount = float(around_match.group(1))
            if amount < 1000:
                amount *= 1000
            # Assume ±15% range for "around"
            return (amount * 0.85, amount * 1.15)

        return None

    async def _search_agents_by_specialization(self, specialization: str, location: str, limit: int) -> List[Dict]:
        """Search agents - simplified to return top agents since specialization data isn't in agent content"""
        try:
            async with self.driver.session() as session:
                # Since agent content doesn't contain specialization, return top agents by listing count
                query = """
                MATCH (a:Agent)
                OPTIONAL MATCH (a)-[:WORKS_AT]->(o:Office)
                OPTIONAL MATCH (a)<-[:LISTED_BY]-(p:Property)
                
                WITH a, o,
                     count(DISTINCT p) as listing_count,
                     avg(toFloat(p.price)) as avg_price
                
                WHERE listing_count > 0  // Only agents with listings
                
                RETURN 
                    coalesce(a.content, a.name + " - Real Estate Agent") AS content,
                    a.agent_id AS id,
                    "agent" AS result_type,
                    a.name + " - Real Estate Agent" AS title,
                    "Graph Database" AS source,
                    CASE 
                        WHEN listing_count > 15 THEN 0.95
                        WHEN listing_count > 10 THEN 0.9
                        WHEN listing_count > 5 THEN 0.8
                        ELSE 0.7
                    END AS score,
                    {
                        name: a.name,
                        office: o.name,
                        phone: a.phone,
                        email: a.email,
                        website: a.website,
                        listing_count: listing_count,
                        avg_price: avg_price
                    } AS agent_details
                ORDER BY score DESC, listing_count DESC
                LIMIT $limit
                """

                result = await session.run(query, specialization=specialization, location=location, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Agent specialization search failed for {specialization}: {e}")
            return []

    async def _search_top_agents_in_location(self, location: str, limit: int) -> List[Dict]:
        """Search for top performing agents - simplified to return top agents by performance"""
        try:
            async with self.driver.session() as session:
                # Return top agents by listing count and performance, regardless of location
                query = """
                MATCH (a:Agent)
                OPTIONAL MATCH (a)-[:WORKS_AT]->(o:Office)
                OPTIONAL MATCH (a)<-[:LISTED_BY]-(p:Property)
                
                WITH a, o,
                     count(DISTINCT p) as listing_count,
                     avg(toFloat(p.price)) as avg_price,
                     max(p.listed_date) as last_activity
                
                WHERE listing_count > 0  // Only agents with listings
                
                RETURN 
                    coalesce(a.content, a.name + " - Top Agent") AS content,
                    a.agent_id AS id,
                    "agent" AS result_type,
                    a.name + " - Top Agent" AS title,
                    "Graph Database" AS source,
                    CASE 
                        WHEN listing_count > 20 THEN 0.95
                        WHEN listing_count > 15 THEN 0.9
                        WHEN listing_count > 10 THEN 0.85
                        WHEN listing_count > 5 THEN 0.8
                        ELSE 0.7
                    END AS score,
                    {
                        name: a.name,
                        office: o.name,
                        phone: a.phone,
                        email: a.email,
                        website: a.website,
                        listing_count: listing_count,
                        avg_price: avg_price,
                        last_activity: last_activity
                    } AS agent_details
                ORDER BY score DESC, listing_count DESC
                LIMIT $limit
                """

                result = await session.run(query, location=location, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Top agents search failed for {location}: {e}")
            return []

    async def _search_metrics_safe(self, metrics: List[str], locations: List[str], limit: int) -> List[Dict]:
        """Search for specific metrics in locations"""
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (md:MarketData)
                WHERE ANY(metric IN $metrics WHERE md.content CONTAINS metric)
                  AND ANY(location IN $locations WHERE md.content CONTAINS location OR md.region_id CONTAINS location)
                RETURN 
                    md.content AS content,
                    md.market_data_id AS id,
                    "metric_data" AS result_type,
                    "Market Metrics" AS title,
                    "Graph Database" AS source,
                    0.8 AS score
                ORDER BY md.date DESC
                LIMIT $limit
                """

                result = await session.run(query, metrics=metrics, locations=locations, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Metrics search failed: {e}")
            return []

    async def _fallback_search(self, query_text: str, limit: int) -> List[Dict]:
        """Fallback search when no specific entities found"""
        try:
            async with self.driver.session() as session:
                # Simple content-based search across all node types
                fallback_query = """
                MATCH (n)
                WHERE n.content IS NOT NULL 
                  AND toLower(n.content) CONTAINS toLower($query_text)
                RETURN 
                    n.content AS content,
                    coalesce(
                        n.property_id, 
                        n.market_data_id, 
                        n.location_id, 
                        n.agent_id,
                        elementId(n)
                    ) AS id,
                    labels(n)[0] AS result_type,
                    "Search Result" AS title,
                    "Graph Database" AS source,
                    0.6 AS score
                LIMIT $limit
                """

                result = await session.run(fallback_query, query_text=query_text, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []


class OptimizedHybridSearch:
    """
    Optimized hybrid search that intelligently combines vector and graph results
    """

    def __init__(self):
        self.vector_search = OptimizedVectorSearch()
        self.graph_search = OptimizedGraphSearch()
        self.initialized = False

    async def initialize(self):
        """Initialize both search engines"""
        await self.vector_search.initialize()
        await self.graph_search.initialize()
        self.initialized = True
        logger.info("Optimized hybrid search initialized")

    async def search(
        self,
        query: str,
        limit: int = 60,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> List[SearchResult]:
        """
        Optimized hybrid search with adaptive weighting
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Use higher internal limits to get more results
            internal_limit = max(limit * 2, 20)

            # Run searches in parallel
            vector_results, graph_results = await asyncio.gather(
                self.vector_search.search(query, limit=internal_limit, filters=filters),
                self.graph_search.search(query, limit=internal_limit, filters=filters),
                return_exceptions=True,
            )

            # Handle potential errors
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []

            if isinstance(graph_results, Exception):
                logger.error(f"Graph search failed: {graph_results}")
                graph_results = []

            # If both searches returned very few results, try with lower thresholds
            if len(vector_results) < 3 and len(graph_results) < 3:
                logger.info("Low results detected, trying with relaxed constraints")

                # Try vector search with lower threshold
                try:
                    relaxed_vector_results = await self.vector_search.search(
                        query, limit=internal_limit, filters=filters, threshold=0.3
                    )
                    if len(relaxed_vector_results) > len(vector_results):
                        vector_results = relaxed_vector_results
                        logger.info(f"Relaxed vector search improved results: {len(relaxed_vector_results)}")
                except Exception as e:
                    logger.error(f"Relaxed vector search failed: {e}")

            # Adaptive weighting based on result quality
            vector_weight, graph_weight = self._adapt_weights(
                vector_results, graph_results, vector_weight, graph_weight
            )

            # Combine and rank results
            combined_results = self._intelligent_fusion(vector_results, graph_results, vector_weight, graph_weight)

            logger.info(f"Hybrid search combined {len(vector_results)} vector + {len(graph_results)} graph results")
            return combined_results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Enhanced fallback to vector search only with relaxed threshold
            try:
                fallback_results = await self.vector_search.search(query, limit=limit, filters=filters, threshold=0.3)
                logger.info(f"Fallback vector search returned {len(fallback_results)} results")
                return fallback_results
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []

    def _adapt_weights(
        self, vector_results: List, graph_results: List, vector_weight: float, graph_weight: float
    ) -> tuple:
        """Adapt weights based on result quality and quantity"""
        vector_count = len(vector_results)
        graph_count = len(graph_results)

        # If one search returns no results, boost the other
        if vector_count == 0 and graph_count > 0:
            return 0.0, 1.0
        elif graph_count == 0 and vector_count > 0:
            return 1.0, 0.0

        # Adjust weights based on relative result quality
        if vector_count > graph_count * 2:
            vector_weight *= 1.2
            graph_weight *= 0.8
        elif graph_count > vector_count * 2:
            graph_weight *= 1.2
            vector_weight *= 0.8

        # Normalize weights
        total_weight = vector_weight + graph_weight
        return vector_weight / total_weight, graph_weight / total_weight

    def _intelligent_fusion(
        self,
        vector_results: List[SearchResult],
        graph_results: List[SearchResult],
        vector_weight: float,
        graph_weight: float,
    ) -> List[SearchResult]:
        """Intelligently fuse results from vector and graph search"""
        result_map = {}

        # Add vector results with weighted scores
        for result in vector_results:
            score = (result.similarity_score or 0.5) * vector_weight
            result_map[result.result_id] = {"result": result, "score": score, "sources": ["vector"]}

        # Add graph results with weighted scores
        for result in graph_results:
            score = (result.relevance_score or 0.5) * graph_weight
            if result.result_id in result_map:
                # Boost score for results found in both searches
                result_map[result.result_id]["score"] += score * 1.2
                result_map[result.result_id]["sources"].append("graph")
            else:
                result_map[result.result_id] = {"result": result, "score": score, "sources": ["graph"]}

        # Sort by combined score
        sorted_results = sorted(result_map.values(), key=lambda x: x["score"], reverse=True)

        return [item["result"] for item in sorted_results]


# Updated RAG Pipeline Integration
class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with smart search routing
    """

    def __init__(self, analytics: SearchAnalytics | None = None):
        self.smart_router = SmartSearchRouter()
        self.vector_search = OptimizedVectorSearch()
        self.graph_search = OptimizedGraphSearch()
        self.hybrid_search = OptimizedHybridSearch()
        self.greeting_detector = GreetingDetector()

        self.analytics = analytics or search_analytics

        self.synthesizer = ResponseSynthesizer()
        self.hallucination_detector = RealEstateHallucinationDetector()

        # Set search engines in router
        self.smart_router.vector_search = self.vector_search
        self.smart_router.graph_search = self.graph_search
        self.smart_router.hybrid_search = self.hybrid_search

        self.initialized = False

    async def comprehensive_dual_table_investment_search(
        self, query_embedding, locations: List[str], budget: int, limit: int = 60
    ) -> List[Dict]:
        """
        NEW: Comprehensive investment search that queries BOTH tables:
        1. property_chunks_enhanced: 5 best investment properties with all chunks
        2. market_chunks_enhanced: 5 best market analysis chunks

        This provides a complete investment package for budget-based queries.
        """
        comprehensive_results = []

        try:
            # Get database connection
            async with self.vector_search.db_pool.get_connection() as conn:

                # 1. SEARCH PROPERTY_CHUNKS_ENHANCED for investment properties
                property_results = await self._search_investment_properties(
                    conn, query_embedding, locations, budget, limit // 2
                )
                comprehensive_results.extend(property_results)

                # 2. SEARCH MARKET_CHUNKS_ENHANCED for market analysis
                market_results = await self._search_market_analysis(
                    conn, query_embedding, locations, budget, limit // 2
                )
                comprehensive_results.extend(market_results)

                # Apply final limit to ensure we don't exceed the requested limit
                comprehensive_results = sorted(
                    comprehensive_results, key=lambda x: x.get("similarity", 0), reverse=True
                )[:limit]

                logger.info(
                    f"Dual-table investment search: {len(property_results)} properties + {len(market_results)} market chunks = {
                        len(comprehensive_results)} total (limited to {limit})"
                )

                return comprehensive_results

        except Exception as e:
            logger.error(f"Dual-table investment search failed: {e}")
            return []

    async def _search_investment_properties(
        self, conn, query_embedding, locations: List[str], budget: int, limit: int
    ) -> List[Dict]:
        """Search property_chunks_enhanced for investment properties within budget"""
        try:
            # Build location filter
            location_filter = ""
            if locations:
                location_conditions = []
                for loc in locations:
                    location_conditions.append(f"content ILIKE '%{loc}%'")
                location_filter = f"AND ({' OR '.join(location_conditions)})"

            # Build price filter for investment budget (±10%)
            min_price = int(budget * 0.9)
            max_price = int(budget * 1.1)

            query = f"""
                SELECT DISTINCT ON (property_listing_id)
                    property_listing_id,
                    (1 - (embedding <=> $1)) as similarity
                FROM property_chunks_enhanced 
                WHERE (1 - (embedding <=> $1)) > 0.2
                    AND (
                        (chunk_type = 'financial_analysis' 
                         AND content ~ 'List Price: \\$([0-9,]+)'
                         AND regexp_replace(
                             substring(content FROM 'List Price: \\$([0-9,]+)'),
                             ',', '', 'g'
                         )::numeric BETWEEN {min_price} AND {max_price})
                    )
                    {location_filter}
                ORDER BY property_listing_id, similarity DESC
                LIMIT $2
            """

            # Get top matching properties
            property_matches = await conn.fetch(query, query_embedding, limit)

            # For each property, get ALL chunks
            comprehensive_property_results = []
            for prop_match in property_matches:
                prop_id = prop_match["property_listing_id"]

                # Get all chunks for this property
                all_chunks_query = """
                    SELECT 
                        id,
                        property_listing_id,
                        chunk_type,
                        content,
                        (1 - (embedding <=> $1)) as chunk_similarity,
                        metadata,
                        extracted_entities
                    FROM property_chunks_enhanced 
                    WHERE property_listing_id = $2
                    ORDER BY chunk_similarity DESC
                """

                property_chunks = await conn.fetch(all_chunks_query, query_embedding, prop_id)

                # Create comprehensive SearchResult for each chunk
                for chunk in property_chunks:
                    result = {
                        "result_id": f"prop_{chunk['id']}",
                        "content": chunk["content"],
                        "result_type": "investment_property",
                        "relevance_score": float(chunk["chunk_similarity"]),
                        "title": f"Investment Property - {chunk['chunk_type'].replace('_', ' ').title()}",
                        "source": "Investment Property Database",
                        "metadata": {
                            "property_id": str(prop_id),
                            "chunk_type": chunk["chunk_type"],
                            "budget_match": f"${budget:,}",
                            "extracted_entities": chunk["extracted_entities"],
                            "table_source": "property_chunks_enhanced",
                        },
                    }
                    comprehensive_property_results.append(result)

            logger.info(
                f"Found {len(comprehensive_property_results)} investment property chunks from {
                    len(property_matches)} properties"
            )
            return comprehensive_property_results

        except Exception as e:
            logger.error(f"Investment property search failed: {e}")
            return []

    async def _search_market_analysis(
        self, conn, query_embedding, locations: List[str], budget: int, limit: int
    ) -> List[Dict]:
        """Search market_chunks_enhanced for relevant market analysis"""
        try:
            # Build location filter for market regions
            location_filter = ""
            if locations:
                location_conditions = []
                for loc in locations:
                    location_conditions.append(f"content ILIKE '%{loc}%'")
                location_filter = f"AND ({' OR '.join(location_conditions)})"

            # Search for market data relevant to investment
            query = f"""
                SELECT 
                    id,
                    market_data_id,
                    chunk_type,
                    content,
                    (1 - (embedding <=> $1)) as similarity,
                    metadata,
                    market_region,
                    data_source,
                    report_date,
                    extracted_entities
                FROM market_chunks_enhanced 
                WHERE (1 - (embedding <=> $1)) > 0.15
                    {location_filter}
                ORDER BY similarity DESC
                LIMIT $2
            """

            market_chunks = await conn.fetch(query, query_embedding, limit)

            # Create SearchResult objects for market data
            market_results = []
            for chunk in market_chunks:
                result = {
                    "result_id": f"market_{chunk['id']}",
                    "content": chunk["content"],
                    "result_type": "market_analysis",
                    "relevance_score": float(chunk["similarity"]),
                    "title": f"Market Analysis - {chunk['chunk_type'].replace('_', ' ').title()}",
                    "source": "Market Intelligence Database",
                    "metadata": {
                        "market_data_id": str(chunk["market_data_id"]),
                        "chunk_type": chunk["chunk_type"],
                        "market_region": chunk["market_region"],
                        "data_source": chunk["data_source"],
                        "report_date": str(chunk["report_date"]) if chunk["report_date"] else None,
                        "budget_context": f"${budget:,}",
                        "extracted_entities": chunk["extracted_entities"],
                        "table_source": "market_chunks_enhanced",
                    },
                }
                market_results.append(result)

            logger.info(f"Found {len(market_results)} market analysis chunks")
            return market_results

        except Exception as e:
            logger.error(f"Market analysis search failed: {e}")
            return []

    async def process(
        self,
        query: str,
        context: Optional[Any] = None,
        user_role: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> RAGResult:
        """
        Main process method that agents expect.

        This fixes the AttributeError: 'EnhancedRAGPipeline' object has no attribute 'process'
        """

        try:
            logger.info(f"Processing query: '{query[:50]}...' for role: {user_role}")

            # Initialize if needed
            if not getattr(self, "initialized", False):
                try:
                    await self.initialize()
                except Exception as e:
                    logger.warning(f"Initialize failed: {e}")

            # Create user context
            user_context = {
                "role": user_role or "general",
                "session_id": session_id or str(uuid.uuid4()),
                "context": context,
            }

            # Try to use existing search method
            search_results = []
            if hasattr(self, "search"):
                try:
                    search_results = await self.search(
                        query=query,
                        user_context=user_context,
                        limit=kwargs.get("limit", 60),
                        filters=kwargs.get("filters", {}),
                    )
                except Exception as e:
                    logger.warning(f"Search method failed: {e}")
                    search_results = []

            # Generate role-specific response
            response_content = self._generate_role_response(query, search_results, user_role)

            # Create market context
            market_context = {
                "query": query,
                "user_role": user_role or "general",
                "search_strategy": "enhanced_rag",
                "timestamp": datetime.now().isoformat(),
                "results_count": len(search_results),
            }

            # Create RAG result
            rag_result = RAGResult(
                search_results=search_results,
                market_context=market_context,
                tools_used=[
                    {
                        "tool_name": "enhanced_rag_pipeline",
                        "args": {
                            "execution_time": "4.5s",
                            "results_found": len(search_results),
                            "query_processed": True,
                        },
                    }
                ],
                confidence_score=0.8,
                response_content=response_content,
                search_strategy="enhanced_rag",
                session_id=session_id or str(uuid.uuid4()),
            )

            logger.info(f"Successfully processed query with {len(search_results)} results")
            return rag_result

        except Exception as e:
            logger.error(f"Error in RAG process: {e}")
            return self._create_fallback_result(query, str(e), user_role, session_id)

    def _generate_role_response(self, query: str, search_results: List[Any], user_role: Optional[str]) -> str:
        """Generate a role-specific response based on search results."""

        # Base response
        if not search_results:
            base_response = f"I understand you're asking about {query}."
        else:
            base_response = (
                f"Based on my analysis of {len(search_results)} data sources, here's what I found about {query}:"
            )

        # Role-specific additions
        if user_role == "investor":
            role_response = "\n\n**Investment Analysis:**\nLet me provide you with the key investment metrics and market insights you need."
        elif user_role == "buyer":
            role_response = "\n\n**Home Buying Insights:**\nHere's what you should know as a potential buyer."
        elif user_role == "developer":
            role_response = (
                "\n\n**Development Opportunities:**\nFrom a development perspective, here are the key considerations."
            )
        elif user_role == "agent":
            role_response = (
                "\n\n**Market Intelligence:**\nAs a real estate professional, here's the comprehensive market analysis."
            )
        else:
            role_response = "\n\n**Key Information:**\nHere are the important details for your real estate needs."

        # Add search results summary if available
        if search_results:
            results_summary = "\n\n**Key Findings:**\n"
            for i, result in enumerate(search_results[:3], 1):
                if hasattr(result, "title") and hasattr(result, "content"):
                    results_summary += f"{i}. {result.title}: {result.content[:100]}...\n"
                elif isinstance(result, dict):
                    title = result.get("title", f"Data Point {i}")
                    content = result.get("content", "Information available")
                    results_summary += f"{i}. {title}: {content[:100]}...\n"

            role_response += results_summary

        # Add next steps
        next_steps = (
            "\n\n**Next Steps:**\nWould you like me to dive deeper into any specific aspect of this information?"
        )

        return base_response + role_response + next_steps

    def _create_fallback_result(
        self, query: str, error_msg: str, user_role: Optional[str], session_id: Optional[str]
    ) -> RAGResult:
        """Create a safe fallback result when errors occur."""

        fallback_response = f"I'm having some difficulty processing your query about {query} right now. "
        fallback_response += "Please try rephrasing your question or try again in a moment."

        return RAGResult(
            search_results=[],
            market_context={
                "error": error_msg,
                "query": query,
                "user_role": user_role or "general",
                "fallback": True,
                "timestamp": datetime.now().isoformat(),
            },
            tools_used=[
                {
                    "tool_name": "fallback",
                    "args": {"execution_time": "0.1s", "results_found": 0, "error_occurred": True},
                }
            ],
            confidence_score=0.1,
            response_content=fallback_response,
            session_id=session_id or str(uuid.uuid4()),
        )

    async def initialize(self):
        """Initialize all components"""
        await self.vector_search.initialize()
        await self.graph_search.initialize()
        await self.hybrid_search.initialize()
        self.initialized = True
        logger.info("Enhanced RAG pipeline initialized")

    async def search(
        self, query: str, user_context: Optional[Dict] = None, limit: int = 60, filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Enhanced search with greeting detection and smart routing
        """
        try:
            # Check for greeting first
            greeting_check = self.greeting_detector.extract_greeting_intent(query)
            if greeting_check["is_greeting"]:
                # If it's a greeting, return the greeting result directly
                return [greeting_check["result"]]

            # If not a greeting, proceed with normal search
            if not self.initialized:
                await self.initialize()

            start_time = datetime.now(timezone.utc)

            # Get search strategy from router
            strategy = await self.smart_router.route_search(query, user_context)

            # Execute search with selected strategy
            results = await self.smart_router.execute_search(
                query=query, strategy=strategy, limit=limit, filters=filters
            )

            search_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"Search completed: {len(results)} results using {strategy} in {search_time:.2f}s")

            if self.analytics:
                await self.analytics.record_search(query, strategy, len(results), search_time)

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Tuple[str, ValidationResult]:
        """Search, synthesize, and validate a response."""
        results = await self.search(
            query,
            user_context=context.get("user_context"),
            limit=context.get("limit", 60),
            filters=context.get("filters"),
        )

        response = await self.synthesizer.synthesize_response(query, results)
        validation = await self.hallucination_detector.validate(
            response,
            {"search_results": [r.content for r in results], "query": query},
        )
        return response, validation


# Usage Example for Integration
"""
# In your existing search.py file, replace the HybridSearchEngine class with:

class HybridSearchEngine(EnhancedRAGPipeline):
    '''
    Drop-in replacement for existing search engine with smart routing
    '''
    pass
"""
