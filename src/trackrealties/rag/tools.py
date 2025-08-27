"""
Tools for the RAG pipeline.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import get_settings
from ..core.graph import graph_manager
from ..data.embedding.optimized_embedder import (NeonDBEmbeddingManager,
                                                 OptimizedEmbeddingPipeline)

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorSearchTool:
    """
    Lean Vector Search Tool - Data Retrieval Only

    Purpose: Fast, efficient semantic search from NeonDB PostgreSQL
    Responsibility: Query embedding generation and vector similarity search

    Analysis and formatting handled by separate tools:
    - AnalyticsTool: Content analysis, relevance scoring
    - FormatterTool: Role-specific output formatting

    Database: NeonDB PostgreSQL with pgvector extension
    Tables: property_chunks_enhanced, market_chunks_enhanced with HNSW indexes
    """

    def __init__(self):
        self.name = "vector_search"
        self.description = "Fast semantic search using vector embeddings from NeonDB enhanced tables. Returns raw chunks with similarity scores."
        self.embedding_pipeline = None
        self.db_manager = None
        self.initialized = False

    async def initialize(self):
        """Initialize embedding pipeline and database connection."""
        try:
            # Initialize embedding pipeline
            self.embedding_pipeline = OptimizedEmbeddingPipeline(
                batch_size=10, max_concurrent_batches=1, enable_cache_warming=False
            )
            await self.embedding_pipeline.initialize()

            # Initialize database manager
            self.db_manager = NeonDBEmbeddingManager(settings.neon_database_url)
            await self.db_manager.initialize()

            self.initialized = True
            logger.info("VectorSearchTool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize VectorSearchTool: {e}")
            self.initialized = False

    async def execute(
        self,
        query: str,
        search_type: str = "combined",
        limit: int = 10,
        similarity_threshold: float = 0.6,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute vector search and return raw similarity results.

        Args:
            query: Natural language search query
            search_type: "property", "market", "combined"
            limit: Maximum number of results per type
            similarity_threshold: Minimum similarity score (0.0-1.0)
            filters: Optional filters (chunk_type, semantic_level, entity_types)

        Returns:
            Dict with raw search results - no analysis or formatting
        """

        if not self.initialized:
            await self.initialize()

        if not self.embedding_pipeline or not self.db_manager:
            return {"success": False, "error": "Vector search not initialized", "data": []}

        try:
            logger.info(f"Vector search: '{query}' type: {search_type}")

            # Generate query embedding
            query_embeddings, _, metrics = await self.embedding_pipeline.generate_embeddings_optimized([query])
            query_embedding = query_embeddings[0]

            # Execute search based on type
            results = []

            if search_type in ["property", "combined"]:
                property_results = await self._search_property_chunks_enhanced(
                    query_embedding, limit, similarity_threshold, filters
                )
                results.extend(property_results)

            if search_type in ["market", "combined"]:
                market_results = await self._search_market_chunks_enhanced(
                    query_embedding, limit, similarity_threshold, filters
                )
                results.extend(market_results)

            # Sort by combined score if available, otherwise similarity
            if search_type == "combined":
                results.sort(key=lambda x: x.get("combined_score", x.get("similarity", 0)), reverse=True)
                results = results[:limit]

            return {
                "success": True,
                "data": results,
                "search_type": search_type,
                "total_results": len(results),
                "embedding_metrics": {
                    "cache_hit_rate": metrics.cache_hit_rate,
                    "api_calls": metrics.api_calls,
                    "total_time": metrics.total_time,
                },
            }

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"success": False, "error": str(e), "data": []}

    async def _search_property_chunks_enhanced(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search property_chunks_enhanced using vector similarity with semantic scoring."""

        try:
            # Convert embedding to PostgreSQL vector format
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
                    elif key == "semantic_level":
                        filter_conditions.append(f"semantic_level = ${param_count}")
                    elif key == "min_semantic_score":
                        filter_conditions.append(f"semantic_score >= ${param_count}")
                    elif key == "entity_type":
                        filter_conditions.append(f"${param_count} = ANY(entity_types)")
                    else:
                        filter_conditions.append(f"metadata->>'{key}' = ${param_count}")
                    filter_params.append(value)

            filter_clause = " AND " + " AND ".join(filter_conditions) if filter_conditions else ""

            # Advanced search query using enhanced schema
            search_query = f"""
            SELECT 
                'property_' || id::text as result_id,
                content,
                'property' as result_type,
                1 - (embedding <=> $1::vector) AS similarity,
                semantic_score,
                content_density,
                (0.6 * (1 - (embedding <=> $1::vector)) + 
                 0.3 * semantic_score + 
                 0.1 * COALESCE(content_density, 0)) as combined_score,
                chunk_type,
                semantic_level,
                entity_types,
                metadata,
                extracted_entities,
                token_count,
                created_at
            FROM property_chunks_enhanced
            WHERE 
                embedding IS NOT NULL
                AND (1 - (embedding <=> $1::vector)) >= $2
                {filter_clause}
            ORDER BY combined_score DESC
            LIMIT $3
            """

            async with self.db_manager.pool.acquire() as conn:
                records = await conn.fetch(search_query, *filter_params)

                results = []
                for record in records:
                    results.append(
                        {
                            "type": "property",
                            "id": record["result_id"],
                            "content": record["content"],
                            "similarity": float(record["similarity"]),
                            "semantic_score": float(record.get("semantic_score", 0.0)),
                            "content_density": float(record.get("content_density", 0.0)),
                            "combined_score": float(record.get("combined_score", 0.0)),
                            "chunk_type": record.get("chunk_type"),
                            "semantic_level": record.get("semantic_level"),
                            "entity_types": list(record.get("entity_types", [])),
                            "metadata": record.get("metadata", {}),
                            "extracted_entities": record.get("extracted_entities", {}),
                            "token_count": record.get("token_count", 0),
                            "result_type": "property",
                            "table_source": "property_chunks_enhanced",
                        }
                    )

                return results

        except Exception as e:
            logger.error(f"Property chunks enhanced search error: {e}")
            return []

    async def _search_market_chunks_enhanced(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search market_chunks_enhanced using vector similarity."""

        try:
            # Convert embedding to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Build dynamic filter conditions for market chunks
            filter_conditions = []
            filter_params = [embedding_str, similarity_threshold, limit]
            param_count = 3

            if filters:
                for key, value in filters.items():
                    param_count += 1
                    if key == "chunk_type":
                        filter_conditions.append(f"chunk_type = ${param_count}")
                    elif key == "market_region":
                        filter_conditions.append(f"market_region ILIKE ${param_count}")
                        value = f"%{value}%"
                    elif key == "data_source":
                        filter_conditions.append(f"data_source = ${param_count}")
                    elif key == "min_semantic_score":
                        filter_conditions.append(f"semantic_score >= ${param_count}")
                    elif key == "entity_type":
                        filter_conditions.append(f"${param_count} = ANY(entity_types)")
                    else:
                        filter_conditions.append(f"metadata->>'{key}' = ${param_count}")
                    filter_params.append(value)

            filter_clause = " AND " + " AND ".join(filter_conditions) if filter_conditions else ""

            # Market chunks search query
            search_query = f"""
            SELECT 
                'market_' || id::text as result_id,
                content,
                'market' as result_type,
                1 - (embedding <=> $1::vector) AS similarity,
                semantic_score,
                (0.7 * (1 - (embedding <=> $1::vector)) + 
                 0.3 * semantic_score) as combined_score,
                chunk_type,
                entity_types,
                metadata,
                extracted_entities,
                token_count,
                market_region,
                data_source,
                report_date,
                created_at
            FROM market_chunks_enhanced
            WHERE 
                embedding IS NOT NULL
                AND (1 - (embedding <=> $1::vector)) >= $2
                {filter_clause}
            ORDER BY combined_score DESC
            LIMIT $3
            """

            async with self.db_manager.pool.acquire() as conn:
                records = await conn.fetch(search_query, *filter_params)

                results = []
                for record in records:
                    results.append(
                        {
                            "type": "market",
                            "id": record["result_id"],
                            "content": record["content"],
                            "similarity": float(record["similarity"]),
                            "semantic_score": float(record.get("semantic_score", 0.0)),
                            "combined_score": float(record.get("combined_score", 0.0)),
                            "chunk_type": record.get("chunk_type"),
                            "entity_types": list(record.get("entity_types", [])),
                            "metadata": record.get("metadata", {}),
                            "extracted_entities": record.get("extracted_entities", {}),
                            "token_count": record.get("token_count", 0),
                            "market_region": record.get("market_region"),
                            "data_source": record.get("data_source"),
                            "report_date": record.get("report_date"),
                            "result_type": "market",
                            "table_source": "market_chunks_enhanced",
                        }
                    )

                return results

        except Exception as e:
            logger.error(f"Market chunks enhanced search error: {e}")
            return []

    async def search_by_semantic_level(
        self, query: str, semantic_level: str, limit: int = 10, similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search specifically by semantic level (property_overview, location_details, financial_details).
        Leverages your enhanced semantic chunking implementation.
        """

        filters = {"semantic_level": semantic_level}

        result = await self.execute(
            query=query, search_type="property", limit=limit, similarity_threshold=similarity_threshold, filters=filters
        )

        return result.get("data", [])

    async def search_by_chunk_type(
        self, query: str, chunk_type: str, search_type: str = "combined", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search by specific chunk type from your semantic chunking.

        Valid chunk_types:
        - property_core, location_context, features_amenities
        - financial_analysis, agent_info, general
        """

        filters = {"chunk_type": chunk_type}

        result = await self.execute(query=query, search_type=search_type, limit=limit, filters=filters)

        return result.get("data", [])

    async def search_by_entity_type(
        self, query: str, entity_type: str, search_type: str = "combined", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search by specific entity type from extracted entities.

        Common entity_types: property, location, agent, price, financial
        """

        filters = {"entity_type": entity_type}

        result = await self.execute(query=query, search_type=search_type, limit=limit, filters=filters)

        return result.get("data", [])


# Keep the existing function for backward compatibility
async def vector_search_tool(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Legacy function - use VectorSearchTool class instead.
    Fast vector similarity search using embeddings from enhanced tables.
    """

    try:
        # Use the new VectorSearchTool class
        tool = VectorSearchTool()
        await tool.initialize()

        result = await tool.execute(query=query, search_type="combined", limit=limit, similarity_threshold=0.6)

        if result["success"]:
            return result["data"]
        else:
            logger.error(f"Vector search failed: {result.get('error', 'Unknown error')}")
            return []

    except Exception as e:
        logger.error(f"Vector search tool error: {e}")
        return []


class GraphSearchTool:
    """
    Lean Graph Search Tool - Data Retrieval Only
    
    Purpose: Fast, efficient data retrieval from Neo4j knowledge graph
    Responsibility: Query execution and raw data return
    
    Analysis, formatting, and role-specific processing handled by separate tools:
    - AnalyticsTool: Market analysis, ROI calculations, scoring
    - FormatterTool: Role-specific output formatting
    Schema: Property, Agent, Office, Location, MarketData, Metric
    Relationships: LISTED_BY, LISTED_BY_OFFICE, WORKS_AT, LOCATED_IN, HAS_MARKET_DATA, HAS_METRIC, MARKET_CONTEXT
    """

    # US States mapping for location search optimization
    US_STATES_MAPPING = {
        # Full name to abbreviation
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
        "district of columbia": "DC",
        # Abbreviation to full name (reverse mapping)
        "al": "Alabama",
        "ak": "Alaska",
        "az": "Arizona",
        "ar": "Arkansas",
        "ca": "California",
        "co": "Colorado",
        "ct": "Connecticut",
        "de": "Delaware",
        "fl": "Florida",
        "ga": "Georgia",
        "hi": "Hawaii",
        "id": "Idaho",
        "il": "Illinois",
        "in": "Indiana",
        "ia": "Iowa",
        "ks": "Kansas",
        "ky": "Kentucky",
        "la": "Louisiana",
        "me": "Maine",
        "md": "Maryland",
        "ma": "Massachusetts",
        "mi": "Michigan",
        "mn": "Minnesota",
        "ms": "Mississippi",
        "mo": "Missouri",
        "mt": "Montana",
        "ne": "Nebraska",
        "nv": "Nevada",
        "nh": "New Hampshire",
        "nj": "New Jersey",
        "nm": "New Mexico",
        "ny": "New York",
        "nc": "North Carolina",
        "nd": "North Dakota",
        "oh": "Ohio",
        "ok": "Oklahoma",
        "or": "Oregon",
        "pa": "Pennsylvania",
        "ri": "Rhode Island",
        "sc": "South Carolina",
        "sd": "South Dakota",
        "tn": "Tennessee",
        "tx": "Texas",
        "ut": "Utah",
        "vt": "Vermont",
        "va": "Virginia",
        "wa": "Washington",
        "wv": "West Virginia",
        "wi": "Wisconsin",
        "wy": "Wyoming",
        "dc": "District of Columbia",
    }

    def __init__(self):
        self.name = "graph_search"
        self.description = (
            "Fast data retrieval from Neo4j knowledge graph. Returns raw property, agent, market, and location data."
        )
        self.driver = None
        self.initialized = False

    async def initialize(self):
        """Initialize Neo4j connection."""
        try:
            await graph_manager.initialize()
            self.driver = graph_manager._driver
            self.initialized = True
            logger.info("GraphSearchTool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GraphSearchTool: {e}")
            self.initialized = False

    async def execute(self, query: str, search_type: str = "auto", limit: int = 10) -> Dict[str, Any]:
        """
        Execute graph search and return raw data.

        Args:
            query: Natural language query or specific search terms
            search_type: "property", "agent", "location", "market", "auto"
            limit: Maximum number of results

        Returns:
            Dict with raw data from Neo4j - no analysis or formatting
        """

        if not self.initialized:
            await self.initialize()

        if not self.driver:
            return {"success": False, "error": "Neo4j connection not available", "data": []}

        try:
            logger.info(f"Graph search: '{query}' type: {search_type}")

            # Extract entities from query
            entities = self._extract_entities(query)

            # Determine search strategy
            if search_type == "auto":
                search_type = self._auto_detect_search_type(entities, query)

            # Execute appropriate search
            results = await self._execute_search(search_type, entities, query, limit)

            return {
                "success": True,
                "data": results,
                "search_type": search_type,
                "entities_found": entities,
                "total_results": len(results),
            }

        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return {"success": False, "error": str(e), "data": []}

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query."""

        return {
            "locations": self._extract_locations(query),
            "properties": self._extract_property_refs(query),
            "agents": self._extract_agent_names(query),
            "price_ranges": self._extract_price_ranges(query),
            "property_types": self._extract_property_types(query),
        }

    def _auto_detect_search_type(self, entities: Dict, query: str) -> str:
        """Auto-detect search type based on entities found."""

        if entities["properties"]:
            return "property"
        elif entities["agents"]:
            return "agent"
        elif entities["locations"]:
            return "location"
        elif "market" in query.lower() or "metric" in query.lower():
            return "market"
        else:
            return "general"

    async def _execute_search(self, search_type: str, entities: Dict, query: str, limit: int) -> List[Dict[str, Any]]:
        """Execute the appropriate search based on type."""

        if search_type == "property":
            return await self._search_properties(entities, limit)
        elif search_type == "agent":
            return await self._search_agents(entities, limit)
        elif search_type == "location":
            return await self._search_locations(entities, limit)
        elif search_type == "market":
            return await self._search_market_data(entities, limit)
        else:
            return await self._search_general(query, limit)

    # =====================================================
    # FAST CYPHER QUERIES - DATA RETRIEVAL ONLY
    # =====================================================

    async def _search_properties(self, entities: Dict, limit: int) -> List[Dict[str, Any]]:
        """Search properties with full relationship data."""

        results = []

        for prop_ref in entities["properties"]:
            query = """
            MATCH (p:Property)
            WHERE p.property_id CONTAINS $prop_ref 
               OR p.address CONTAINS $prop_ref
               OR toString(p.property_id) CONTAINS $prop_ref
            
            OPTIONAL MATCH (p)-[:LISTED_BY]->(a:Agent)
            OPTIONAL MATCH (p)-[:LISTED_BY_OFFICE]->(o:Office)
            OPTIONAL MATCH (p)-[:LOCATED_IN]->(loc:Location)
            OPTIONAL MATCH (loc)-[:HAS_MARKET_DATA]->(md:MarketData)
            OPTIONAL MATCH (md)-[:HAS_METRIC]->(m:Metric)
            OPTIONAL MATCH (p)-[:MARKET_CONTEXT]->(market_ctx:MarketData)
            
            RETURN 
                p {.*} as property,
                a {.*} as agent,
                o {.*} as office,
                loc {.*} as location,
                collect(DISTINCT md {.*}) as market_data,
                collect(DISTINCT m {.*}) as metrics,
                market_ctx {.*} as market_context
            
            LIMIT $limit
            """

            try:
                async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                    result = await session.run(query, prop_ref=prop_ref, limit=limit)
                    records = await result.data()

                    for record in records:
                        results.append(
                            {
                                "type": "property",
                                "property": record["property"],
                                "agent": record["agent"],
                                "office": record["office"],
                                "location": record["location"],
                                "market_data": record["market_data"],
                                "metrics": record["metrics"],
                                "market_context": record["market_context"],
                            }
                        )

            except Exception as e:
                logger.error(f"Property search error: {e}")

        return results

    async def _search_locations(self, entities: Dict, limit: int) -> List[Dict[str, Any]]:
        """Search locations with market and property data."""

        results = []

        for location in entities["locations"]:
            # Parse city, state
            if "," in location:
                city, state = [part.strip() for part in location.split(",", 1)]
            else:
                city, state = location, ""

            # Get state abbreviation
            state_abbrev = self.US_STATES_MAPPING.get(state.lower(), state) if state else ""

            query = """
            MATCH (loc:Location)
            WHERE toLower(loc.city) CONTAINS toLower($city)
               OR toLower(loc.region_name) CONTAINS toLower($city)
               OR toLower(loc.state) = toLower($state)
               OR toLower(loc.state) = toLower($state_abbrev)
            
            OPTIONAL MATCH (loc)-[:HAS_MARKET_DATA]->(md:MarketData)
            OPTIONAL MATCH (md)-[:HAS_METRIC]->(m:Metric)
            OPTIONAL MATCH (loc)<-[:LOCATED_IN]-(p:Property)
            OPTIONAL MATCH (p)-[:LISTED_BY]->(a:Agent)
            
            WITH loc, md, m, 
                 count(DISTINCT p) as property_count,
                 avg(p.price) as avg_price,
                 collect(DISTINCT a.name)[0..10] as agents
            
            RETURN 
                loc {.*} as location,
                collect(DISTINCT md {.*}) as market_data,
                collect(DISTINCT m {.*}) as metrics,
                property_count,
                avg_price,
                agents
            
            LIMIT $limit
            """

            try:
                async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                    result = await session.run(query, city=city, state=state, state_abbrev=state_abbrev, limit=limit)
                    records = await result.data()

                    for record in records:
                        results.append(
                            {
                                "type": "location",
                                "location": record["location"],
                                "market_data": record["market_data"],
                                "metrics": record["metrics"],
                                "property_count": record["property_count"],
                                "avg_price": record["avg_price"],
                                "agents": record["agents"],
                            }
                        )

            except Exception as e:
                logger.error(f"Location search error: {e}")

        return results

    async def _search_agents(self, entities: Dict, limit: int) -> List[Dict[str, Any]]:
        """Search agents with performance data."""

        results = []

        for agent_name in entities["agents"]:
            query = """
            MATCH (a:Agent)
            WHERE toLower(a.name) CONTAINS toLower($agent_name)
            
            OPTIONAL MATCH (a)-[:WORKS_AT]->(o:Office)
            OPTIONAL MATCH (a)<-[:LISTED_BY]-(p:Property)
            OPTIONAL MATCH (p)-[:LOCATED_IN]->(loc:Location)
            
            WITH a, o,
                 count(DISTINCT p) as listing_count,
                 avg(p.price) as avg_price,
                 collect(DISTINCT loc.city)[0..5] as active_cities
            
            RETURN 
                a {.*} as agent,
                o {.*} as office,
                listing_count,
                avg_price,
                active_cities
            
            LIMIT $limit
            """

            try:
                async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                    result = await session.run(query, agent_name=agent_name, limit=limit)
                    records = await result.data()

                    for record in records:
                        results.append(
                            {
                                "type": "agent",
                                "agent": record["agent"],
                                "office": record["office"],
                                "listing_count": record["listing_count"],
                                "avg_price": record["avg_price"],
                                "active_cities": record["active_cities"],
                            }
                        )

            except Exception as e:
                logger.error(f"Agent search error: {e}")

        return results

    async def _search_market_data(self, entities: Dict, limit: int) -> List[Dict[str, Any]]:
        """Search market data and metrics."""

        query = """
        MATCH (md:MarketData)-[:HAS_METRIC]->(m:Metric)
        OPTIONAL MATCH (loc:Location)-[:HAS_MARKET_DATA]->(md)
        
        RETURN 
            loc {.*} as location,
            md {.*} as market_data,
            collect(m {.*}) as metrics
        
        LIMIT $limit
        """

        results = []

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, limit=limit)
                records = await result.data()

                for record in records:
                    results.append(
                        {
                            "type": "market",
                            "location": record["location"],
                            "market_data": record["market_data"],
                            "metrics": record["metrics"],
                        }
                    )

        except Exception as e:
            logger.error(f"Market search error: {e}")

        return results

    async def _search_general(self, query_text: str, limit: int) -> List[Dict[str, Any]]:
        """General graph traversal."""

        query = """
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE toString(n[prop]) CONTAINS $query_text)
        
        OPTIONAL MATCH (n)-[r]->(connected)
        
        RETURN 
            n {.*} as node,
            labels(n) as node_type,
            collect(DISTINCT {node: connected {.*}, relationship: type(r)})[0..5] as connections
        
        LIMIT $limit
        """

        results = []

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, query_text=query_text, limit=limit)
                records = await result.data()

                for record in records:
                    results.append(
                        {
                            "type": "general",
                            "node": record["node"],
                            "node_type": record["node_type"],
                            "connections": record["connections"],
                        }
                    )

        except Exception as e:
            logger.error(f"General search error: {e}")

        return results

    # =====================================================
    # SIMPLE ENTITY EXTRACTION - NO HEAVY PROCESSING
    # =====================================================

    def _extract_locations(self, query: str) -> List[str]:
        """Extract location references."""
        locations = []

        # City, State patterns
        city_state_pattern = r"\b([A-Z][a-zA-Z\s]+),\s*([A-Z]{2})\b"
        matches = re.findall(city_state_pattern, query)
        for city, state in matches:
            locations.append(f"{city.strip()}, {state}")

        # State patterns
        for state_name, state_abbrev in self.US_STATES_MAPPING.items():
            if len(state_name) > 2 and state_name in query.lower():
                locations.append(state_name.title())
            elif len(state_abbrev) == 2 and re.search(r"\b" + re.escape(state_abbrev) + r"\b", query, re.IGNORECASE):
                locations.append(state_abbrev.upper())

        # Known markets
        known_markets = ["austin", "dallas", "houston", "travis county", "williamson county"]
        query_lower = query.lower()
        for market in known_markets:
            if market in query_lower:
                locations.append(market.title())

        return list(set(locations))

    def _extract_property_refs(self, query: str) -> List[str]:
        """Extract property references."""
        properties = []

        # Property IDs
        patterns = [
            r"\b(?:prop|property)[\s_-]*([a-zA-Z0-9\-_]+)\b",
            r"\b([A-Z0-9]{3,}[-_]?[A-Z0-9]+)\b",
            r"\b(\d{5,})\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            properties.extend(matches)

        # Addresses
        address_pattern = r"\b(\d+\s+[A-Z][a-zA-Z\s]*(?:Street|St|Avenue|Ave|Road|Rd|Way|Drive|Dr|Lane|Ln))\b"
        matches = re.findall(address_pattern, query)
        properties.extend(matches)

        return properties

    def _extract_agent_names(self, query: str) -> List[str]:
        """Extract agent names."""
        agents = []

        patterns = [
            r"agent\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"listed\s+by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"realtor\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            agents.extend(matches)

        return agents
    
    def _extract_price_ranges(self, query: str) -> List[Tuple[int, int]]:
        """Extract price ranges."""
        ranges = []

        patterns = [
            r"\$(\d+(?:,\d{3})*(?:k|K)?)\s*(?:to|-)\s*\$(\d+(?:,\d{3})*(?:k|K)?)",
            r"between\s+\$(\d+(?:,\d{3})*(?:k|K)?)\s*and\s*\$(\d+(?:,\d{3})*(?:k|K)?)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for min_price, max_price in matches:
                min_val = self._parse_price(min_price)
                max_val = self._parse_price(max_price)
                if min_val and max_val:
                    ranges.append((min_val, max_val))

        return ranges

    def _extract_property_types(self, query: str) -> List[str]:
        """Extract property types."""
        types = []

        property_types = ["single family", "condo", "townhouse", "duplex", "apartment", "house", "home"]
        query_lower = query.lower()

        for prop_type in property_types:
            if prop_type in query_lower:
                types.append(prop_type)

        return types

    def _parse_price(self, price_str: str) -> Optional[int]:
        """Parse price string to integer."""
        try:
            price_str = price_str.replace(",", "").lower()
            if price_str.endswith("k"):
                return int(float(price_str[:-1]) * 1000)
            return int(price_str)
        except (ValueError, TypeError):
            return None


async def graph_search_tool(query: str) -> List[Dict[str, Any]]:
    """
    Legacy function - use GraphSearchTool class instead.
    """
    print(f"--- Performing graph search for: {query} ---")
    return [{"entity": "Austin, TX", "relationship": "has_market_trend", "value": "strong_growth"}]
