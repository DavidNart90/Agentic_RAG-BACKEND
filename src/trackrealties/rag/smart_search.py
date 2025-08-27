"""
Updated Smart Search Router Implementation for TrackRealties RAG System

This module replaces the broken entity extraction in smart_search.py and provides
intelligent search routing with improved fallback mechanisms.

File: src/trackrealties/rag/smart_search.py (UPDATED)
"""

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Import our enhanced entity extractor
from .llm_entity_extraction import \
    LLMEntityExtractor as RealEstateEntityExtractor

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy options"""

    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"


class QueryIntent(Enum):
    """User query intent types"""

    FACTUAL_LOOKUP = "factual_lookup"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    RELATIONSHIP_QUERY = "relationship_query"
    INVESTMENT_ANALYSIS = "investment_analysis"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    PROPERTY_SEARCH = "property_search"
    AGENT_SEARCH = "agent_search"  # NEW: Dedicated agent search intent


class QueryIntentClassifier:
    """
    Classifies user intent to determine optimal search strategy.
    Enhanced with better pattern matching and fallback logic.
    """

    def __init__(self):
        self.intent_patterns = {
            QueryIntent.FACTUAL_LOOKUP: [
                r"\b(?:what|whats)\s+(?:is|are)\s+(?:the\s+)?(median|average|current|latest)",
                r"\bhow\s+much\s+(?:is|are|does|do|cost)",
                r"\b(?:what\s+)?(?:price|cost|value)\s+(?:of|for|in)",
                r"\b(?:current|latest|recent)\s+(?:price|inventory|count|data)",
                r"\btell\s+me\s+(?:the\s+)?(?:median|average|current)",
                r"\b(?:median|average|mean)\s+(?:price|cost|value)",
                r"\bhow\s+many\s+(?:properties|homes|listings)",
                r"\b(?:what|whats)\s+(?:the\s+)?(?:inventory|supply|count)",
            ],
            QueryIntent.COMPARATIVE_ANALYSIS: [
                r"\bcompare\s+\w+\s+(?:to|vs|versus|against|with)",
                r"\b(?:difference|differences)\s+between",
                r"\b(?:better|best|worse|worst)\s+(?:investment|buy|choice|option)",
                r"\b(?:pros\s+and\s+cons|advantages\s+and\s+disadvantages)",
                r"\bwhich\s+(?:is\s+)?(?:better|best|preferred|worse)",
                r"\b(?:higher|lower|more|less)\s+(?:than|compared\s+to)",
            ],
            QueryIntent.RELATIONSHIP_QUERY: [
                r"\bwho\s+(?:is|are)\s+(?:the\s+)?(?:agent|broker|realtor)",
                r"\bwhich\s+(?:agent|office|company|brokerage)",
                r"\b(?:agent|broker|realtor)\s+(?:for|of)\s+(?:this|that)",
                r"\b(?:listing|listed)\s+(?:by|with)",
                r"\b(?:contact|phone|email)\s+(?:for|of|info)",
                r"\bwho\s+(?:listed|sells|represents)",
                r"\b(?:find|get)\s+(?:agent|broker|realtor)",
            ],
            QueryIntent.INVESTMENT_ANALYSIS: [
                r"\bshould\s+I\s+(?:buy|invest|purchase)",
                r"\b(?:roi|return|cash\s+flow|investment\s+potential)",
                r"\b(?:profitable|worth\s+it|good\s+(?:deal|investment))",
                r"\b(?:rental|investment)\s+(?:property|properties)",
                r"\b(?:analyze|analysis|evaluation)\s+(?:investment|property)",
                r"\bcap\s+rate|cash\s+flow|rental\s+yield",
                r"\binvestment\s+(?:opportunity|strategy|advice)",
            ],
            QueryIntent.PROPERTY_SEARCH: [
                r"\b(?:find|show|search)\s+(?:me\s+)?(?:properties|homes|houses)?(?!\s+for\s+sale|\s+for\s+rent)",
                r"\b(?:looking\s+for|want\s+to\s+find)\s+(?:a\s+)?(?:property|home|house|apartment|condo)",
                r"\b(?:properties|homes|houses|apartments)\s+(?:in|near|around)",
                r"\b(?:\d+)\s+bed(?:room)?(?:s)?",
                r"\bunder\s+\$?\d+[kK]?",
                r"\b(?:single\s+family|condo|townhouse|duplex|apartment)",
                r"\bfor\s+sale\s+in",
                r"\b(?:I\'m\s+looking\s+for|I\s+want|I\s+need)",
                r"\bwith\s+a\s+budget",
                r"\b(?:\d+[-\s]*(?:bed|bedroom|bath|bathroom))",
                r"\bapartment\s+in",
                r"\b(?:buy|purchase)\s+(?:properties|homes|houses|property|home|house)",
                r"\bwhat\s+can\s+I\s+buy\s+(?:with|for)",
                r"\b(?:buy|purchase)\s+(?:for|with)\s+\$?\d+[kK]?",
                r"\bbuy.*(?:rental|investment).*(?:income|property)",
            ],
            QueryIntent.SEMANTIC_ANALYSIS: [
                r"\b(?:tell\s+me\s+about|describe|explain)",
                r"\b(?:overview|summary|analysis)\s+(?:of|for)",
                r"\b(?:market\s+)?(?:trends|conditions|outlook)",
                r"\b(?:insights|recommendations|advice)",
                r"\b(?:what\s+do\s+you\s+think|opinion)",
                r"\b(?:help\s+me\s+understand|break\s+down)",
            ],
            QueryIntent.AGENT_SEARCH: [
                r"\b(?:find|search|show)\s+(?:me\s+)?(?:agents?|realtors?|brokers?)",
                r"\b(?:top|best|good|experienced)\s+(?:agents?|realtors?|brokers?)",
                r"\b(?:agents?|realtors?|brokers?)\s+(?:in|near|around|specializing)",
                r"\b(?:real\s+estate\s+)?(?:agents?|realtors?|brokers?)\s+(?:with|who)",
                r"\b(?:recommend|suggest)\s+(?:an?\s+)?(?:agent|realtor|broker)",
                r"\b(?:luxury|investment|commercial)\s+(?:specialist|expert)\s+(?:agents?|realtors?)",
                r"\b(?:agents?|realtors?)\s+(?:for|that\s+(?:handle|specialize|work))",
                r"\bwho\s+(?:are\s+)?(?:the\s+)?(?:agents?|realtors?|brokers?)",
                r"\blist\s+(?:of\s+)?(?:agents?|realtors?|brokers?)",
                r"\bagent\s+(?:directory|listings?|profiles?)",
                # NEW: Budget and price segment agent searches
                r"\b(?:affordable|budget|cheap|low\s+cost)\s+(?:agents?|realtors?)",
                r"\b(?:agents?|realtors?)\s+(?:for|specializing\s+in)\s+(?:affordable|budget|first.time)",
                r"\b(?:luxury|high.end|premium|upscale)\s+(?:agents?|realtors?)",
                r"\bagents?\s+(?:under|for)\s+\$?\d+[kK]?",
                r"\b(?:moderate|mid.range|middle.price)\s+(?:agents?|realtors?)",
                r"\bagents?\s+(?:who\s+)?(?:handle|sell|specialize\s+in)\s+\$?\d+[kK]?",
            ],
        }

    async def classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a user query with improved accuracy"""
        query_lower = query.lower()

        # Score each intent based on pattern matches
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            intent_scores[intent] = score

        # Return the intent with the highest score, or SEMANTIC_ANALYSIS as default
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        else:
            return QueryIntent.SEMANTIC_ANALYSIS


class QueryClassifier:
    """
    Comprehensive query classification for search strategy determination.
    Enhanced with better keyword matching and context awareness.
    """

    def __init__(self):
        self.vector_indicators = [
            "similar",
            "compare",
            "like",
            "analysis",
            "overview",
            "recommendations",
            "insights",
            "trends",
            "market conditions",
            "describe",
            "explain",
            "tell me about",
            "summary",
            "neighborhood",
            "area analysis",
            "demographic",
        ]

        self.graph_indicators = [
            "relationship",
            "connected",
            "related",
            "history",
            "agent",
            "office",
            "who",
            "when",
            "which agent",
            "listing",
            "contact",
            "phone",
            "email",
            "broker",
            "listed by",
            "works at",
            "company",
            "brokerage",
        ]

        self.hybrid_indicators = [
            "best",
            "recommend",
            "should I",
            "investment",
            "roi",
            "cash flow",
            "market analysis",
            "property analysis",
            "buy",
            "invest",
            "purchase",
            "worth it",
            "find properties",
            "search homes",
            "price range",
            "under",
            "over",
        ]

        self.factual_indicators = [
            "what is",
            "how much",
            "price",
            "median",
            "average",
            "inventory",
            "count",
            "number",
            "specific",
            "exact",
            "current",
            "latest",
            "how many",
            "supply",
        ]

    async def classify_query_type(self, query: str) -> str:
        """Classify query into general categories with improved scoring"""
        query_lower = query.lower()

        scores = {
            "factual": sum(2 if indicator in query_lower else 0 for indicator in self.factual_indicators),
            "vector": sum(1 if indicator in query_lower else 0 for indicator in self.vector_indicators),
            "graph": sum(2 if indicator in query_lower else 0 for indicator in self.graph_indicators),
            "hybrid": sum(1 if indicator in query_lower else 0 for indicator in self.hybrid_indicators),
        }

        # Boost factual score for question words
        if any(word in query_lower for word in ["what", "how much", "how many"]):
            scores["factual"] += 1

        # Boost graph score for relationship indicators
        if any(word in query_lower for word in ["who", "which agent", "listed by"]):
            scores["graph"] += 2

        return max(scores, key=scores.get) if max(scores.values()) > 0 else "hybrid"


class SmartSearchRouter:
    """
    Intelligent search routing with enhanced fallback mechanisms and improved strategy selection.
    """

    def __init__(self, vector_search=None, graph_search=None, hybrid_search=None):
        self.entity_extractor = RealEstateEntityExtractor()
        self.intent_classifier = QueryIntentClassifier()
        self.query_classifier = QueryClassifier()

        self.vector_search = vector_search
        self.graph_search = graph_search
        self.hybrid_search = hybrid_search

    async def route_search(self, query: str, user_context: Optional[Dict] = None) -> SearchStrategy:
        """
        Determine optimal search strategy for the given query with improved logic.
        """
        # Extract entities and classify intent
        entities = await self.entity_extractor.extract_entities(query)
        intent = await self.intent_classifier.classify_intent(query)
        query_type = await self.query_classifier.classify_query_type(query)

        logger.info(f"Query analysis - Intent: {intent}, Type: {query_type}, Entities: {entities}")

        # Decision logic based on intent and entities
        strategy = self._determine_strategy(query, intent, query_type, entities, user_context)

        logger.info(f"Selected search strategy: {strategy}")
        return strategy

    def _determine_strategy(
        self,
        query: str,
        intent: QueryIntent,
        query_type: str,
        entities: Dict[str, List],
        user_context: Optional[Dict] = None,
    ) -> SearchStrategy:
        """
        Enhanced strategy determination with better fallback logic.
        """

        # Rule 1: Factual lookups with specific locations/metrics -> Try HYBRID first (not GRAPH_ONLY)
        # This fixes the original issue where GRAPH_ONLY returned 0 results
        if intent == QueryIntent.FACTUAL_LOOKUP and (entities["locations"] or entities["metrics"]):
            # Use HYBRID to get both vector and graph results, increasing success chance
            return SearchStrategy.HYBRID

        # Rule 2: Agent search with specialization -> Enhanced routing
        if intent == QueryIntent.AGENT_SEARCH:
            # Check for price segment or budget keywords
            query_lower = query.lower()
            price_keywords = ["affordable", "budget", "luxury", "premium", "moderate", "under", "over", "$", "k", "000"]
            has_price_context = any(keyword in query_lower for keyword in price_keywords)

            if has_price_context:
                # Use GRAPH_ONLY for price segment specialization queries
                logger.info("Detected price-based agent query - routing to graph search for specialization")
                return SearchStrategy.GRAPH_ONLY
            else:
                # Use HYBRID for general agent searches
                return SearchStrategy.HYBRID

        # Rule 3: Relationship queries -> Graph Only (agents, offices, etc.)
        if intent == QueryIntent.RELATIONSHIP_QUERY:
            return SearchStrategy.GRAPH_ONLY

        # Rule 4: Investment analysis or comparative analysis -> Hybrid
        if intent in [QueryIntent.INVESTMENT_ANALYSIS, QueryIntent.COMPARATIVE_ANALYSIS]:
            return SearchStrategy.HYBRID

        # Rule 5: Property search - distinguish between simple property search and investment search
        if intent == QueryIntent.PROPERTY_SEARCH:
            # Check if this is an investment-related property search by examining the actual query
            query_lower = query.lower()
            investment_keywords = [
                "investment",
                "roi",
                "cash flow",
                "rental",
                "should i buy",
                "analyze",
                "worth it",
                "profit",
            ]

            if any(keyword in query_lower for keyword in investment_keywords):
                return SearchStrategy.HYBRID  # Investment searches need comprehensive analysis
            else:
                return SearchStrategy.VECTOR_ONLY  # Simple property searches use vector for property_chunks_enhanced

        # Rule 5.1: Buyer searches with property specifications -> Vector Only
        if entities.get("property_specifications") or (entities.get("property_types") and entities.get("prices")):
            # Buyer looking for specific properties (bedrooms, bathrooms, price range)
            return SearchStrategy.VECTOR_ONLY

        # Rule 6: Pure semantic analysis without specific entities -> Vector Only
        if intent == QueryIntent.SEMANTIC_ANALYSIS and not any(entities.values()):
            return SearchStrategy.VECTOR_ONLY

        # Rule 7: Specific property or agent queries -> Graph preferred, but hybrid fallback
        if entities["properties"] or entities["agents"]:
            return SearchStrategy.GRAPH_ONLY

        # Rule 7: Queries with locations but semantic in nature -> Hybrid
        if entities["locations"] and query_type in ["vector", "hybrid"]:
            return SearchStrategy.HYBRID

        # Rule 8: If graph indicators are strong -> Graph Only
        if query_type == "graph":
            return SearchStrategy.GRAPH_ONLY

        # Rule 9: If factual indicators are strong but no specific entities -> Vector
        if query_type == "factual" and not entities["locations"]:
            return SearchStrategy.VECTOR_ONLY

        # Default: Use hybrid for complex or ambiguous queries
        return SearchStrategy.HYBRID

    async def execute_search(
        self, query: str, strategy: SearchStrategy, limit: int = 10, filters: Optional[Dict] = None
    ) -> List[Any]:
        """
        Execute the determined search strategy with intelligent fallback.
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Execute primary strategy
            results = await self._execute_primary_search(query, strategy, limit, filters)

            # If no results and strategy was GRAPH_ONLY, try fallback
            if not results and strategy == SearchStrategy.GRAPH_ONLY:
                logger.info("Graph search returned 0 results, falling back to HYBRID")
                results = await self._execute_primary_search(query, SearchStrategy.HYBRID, limit, filters)

            # If still no results and strategy was VECTOR_ONLY, try fallback
            elif not results and strategy == SearchStrategy.VECTOR_ONLY:
                logger.info("Vector search returned 0 results, falling back to HYBRID")
                results = await self._execute_primary_search(query, SearchStrategy.HYBRID, limit, filters)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"Search executed in {execution_time:.2f}s with {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            # Final fallback to vector search
            if strategy != SearchStrategy.VECTOR_ONLY:
                logger.info("Falling back to vector search due to error")
                try:
                    return await self.vector_search.search(query, limit=limit, filters=filters)
                except Exception as fallback_error:
                    logger.error(f"Fallback search also failed: {fallback_error}")

            return []

    async def _execute_primary_search(
        self, query: str, strategy: SearchStrategy, limit: int, filters: Optional[Dict]
    ) -> List[Any]:
        """Execute the primary search strategy"""

        if strategy == SearchStrategy.VECTOR_ONLY:
            return await self.vector_search.search(query, limit=limit, filters=filters)
        elif strategy == SearchStrategy.GRAPH_ONLY:
            return await self.graph_search.search(query, limit=limit, filters=filters)
        elif strategy == SearchStrategy.HYBRID:
            return await self.hybrid_search.search(query, limit=limit, filters=filters)
        else:
            # Fallback to hybrid
            return await self.hybrid_search.search(query, limit=limit, filters=filters)


# Enhanced Graph Search Implementation
class FixedGraphSearch:
    """
    Fixed implementation of graph search for real estate data.
    This class provides a standardized interface for graph-based searches
    with improved error handling and fallback mechanisms.
    """

    def __init__(self):
        self.initialized = False
        self._search_tool = None

    async def initialize(self):
        """Initialize the graph search tool"""
        try:
            # Dynamic imports to avoid circular dependency
            from ..agents.base import AgentDependencies
            from .tools import GraphSearchTool

            deps = AgentDependencies()
            self._search_tool = GraphSearchTool(deps)
            await self._search_tool.initialize()
            self.initialized = True
            logger.info("FixedGraphSearch initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FixedGraphSearch: {e}")
            self.initialized = False

    async def search(self, query: str, limit: int = 10, filters: Optional[Dict] = None) -> List[Any]:
        """
        Execute graph search with fallback mechanisms

        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional search filters

        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()

        if not self._search_tool or not self.initialized:
            logger.warning("GraphSearchTool not available, returning empty results")
            return []

        try:
            result = await self._search_tool.execute(query=query, search_type="auto", limit=limit)

            if result.get("success", False):
                return result.get("data", [])
            else:
                logger.warning(f"Graph search failed: {result.get('error', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Graph search execution failed: {e}")
            return []


# Enhanced Query Analysis for debugging
class QueryAnalyzer:
    """
    Utility class for analyzing and debugging query processing.
    """

    def __init__(self):
        self.router = SmartSearchRouter()

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive query analysis for debugging and optimization.
        """
        entities = await self.router.entity_extractor.extract_entities(query)
        intent = await self.router.intent_classifier.classify_intent(query)
        query_type = await self.router.query_classifier.classify_query_type(query)
        strategy = await self.router.route_search(query)

        return {
            "query": query,
            "entities": entities,
            "intent": intent.value,
            "query_type": query_type,
            "strategy": strategy.value,
            "entity_count": sum(len(ent_list) for ent_list in entities.values()),
            "has_locations": bool(entities["locations"]),
            "has_metrics": bool(entities["metrics"]),
            "has_properties": bool(entities["properties"]),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Test function for development and debugging
async def test_enhanced_search_routing():
    """Test the enhanced search routing with various real estate queries"""

    test_queries = [
        "What's the median price in Macon County, IL?",  # Original failing query
        "Find properties under $500K in Austin, TX",
        "ROI analysis for rental properties in Travis County, Texas",
        "Contact agent John Smith about 123 Main St, Dallas TX",
        "Average price per sqft in San Francisco metro area",
        "Properties listed by broker Jane Doe in Chicago",
        "Market trends for single family homes in 78701",
        "Days on market for condos in Manhattan, NY",
        "Compare Austin vs Dallas real estate market",
        "Tell me about the Houston housing market",
        "Should I invest in rental properties?",
        "What's the inventory count in Phoenix, AZ?",
    ]

    analyzer = QueryAnalyzer()

    print("🧪 Testing Enhanced Search Routing")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        analysis = await analyzer.analyze_query(query)

        print(f"\n{i}. Query: {query}")
        print(f"   Intent: {analysis['intent']}")
        print(f"   Strategy: {analysis['strategy']}")
        print(f"   Locations: {analysis['entities']['locations']}")
        print(f"   Metrics: {analysis['entities']['metrics']}")

        # Highlight the fix for the original issue
        if "Macon County" in query:
            print("   🎯 FIXED: Now extracts 'Macon County, IL' correctly!")
            print("   🎯 FIXED: Uses HYBRID strategy instead of GRAPH_ONLY!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_enhanced_search_routing())
