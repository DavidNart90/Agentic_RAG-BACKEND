"""
Phase 3: Enhanced Smart Search Router with Performance Analytics Feedback

This module implements intelligent search strategy selection based on performance analytics,
building on the Phase 2 analytics integration and your performance optimizations.

Features:
- Analytics-driven search strategy selection
- Real-time performance monitoring
- Dynamic strategy optimization
- Query pattern recognition
- Fallback optimization based on success rates
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..analytics.search import SearchAnalytics
from ..models.search import SearchResult
from .smart_search import QueryIntent, SearchStrategy, SmartSearchRouter

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """Represents a query pattern for analytics matching"""

    pattern_type: str
    keywords: List[str]
    intent: str
    typical_strategy: str
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    sample_count: int = 0


@dataclass
class StrategyPerformance:
    """Track performance metrics for search strategies"""

    strategy: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    result_counts: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time in milliseconds"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def avg_result_count(self) -> float:
        """Calculate average result count"""
        if not self.result_counts:
            return 0.0
        return statistics.mean(self.result_counts)

    def record_execution(self, success: bool, response_time: float, result_count: int):
        """Record a strategy execution"""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        self.response_times.append(response_time)
        self.result_counts.append(result_count)
        self.last_updated = datetime.now(timezone.utc)


class PerformanceOptimizer:
    """Real-time performance optimization based on analytics feedback"""

    def __init__(self, analytics: SearchAnalytics):
        self.analytics = analytics
        self.strategy_performance: Dict[str, StrategyPerformance] = {
            SearchStrategy.VECTOR_ONLY.value: StrategyPerformance(SearchStrategy.VECTOR_ONLY.value),
            SearchStrategy.GRAPH_ONLY.value: StrategyPerformance(SearchStrategy.GRAPH_ONLY.value),
            SearchStrategy.HYBRID.value: StrategyPerformance(SearchStrategy.HYBRID.value),
        }
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.optimization_thresholds = {
            "min_executions_for_optimization": 10,
            "performance_improvement_threshold": 0.2,  # 20% improvement
            "success_rate_threshold": 0.8,  # 80% success rate
            "response_time_threshold": 2000,  # 2 seconds
        }

    async def get_optimal_strategy(
        self, query: str, base_strategy: SearchStrategy, query_analysis: Dict[str, Any]
    ) -> SearchStrategy:
        """
        Determine optimal strategy based on performance analytics

        Args:
            query: User query string
            base_strategy: Base strategy from SmartSearchRouter
            query_analysis: Query analysis results (intent, entities, etc.)

        Returns:
            Optimized search strategy
        """
        # Get query pattern
        pattern_key = self._extract_query_pattern(query, query_analysis)

        # Check if we have enough data for optimization
        if pattern_key in self.query_patterns:
            pattern = self.query_patterns[pattern_key]
            if pattern.sample_count >= self.optimization_thresholds["min_executions_for_optimization"]:
                # Use analytics-driven optimization
                optimized_strategy = await self._optimize_strategy_from_pattern(pattern, base_strategy)
                if optimized_strategy != base_strategy:
                    logger.info(
                        f"Strategy optimized from {base_strategy.value} to {optimized_strategy.value} "
                        f"based on pattern '{pattern_key}' (success rate: {pattern.success_rate:.2%})"
                    )
                return optimized_strategy

        # Check overall strategy performance for fallback decisions
        optimized_strategy = self._optimize_strategy_from_performance(base_strategy)
        if optimized_strategy != base_strategy:
            logger.info(
                f"Strategy optimized from {base_strategy.value} to {optimized_strategy.value} "
                f"based on overall performance"
            )

        return optimized_strategy

    def _extract_query_pattern(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Extract semantic pattern for analytics matching"""
        query_lower = query.lower()
        intent = query_analysis.get("intent", "unknown")

        # Agent search patterns
        if intent == QueryIntent.AGENT_SEARCH.value:
            if any(word in query_lower for word in ["luxury", "high-end", "premium", "upscale"]):
                return "agent_search_luxury"
            elif any(word in query_lower for word in ["affordable", "budget", "cheap", "first-time"]):
                return "agent_search_affordable"
            elif any(word in query_lower for word in ["investment", "investor", "commercial"]):
                return "agent_search_investment"
            else:
                return "agent_search_general"

        # Price search patterns
        elif any(word in query_lower for word in ["price", "cost", "expensive", "cheap", "under", "over"]):
            if any(word in query_lower for word in ["median", "average", "market"]):
                return "price_search_market"
            else:
                return "price_search_property"

        # Investment search patterns
        elif any(word in query_lower for word in ["investment", "roi", "cash flow", "return", "should i buy"]):
            return "investment_search"

        # Location-based searches
        elif query_analysis.get("has_locations", False):
            if any(word in query_lower for word in ["market", "trends", "analysis"]):
                return "location_market_search"
            else:
                return "location_property_search"

        # Property specification searches
        elif any(word in query_lower for word in ["bedroom", "bathroom", "sqft", "garage", "pool"]):
            return "property_specs_search"

        # Market analysis
        elif any(word in query_lower for word in ["market", "trends", "analysis", "forecast"]):
            return "market_analysis_search"

        else:
            return "general_search"

    async def _optimize_strategy_from_pattern(
        self, pattern: QueryPattern, base_strategy: SearchStrategy
    ) -> SearchStrategy:
        """Optimize strategy based on query pattern performance"""

        # If pattern has high success rate with a specific strategy, use it
        if pattern.success_rate >= self.optimization_thresholds["success_rate_threshold"]:
            if pattern.typical_strategy != base_strategy.value:
                try:
                    return SearchStrategy(pattern.typical_strategy)
                except ValueError:
                    logger.warning(f"Invalid strategy in pattern: {pattern.typical_strategy}")

        # Check if base strategy is performing poorly for this pattern
        base_perf = self.strategy_performance.get(base_strategy.value)
        if base_perf and base_perf.success_rate < 0.5:  # Less than 50% success
            # Try alternative strategies
            alternatives = [s for s in SearchStrategy if s != base_strategy]
            for alt_strategy in alternatives:
                alt_perf = self.strategy_performance.get(alt_strategy.value)
                if alt_perf and alt_perf.success_rate > base_perf.success_rate + 0.2:  # 20% better
                    return alt_strategy

        return base_strategy

    def _optimize_strategy_from_performance(self, base_strategy: SearchStrategy) -> SearchStrategy:
        """Optimize strategy based on overall performance metrics"""

        base_perf = self.strategy_performance.get(base_strategy.value)
        if not base_perf or base_perf.total_executions < 5:
            return base_strategy

        # Check if base strategy is performing poorly
        if (
            base_perf.success_rate < 0.6 or base_perf.avg_response_time > self.optimization_thresholds["response_time_threshold"]
            # Less than 60% success or too slow
        ):

            # Find better performing alternative
            best_strategy = base_strategy
            best_score = self._calculate_performance_score(base_perf)

            for strategy_name, perf in self.strategy_performance.items():
                if strategy_name != base_strategy.value and perf.total_executions >= 5:

                    score = self._calculate_performance_score(perf)
                    if score > best_score + 0.1:  # 10% better score
                        try:
                            best_strategy = SearchStrategy(strategy_name)
                            best_score = score
                        except ValueError:
                            continue

            return best_strategy

        return base_strategy

    def _calculate_performance_score(self, perf: StrategyPerformance) -> float:
        """Calculate composite performance score (0.0 to 1.0)"""
        if perf.total_executions == 0:
            return 0.0

        # Weighted scoring: 60% success rate, 30% response time, 10% result count
        success_score = perf.success_rate

        # Response time score (inverted - lower is better)
        max_time = self.optimization_thresholds["response_time_threshold"]
        time_score = max(0.0, 1.0 - (perf.avg_response_time / max_time))

        # Result count score (normalized)
        result_score = min(1.0, perf.avg_result_count / 10.0)  # 10+ results = perfect score

        return 0.6 * success_score + 0.3 * time_score + 0.1 * result_score

    async def record_strategy_performance(
        self,
        strategy: SearchStrategy,
        query: str,
        success: bool,
        response_time: float,
        result_count: int,
        query_analysis: Dict[str, Any],
    ):
        """Record strategy performance for future optimization"""

        # Record overall strategy performance
        strategy_perf = self.strategy_performance[strategy.value]
        strategy_perf.record_execution(success, response_time, result_count)

        # Record query pattern performance
        pattern_key = self._extract_query_pattern(query, query_analysis)
        if pattern_key not in self.query_patterns:
            self.query_patterns[pattern_key] = QueryPattern(
                pattern_type=pattern_key,
                keywords=self._extract_keywords(query),
                intent=query_analysis.get("intent", "unknown"),
                typical_strategy=strategy.value,
            )

        pattern = self.query_patterns[pattern_key]
        pattern.sample_count += 1

        # Update pattern success rate (running average)
        if pattern.sample_count == 1:
            pattern.success_rate = 1.0 if success else 0.0
            pattern.avg_response_time = response_time
        else:
            # Weighted average favoring recent results
            weight = 0.1  # 10% weight for new sample
            pattern.success_rate = pattern.success_rate * (1 - weight) + (1.0 if success else 0.0) * weight
            pattern.avg_response_time = pattern.avg_response_time * (1 - weight) + response_time * weight

        # Update typical strategy if this one is performing better
        if success and pattern.success_rate > 0.7:  # 70% threshold
            pattern.typical_strategy = strategy.value

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key keywords from query for pattern matching"""
        keywords = []
        query_lower = query.lower()

        # Real estate specific keywords
        re_keywords = [
            "property",
            "house",
            "home",
            "listing",
            "agent",
            "realtor",
            "broker",
            "price",
            "cost",
            "expensive",
            "cheap",
            "affordable",
            "luxury",
            "investment",
            "roi",
            "cash flow",
            "rental",
            "market",
            "trends",
            "bedroom",
            "bathroom",
            "sqft",
            "garage",
            "pool",
        ]

        for keyword in re_keywords:
            if keyword in query_lower:
                keywords.append(keyword)

        return keywords[:5]  # Limit to top 5 keywords

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            "strategy_performance": {},
            "query_patterns": {},
            "optimization_summary": {
                "total_patterns": len(self.query_patterns),
                "optimizable_patterns": sum(1 for p in self.query_patterns.values() if p.sample_count >= 10),
                "high_performing_strategies": [],
            },
        }

        # Strategy performance stats
        for strategy_name, perf in self.strategy_performance.items():
            stats["strategy_performance"][strategy_name] = {
                "total_executions": perf.total_executions,
                "success_rate": perf.success_rate,
                "avg_response_time": perf.avg_response_time,
                "avg_result_count": perf.avg_result_count,
                "performance_score": self._calculate_performance_score(perf),
            }

            if perf.total_executions >= 10 and perf.success_rate >= 0.8:
                stats["optimization_summary"]["high_performing_strategies"].append(strategy_name)

        # Query pattern stats
        for pattern_key, pattern in self.query_patterns.items():
            if pattern.sample_count >= 5:  # Only include patterns with enough data
                stats["query_patterns"][pattern_key] = {
                    "sample_count": pattern.sample_count,
                    "success_rate": pattern.success_rate,
                    "avg_response_time": pattern.avg_response_time,
                    "typical_strategy": pattern.typical_strategy,
                    "keywords": pattern.keywords,
                }

        return stats


class EnhancedSmartSearchRouter(SmartSearchRouter):
    """
    Enhanced SmartSearchRouter with analytics feedback and performance optimization.

    This router builds on the base SmartSearchRouter by adding:
    - Performance analytics integration
    - Dynamic strategy optimization
    - Query pattern learning
    - Real-time performance monitoring
    """

    def __init__(
        self, vector_search=None, graph_search=None, hybrid_search=None, analytics: Optional[SearchAnalytics] = None
    ):
        super().__init__(vector_search, graph_search, hybrid_search)

        self.analytics = analytics or SearchAnalytics()
        self.optimizer = PerformanceOptimizer(self.analytics)
        self.performance_cache = {}
        self.last_cache_cleanup = datetime.now(timezone.utc)

    async def route_query(self, query: str, user_context: Optional[Dict] = None) -> SearchStrategy:
        """
        Enhanced query routing with analytics feedback and performance optimization.

        This method:
        1. Gets base routing decision from parent class
        2. Analyzes query for pattern matching
        3. Applies performance-based optimizations
        4. Returns optimized strategy
        """
        start_time = time.time()

        # Get base routing decision from parent
        base_strategy = await super().route_search(query, user_context)

        # Enhanced query analysis
        query_analysis = await self._comprehensive_query_analysis(query, user_context)

        # Apply analytics-driven optimization
        optimized_strategy = await self.optimizer.get_optimal_strategy(query, base_strategy, query_analysis)

        # Cache performance data for this query pattern
        pattern_key = self.optimizer._extract_query_pattern(query, query_analysis)
        if pattern_key not in self.performance_cache:
            self.performance_cache[pattern_key] = {
                "last_strategy": optimized_strategy,
                "usage_count": 0,
                "avg_performance": 0.0,
            }

        self.performance_cache[pattern_key]["usage_count"] += 1
        self.performance_cache[pattern_key]["last_strategy"] = optimized_strategy

        routing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Enhanced routing completed in {routing_time:.2f}ms: "
            f"{base_strategy.value} -> {optimized_strategy.value}"
        )

        return optimized_strategy

    async def execute_search_with_monitoring(
        self, query: str, strategy: SearchStrategy, limit: int = 10, filters: Optional[Dict] = None
    ) -> List[Any]:
        """
        Execute search with comprehensive performance monitoring and optimization.

        This method:
        1. Executes the search with timing
        2. Records performance metrics
        3. Updates optimization models
        4. Handles intelligent fallbacks
        """
        start_time = time.time()
        query_analysis = await self._comprehensive_query_analysis(query)

        try:
            # Execute primary search
            results = await self.execute_search(query, strategy, limit, filters)

            # Calculate performance metrics
            execution_time = (time.time() - start_time) * 1000
            success = len(results) > 0

            # Record performance for optimization
            await self.optimizer.record_strategy_performance(
                strategy, query, success, execution_time, len(results), query_analysis
            )

            # Log to analytics
            await self.analytics.log_search_execution(
                query=query, strategy=strategy, results=results, response_time=execution_time
            )

            # Update performance cache
            await self._update_performance_cache(query, strategy, success, execution_time)

            logger.info(f"Search executed successfully: {len(results)} results in {execution_time:.2f}ms")
            return results

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Search execution failed after {execution_time:.2f}ms: {e}")

            # Record failure
            await self.optimizer.record_strategy_performance(strategy, query, False, execution_time, 0, query_analysis)

            # Attempt intelligent fallback
            fallback_results = await self._intelligent_fallback(query, strategy, limit, filters)
            if fallback_results:
                fallback_time = (time.time() - start_time) * 1000
                logger.info(f"Fallback successful: {len(fallback_results)} results in {fallback_time:.2f}ms")

                # Record fallback performance (with penalty)
                await self.optimizer.record_strategy_performance(
                    strategy, query, True, fallback_time * 1.5, len(fallback_results), query_analysis
                )

            return fallback_results or []

    async def _comprehensive_query_analysis(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Comprehensive query analysis including entities, intent, and context"""

        # Base analysis from parent
        entities = await self.entity_extractor.extract_entities(query)
        intent = await self.intent_classifier.classify_intent(query)
        query_type = await self.query_classifier.classify_query_type(query)

        # Enhanced analysis
        analysis = {
            "entities": entities,
            "intent": intent.value,
            "query_type": query_type,
            "has_locations": bool(entities["locations"]),
            "has_metrics": bool(entities["metrics"]),
            "has_properties": bool(entities["properties"]),
            "has_agents": bool(entities["agents"]),
            "query_length": len(query.split()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add user context if available
        if user_context:
            analysis["user_role"] = user_context.get("user_role", "unknown")
            analysis["session_id"] = user_context.get("session_id", "unknown")
            analysis["previous_queries"] = user_context.get("previous_queries", [])

        return analysis

    async def _intelligent_fallback(
        self, query: str, failed_strategy: SearchStrategy, limit: int, filters: Optional[Dict]
    ) -> List[Any]:
        """Intelligent fallback with strategy selection based on performance"""

        # Get alternative strategies ordered by performance
        alternatives = []
        for strategy in SearchStrategy:
            if strategy != failed_strategy:
                perf = self.optimizer.strategy_performance.get(strategy.value)
                if perf:
                    score = self.optimizer._calculate_performance_score(perf)
                    alternatives.append((strategy, score))

        # Sort by performance score (descending)
        alternatives.sort(key=lambda x: x[1], reverse=True)

        # Try alternatives in order
        for alt_strategy, score in alternatives:
            if score > 0.3:  # Only try if reasonably good performance
                try:
                    logger.info(f"Trying fallback strategy: {alt_strategy.value} (score: {score:.2f})")
                    results = await self.execute_search(query, alt_strategy, limit, filters)
                    if results:
                        return results
                except Exception as e:
                    logger.warning(f"Fallback strategy {alt_strategy.value} also failed: {e}")
                    continue

        # Final fallback to base vector search
        try:
            if hasattr(self, "vector_search") and self.vector_search:
                logger.info("Using final fallback to vector search")
                return await self.vector_search.search(query, limit=limit, filters=filters)
        except Exception as e:
            logger.error(f"Final fallback also failed: {e}")

        return []

    async def _update_performance_cache(
        self, query: str, strategy: SearchStrategy, success: bool, response_time: float
    ):
        """Update performance cache with recent execution data"""

        # Cleanup old cache entries periodically
        if (datetime.now(timezone.utc) - self.last_cache_cleanup).total_seconds() > 3600:  # 1 hour
            await self._cleanup_performance_cache()

        # Update cache entry
        pattern_key = self.optimizer._extract_query_pattern(query, {"intent": "unknown", "query_type": "unknown"})

        if pattern_key in self.performance_cache:
            cache_entry = self.performance_cache[pattern_key]

            # Update running average performance
            if cache_entry["usage_count"] == 1:
                cache_entry["avg_performance"] = 1.0 if success else 0.0
            else:
                weight = 0.1  # 10% weight for new sample
                cache_entry["avg_performance"] = (
                    cache_entry["avg_performance"] * (1 - weight) + (1.0 if success else 0.0) * weight
                )

    async def _cleanup_performance_cache(self):
        """Clean up old performance cache entries"""

        # Remove entries with low usage that haven't been updated recently
        current_time = datetime.now(timezone.utc)
        to_remove = []

        for pattern_key, cache_entry in self.performance_cache.items():
            if cache_entry["usage_count"] < 5:  # Low usage threshold
                to_remove.append(pattern_key)

        for key in to_remove:
            del self.performance_cache[key]

        self.last_cache_cleanup = current_time
        logger.info(f"Cleaned up {len(to_remove)} cache entries")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance and optimization statistics"""

        base_stats = {
            "router_type": "EnhancedSmartSearchRouter",
            "optimization_enabled": True,
            "cache_size": len(self.performance_cache),
            "last_cache_cleanup": self.last_cache_cleanup.isoformat(),
        }

        # Add optimizer stats
        optimizer_stats = self.optimizer.get_performance_stats()

        return {**base_stats, **optimizer_stats}


# Utility functions for Phase 3 testing and validation
async def validate_enhanced_routing_performance():
    """Validate the enhanced routing performance improvements"""

    from ..analytics.search import search_analytics
    from .optimized_pipeline import OptimizedVectorSearch
    from .smart_search import FixedGraphSearch

    # Initialize components
    vector_search = OptimizedVectorSearch()
    graph_search = FixedGraphSearch()

    # Create enhanced router
    enhanced_router = EnhancedSmartSearchRouter(
        vector_search=vector_search, graph_search=graph_search, analytics=search_analytics
    )

    test_queries = [
        "Find luxury agents in Austin Texas specializing in $2M+ properties",
        "What's the median price in Travis County?",
        "Properties under $500K with pool and 3+ bedrooms",
        "ROI analysis for investment properties in Dallas",
        "Market trends for condos in downtown Houston",
        "Agent specializing in affordable first-time buyer homes",
        "Commercial real estate investment opportunities",
        "Best realtor for luxury homes in River Oaks",
    ]

    print("🚀 Phase 3: Enhanced Router Performance Validation")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")

        start_time = time.time()

        # Route query
        strategy = await enhanced_router.route_query(query)
        print(f"   Strategy: {strategy.value}")

        # Execute with monitoring
        results = await enhanced_router.execute_search_with_monitoring(query, strategy, limit=5)

        execution_time = (time.time() - start_time) * 1000
        print(f"   Results: {len(results)} in {execution_time:.2f}ms")

        # Show optimization in action
        if i % 3 == 0:  # Every 3rd query, show stats
            stats = enhanced_router.get_comprehensive_stats()
            print(f"   Cache size: {stats['cache_size']}")
            print(f"   Patterns learned: {stats['optimization_summary']['total_patterns']}")


if __name__ == "__main__":
    asyncio.run(validate_enhanced_routing_performance())
