from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..models.search import SearchResult

logger = logging.getLogger(__name__)


class InMemoryAnalyticsStore:
    """Simple in-memory store for analytics data."""

    def __init__(self) -> None:
        self._search_logs: List[Dict[str, Any]] = []

    async def log_search(self, data: Dict[str, Any]) -> None:
        self._search_logs.append(data)

    async def get_all_searches(self) -> List[Dict[str, Any]]:
        return list(self._search_logs)


class SearchAnalytics:
    """Monitors and analyzes search performance."""

    def __init__(self, analytics_store: Optional[InMemoryAnalyticsStore] = None) -> None:
        self.analytics_store = analytics_store or InMemoryAnalyticsStore()

    async def record_search(
        self,
        query: str,
        strategy: str,
        result_count: int,
        response_time: float,
    ) -> None:
        """Record search execution for analysis (compatibility method)."""
        analytics_data = {
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "strategy": strategy,
            "result_count": result_count,
            "response_time": response_time,
            "has_results": result_count > 0,
            "user_feedback": None,  # Placeholder for future use
        }

        await self.analytics_store.log_search(analytics_data)
        logger.debug("Logged search analytics: %s", analytics_data)

    async def log_search_execution(
        self,
        query: str,
        strategy: Enum,
        results: List[SearchResult],
        response_time: float,
    ) -> None:
        """Log search execution for analysis."""
        analytics_data = {
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "strategy": strategy.value if isinstance(strategy, Enum) else str(strategy),
            "result_count": len(results),
            "response_time": response_time,
            "has_results": len(results) > 0,
            "user_feedback": None,  # Placeholder for future use
        }

        await self.analytics_store.log_search(analytics_data)
        logger.debug("Logged search analytics: %s", analytics_data)

    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate search performance analytics report."""
        return {
            "strategy_performance": await self._analyze_strategy_performance(),
            "query_patterns": await self._analyze_query_patterns(),
            "failure_analysis": await self._analyze_failed_searches(),
            "recommendations": await self._generate_optimization_recommendations(),
        }

    async def _analyze_strategy_performance(self) -> Dict[str, Any]:
        searches = await self.analytics_store.get_all_searches()
        performance: Dict[str, Dict[str, Any]] = {}
        for entry in searches:
            strategy = entry["strategy"]
            perf = performance.setdefault(strategy, {"count": 0, "avg_response_time": 0.0})
            perf["count"] += 1
        for strategy, perf in performance.items():
            times = [s["response_time"] for s in searches if s["strategy"] == strategy]
            perf["avg_response_time"] = sum(times) / len(times) if times else 0.0
        return performance

    async def _analyze_query_patterns(self) -> Dict[str, Any]:
        searches = await self.analytics_store.get_all_searches()
        queries = [s["query"] for s in searches]
        return {
            "total_queries": len(queries),
            "recent_queries": queries[-5:],
        }

    async def _analyze_failed_searches(self) -> Dict[str, Any]:
        searches = await self.analytics_store.get_all_searches()
        failures = [s for s in searches if not s["has_results"]]
        return {
            "failed_count": len(failures),
            "failed_queries": [f["query"] for f in failures][-5:],
        }

    async def _generate_optimization_recommendations(self) -> Dict[str, Any]:
        failures = await self._analyze_failed_searches()
        if failures["failed_count"] > 0:
            return {"message": "Review failed queries for potential data gaps and tune search thresholds."}
        return {"message": "Search system performing optimally."}


# Shared global instance for application use
search_analytics = SearchAnalytics()
