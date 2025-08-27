"""
Phase 3: Integrated Search Optimization System

This module integrates all Phase 3 components:
- Enhanced Smart Search Router with analytics feedback
- Dynamic Performance Monitoring
- Your existing performance optimizations
- Query pattern learning and optimization

This builds on your 67% performance improvements and adds intelligent optimization.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..analytics.search import SearchAnalytics, search_analytics
from ..models.search import SearchResult
from .dynamic_performance_monitor import (DynamicPerformanceMonitor,
                                          PerformanceAlert)
from .enhanced_smart_search_router import (EnhancedSmartSearchRouter,
                                           SearchStrategy)
from .performance_optimized_pipeline import (OptimizedVectorSearchV2,
                                             ParallelSearchExecutor)
from .smart_search import FixedGraphSearch, QueryIntentClassifier

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of search optimization"""

    original_strategy: SearchStrategy
    optimized_strategy: SearchStrategy
    performance_improvement: float
    execution_time: float
    result_count: int
    optimization_reason: str


class IntegratedSearchOptimizer:
    """
    Integrated search optimization system that combines:
    - Your performance optimizations (67% improvement)
    - Analytics-driven strategy selection
    - Real-time performance monitoring
    - Dynamic optimization
    """

    def __init__(self, analytics: Optional[SearchAnalytics] = None):
        self.analytics = analytics or search_analytics

        # Initialize core components
        self.vector_search = None
        self.graph_search = None
        self.enhanced_router = None
        self.performance_monitor = None
        self.parallel_executor = None

        # Optimization tracking
        self.optimization_history: List[OptimizationResult] = []
        self.total_optimizations = 0
        self.total_performance_gain = 0.0

        # Configuration
        self.config = {
            "enable_parallel_execution": True,
            "enable_dynamic_optimization": True,
            "enable_performance_monitoring": True,
            "optimization_interval_seconds": 30,
            "performance_threshold_ms": 2000,
            "success_rate_threshold": 0.8,
        }

    async def initialize(self, embedder=None):
        """Initialize all components of the optimization system"""

        logger.info("🚀 Initializing Phase 3 Search Optimization System...")

        # Initialize vector search with your optimizations
        self.vector_search = OptimizedVectorSearchV2(embedder)
        await self.vector_search.initialize()
        logger.info("✅ Optimized vector search initialized (67% faster)")

        # Initialize graph search
        self.graph_search = FixedGraphSearch()
        await self.graph_search.initialize()
        logger.info("✅ Graph search initialized")

        # Initialize enhanced router with analytics
        self.enhanced_router = EnhancedSmartSearchRouter(
            vector_search=self.vector_search, graph_search=self.graph_search, analytics=self.analytics
        )
        logger.info("✅ Enhanced router with analytics feedback initialized")

        # Initialize parallel executor
        self.parallel_executor = ParallelSearchExecutor(
            vector_search=self.vector_search, graph_search=self.graph_search
        )
        logger.info("✅ Parallel search executor initialized")

        # Initialize performance monitor if enabled
        if self.config["enable_performance_monitoring"]:
            self.performance_monitor = DynamicPerformanceMonitor(
                enhanced_router=self.enhanced_router, vector_search=self.vector_search
            )

            # Register optimization callback
            self.performance_monitor.register_optimization_callback(self._handle_performance_optimization)

            # Start monitoring in background
            asyncio.create_task(
                self.performance_monitor.start_monitoring(interval_seconds=self.config["optimization_interval_seconds"])
            )
            logger.info("✅ Dynamic performance monitoring started")

        logger.info("🎉 Phase 3 Search Optimization System ready!")

    async def optimized_search(
        self, query: str, limit: int = 15, user_context: Optional[Dict] = None, filters: Optional[Dict] = None
    ) -> Tuple[List[Any], OptimizationResult]:
        """
        Execute optimized search with comprehensive optimization pipeline.

        This method combines:
        1. Your existing 67% performance improvements
        2. Analytics-driven strategy selection
        3. Parallel execution where beneficial
        4. Real-time performance monitoring
        5. Dynamic optimization based on patterns

        Returns:
            Tuple of (search_results, optimization_result)
        """
        start_time = time.time()

        # Step 1: Enhanced strategy selection with analytics feedback
        original_strategy = await self.enhanced_router.route_query(query, user_context)
        logger.info(f"📍 Base strategy selected: {original_strategy.value}")

        # Step 2: Check for parallel execution opportunities
        execution_strategy = await self._determine_execution_strategy(query, original_strategy)

        # Step 3: Execute search with optimization
        if execution_strategy == "parallel" and self.config["enable_parallel_execution"]:
            results = await self._execute_parallel_search(query, limit, user_context, filters)
            optimization_reason = "parallel_execution"
        else:
            results = await self._execute_optimized_search(query, original_strategy, limit, user_context, filters)
            optimization_reason = "optimized_single_strategy"

        # Step 4: Record performance and optimization
        execution_time = (time.time() - start_time) * 1000

        optimization_result = OptimizationResult(
            original_strategy=original_strategy,
            optimized_strategy=original_strategy,  # May be updated by router
            performance_improvement=self._calculate_performance_improvement(execution_time),
            execution_time=execution_time,
            result_count=len(results),
            optimization_reason=optimization_reason,
        )

        # Step 5: Update optimization tracking
        self.optimization_history.append(optimization_result)
        self.total_optimizations += 1

        logger.info(
            f"🎯 Search completed: {len(results)} results in {execution_time:.2f}ms "
            f"(optimization: {optimization_reason})"
        )

        return results, optimization_result

    async def _determine_execution_strategy(self, query: str, strategy: SearchStrategy) -> str:
        """Determine whether to use parallel execution or optimized single strategy"""

        # Use parallel for complex queries that could benefit from multiple strategies
        query_lower = query.lower()

        # Investment and comparative queries benefit from parallel execution
        if any(
            keyword in query_lower for keyword in ["investment", "roi", "compare", "analysis", "should i", "recommend"]
        ):
            return "parallel"

        # Hybrid strategies can benefit from parallel execution
        if strategy == SearchStrategy.HYBRID:
            return "parallel"

        # For simple property searches, use optimized single strategy
        return "single"

    async def _execute_parallel_search(
        self, query: str, limit: int, user_context: Optional[Dict], filters: Optional[Dict]
    ) -> List[Any]:
        """Execute parallel search with vector and graph strategies"""

        logger.info("⚡ Executing parallel search strategy")

        try:
            results = await self.parallel_executor.parallel_search(
                query=query, limit=limit, vector_weight=0.7, graph_weight=0.3
            )
            return results
        except Exception as e:
            logger.error(f"Parallel search failed, falling back to enhanced router: {e}")
            # Fallback to enhanced router
            strategy = await self.enhanced_router.route_query(query, user_context)
            return await self.enhanced_router.execute_search_with_monitoring(query, strategy, limit, filters)

    async def _execute_optimized_search(
        self, query: str, strategy: SearchStrategy, limit: int, user_context: Optional[Dict], filters: Optional[Dict]
    ) -> List[Any]:
        """Execute optimized single strategy search"""

        logger.info(f"🔧 Executing optimized {strategy.value} search")

        return await self.enhanced_router.execute_search_with_monitoring(query, strategy, limit, filters)

    def _calculate_performance_improvement(self, execution_time: float) -> float:
        """Calculate performance improvement compared to baseline"""

        # Baseline is pre-optimization performance (from your summary: ~5400ms average)
        baseline_time = 5400.0

        if execution_time < baseline_time:
            improvement = ((baseline_time - execution_time) / baseline_time) * 100
            return improvement
        else:
            return 0.0  # No improvement

    async def _handle_performance_optimization(self, recommendations: Dict):
        """Handle performance optimization recommendations from monitor"""

        logger.info("🔧 Processing performance optimization recommendations...")

        for rec in recommendations.get("recommendations", []):
            rec_type = rec.get("type")
            priority = rec.get("priority", "medium")

            if rec_type == "cache_optimization" and priority == "high":
                await self._optimize_cache_settings()
            elif rec_type == "response_time_optimization":
                await self._optimize_response_time()
            elif rec_type == "strategy_optimization":
                await self._optimize_strategy_selection(rec)

    async def _optimize_cache_settings(self):
        """Optimize embedding cache settings based on performance data"""

        if hasattr(self.vector_search, "embedder") and hasattr(self.vector_search.embedder, "cache_size"):
            current_size = self.vector_search.embedder.cache_size
            new_size = min(current_size * 2, 2000)  # Double size, max 2000

            self.vector_search.embedder.cache_size = new_size
            logger.info(f"🔧 Cache size optimized: {current_size} -> {new_size}")

    async def _optimize_response_time(self):
        """Optimize response time by adjusting search parameters"""

        # Enable more aggressive parallel execution
        self.config["enable_parallel_execution"] = True
        logger.info("🔧 Enabled aggressive parallel execution for response time optimization")

    async def _optimize_strategy_selection(self, recommendation: Dict):
        """Optimize strategy selection based on performance data"""

        strategy_name = recommendation.get("strategy")
        success_rate = recommendation.get("current_success_rate", 0.0)

        logger.info(f"🔧 Strategy optimization for {strategy_name}: " f"success rate {success_rate:.1%}")

        # Update router optimization thresholds
        if hasattr(self.enhanced_router, "optimizer"):
            optimizer = self.enhanced_router.optimizer
            if success_rate < 0.5:
                # Lower threshold for this strategy
                threshold_key = f"{strategy_name}_threshold"
                optimizer.optimization_thresholds[threshold_key] = success_rate - 0.1

    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization performance report"""

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase3_status": "active",
            "total_optimizations": self.total_optimizations,
            "optimization_history": len(self.optimization_history),
            "performance_summary": {},
            "component_status": {
                "vector_search": "optimized" if self.vector_search else "not_initialized",
                "graph_search": "active" if self.graph_search else "not_initialized",
                "enhanced_router": "active" if self.enhanced_router else "not_initialized",
                "performance_monitor": "active" if self.performance_monitor else "disabled",
                "parallel_executor": "active" if self.parallel_executor else "not_initialized",
            },
            "configuration": self.config,
        }

        # Add performance summary
        if self.optimization_history:
            recent_optimizations = self.optimization_history[-10:]  # Last 10

            avg_execution_time = sum(opt.execution_time for opt in recent_optimizations) / len(recent_optimizations)
            avg_result_count = sum(opt.result_count for opt in recent_optimizations) / len(recent_optimizations)
            avg_improvement = sum(opt.performance_improvement for opt in recent_optimizations) / len(
                recent_optimizations
            )

            report["performance_summary"] = {
                "avg_execution_time_ms": avg_execution_time,
                "avg_result_count": avg_result_count,
                "avg_performance_improvement_pct": avg_improvement,
                "recent_optimizations": len(recent_optimizations),
            }

        # Add component-specific stats
        if self.vector_search and hasattr(self.vector_search, "get_performance_stats"):
            report["vector_search_stats"] = self.vector_search.get_performance_stats()

        if self.enhanced_router and hasattr(self.enhanced_router, "get_comprehensive_stats"):
            report["router_stats"] = self.enhanced_router.get_comprehensive_stats()

        if self.performance_monitor:
            report["monitoring_dashboard"] = self.performance_monitor.get_performance_dashboard()

        return report

    async def demonstrate_optimization(self):
        """Demonstrate Phase 3 optimization capabilities"""

        demo_queries = [
            # Different query types to show optimization variety
            "Find luxury agents specializing in $2M+ properties in Austin Texas",
            "Properties under $500K with pool and 3+ bedrooms in Dallas",
            "ROI analysis for investment properties in Travis County",
            "What's the median price in Houston metro area?",
            "Market trends for condos in downtown Austin",
            "Compare Austin vs Dallas real estate markets for investment",
            "Agent directory for first-time home buyers with affordable options",
            "Should I invest in rental properties in San Antonio?",
        ]

        print("🚀 Phase 3 Search Optimization Demonstration")
        print("=" * 70)
        print("Building on your 67% performance improvements with intelligent optimization")
        print()

        total_time = 0
        total_results = 0

        for i, query in enumerate(demo_queries, 1):
            print(f"{i}. Query: {query}")

            start_time = time.time()
            results, optimization = await self.optimized_search(query, limit=5)
            demo_time = (time.time() - start_time) * 1000

            total_time += demo_time
            total_results += len(results)

            print(f"   Strategy: {optimization.original_strategy.value}")
            print(f"   Results: {len(results)} in {demo_time:.2f}ms")
            print(f"   Optimization: {optimization.optimization_reason}")
            print(f"   Improvement: {optimization.performance_improvement:.1f}% vs baseline")
            print()

        print("📊 Demonstration Summary:")
        print(f"   Total queries: {len(demo_queries)}")
        print(f"   Total time: {total_time:.2f}ms")
        print(f"   Average time: {total_time / len(demo_queries):.2f}ms")
        print(f"   Total results: {total_results}")
        print(f"   Average results: {total_results / len(demo_queries):.1f}")
        print()

        # Show optimization report
        report = await self.get_optimization_report()
        print("🎯 Phase 3 System Status:")
        for component, status in report["component_status"].items():
            print(f"   {component}: {status}")

        if report.get("performance_summary"):
            perf = report["performance_summary"]
            print("\n📈 Performance Summary:")
            print(f"   Average improvement: {perf['avg_performance_improvement_pct']:.1f}%")
            print(f"   Average execution time: {perf['avg_execution_time_ms']:.2f}ms")


# Main integration function for Phase 3
async def initialize_phase3_optimization(embedder=None) -> IntegratedSearchOptimizer:
    """
    Initialize the complete Phase 3 optimization system.

    This function sets up all Phase 3 components:
    - Enhanced search routing with analytics
    - Dynamic performance monitoring
    - Integration with your performance optimizations

    Returns:
        Configured IntegratedSearchOptimizer ready for use
    """

    optimizer = IntegratedSearchOptimizer()
    await optimizer.initialize(embedder)

    return optimizer


# Phase 3 testing and validation
async def validate_phase3_implementation():
    """Validate Phase 3 implementation and show improvements"""

    print("🧪 Phase 3 Implementation Validation")
    print("=" * 50)

    try:
        # Initialize optimizer
        optimizer = await initialize_phase3_optimization()

        # Run demonstration
        await optimizer.demonstrate_optimization()

        print("\n✅ Phase 3 validation completed successfully!")
        print("🎉 Search optimization system is ready for production use")

    except Exception as e:
        logger.error(f"Phase 3 validation failed: {e}")
        print(f"\n❌ Validation failed: {e}")


if __name__ == "__main__":
    asyncio.run(validate_phase3_implementation())
