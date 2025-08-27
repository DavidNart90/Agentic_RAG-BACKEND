"""
Phase 3: Dynamic Performance Monitoring and Optimization

This module provides real-time performance tracking and optimization for the RAG pipeline,
building on your existing performance improvements from the optimized pipeline.

Features:
- Real-time performance monitoring
- Dynamic threshold adjustment
- Performance trend analysis
- Automatic optimization recommendations
- Integration with existing PerformanceMetrics
"""

import asyncio
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from ..analytics.search import SearchAnalytics
from .enhanced_smart_search_router import (EnhancedSmartSearchRouter,
                                           StrategyPerformance)
from .performance_optimized_pipeline import (OptimizedVectorSearchV2,
                                             PerformanceMetrics)

logger = logging.getLogger(__name__)


class PerformanceAlert(Enum):
    """Types of performance alerts"""

    HIGH_RESPONSE_TIME = "high_response_time"
    LOW_SUCCESS_RATE = "low_success_rate"
    CACHE_INEFFICIENCY = "cache_inefficiency"
    STRATEGY_UNDERPERFORMANCE = "strategy_underperformance"
    DATABASE_BOTTLENECK = "database_bottleneck"
    MEMORY_PRESSURE = "memory_pressure"


@dataclass
class AlertCondition:
    """Defines conditions for performance alerts"""

    alert_type: PerformanceAlert
    threshold: float
    measurement_window: int  # Number of samples to consider
    severity: str  # 'warning', 'critical'
    enabled: bool = True


@dataclass
class PerformanceTrend:
    """Track performance trends over time"""

    metric_name: str
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new value to the trend"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.values.append(value)
        self.timestamps.append(timestamp)

    @property
    def current_trend(self) -> str:
        """Determine if the trend is improving, degrading, or stable"""
        if len(self.values) < 5:
            return "insufficient_data"

        recent_values = list(self.values)[-5:]
        if len(recent_values) < 2:
            return "stable"

        # Simple trend detection
        increasing = sum(1 for i in range(1, len(recent_values)) if recent_values[i] > recent_values[i - 1])
        decreasing = sum(1 for i in range(1, len(recent_values)) if recent_values[i] < recent_values[i - 1])

        if increasing >= 3:
            return "improving" if self.metric_name in ["success_rate", "cache_hit_rate"] else "degrading"
        elif decreasing >= 3:
            return "degrading" if self.metric_name in ["success_rate", "cache_hit_rate"] else "improving"
        else:
            return "stable"

    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of the trend"""
        if not self.values:
            return {}

        values_list = list(self.values)
        return {
            "mean": statistics.mean(values_list),
            "median": statistics.median(values_list),
            "std_dev": statistics.stdev(values_list) if len(values_list) > 1 else 0.0,
            "min": min(values_list),
            "max": max(values_list),
            "current": values_list[-1],
            "trend": self.current_trend,
        }


class DynamicPerformanceMonitor:
    """
    Real-time performance monitoring with dynamic optimization capabilities.

    This monitor builds on your existing PerformanceMetrics and provides:
    - Real-time performance tracking
    - Trend analysis and alerting
    - Dynamic threshold adjustment
    - Performance optimization recommendations
    """

    def __init__(self, enhanced_router: EnhancedSmartSearchRouter, vector_search: OptimizedVectorSearchV2):
        self.enhanced_router = enhanced_router
        self.vector_search = vector_search

        # Performance tracking
        self.performance_trends: Dict[str, PerformanceTrend] = {
            "search_response_time": PerformanceTrend("search_response_time"),
            "search_success_rate": PerformanceTrend("search_success_rate"),
            "cache_hit_rate": PerformanceTrend("cache_hit_rate"),
            "database_query_time": PerformanceTrend("database_query_time"),
            "embedding_generation_time": PerformanceTrend("embedding_generation_time"),
            "vector_search_time": PerformanceTrend("vector_search_time"),
            "graph_search_time": PerformanceTrend("graph_search_time"),
            "strategy_effectiveness": PerformanceTrend("strategy_effectiveness"),
        }

        # Alert configuration
        self.alert_conditions = [
            AlertCondition(PerformanceAlert.HIGH_RESPONSE_TIME, 5000.0, 5, "warning"),  # 5 seconds
            AlertCondition(PerformanceAlert.HIGH_RESPONSE_TIME, 10000.0, 3, "critical"),  # 10 seconds
            AlertCondition(PerformanceAlert.LOW_SUCCESS_RATE, 0.7, 10, "warning"),  # 70%
            AlertCondition(PerformanceAlert.LOW_SUCCESS_RATE, 0.5, 5, "critical"),  # 50%
            AlertCondition(PerformanceAlert.CACHE_INEFFICIENCY, 0.3, 20, "warning"),  # 30% hit rate
            AlertCondition(PerformanceAlert.STRATEGY_UNDERPERFORMANCE, 0.6, 10, "warning"),  # 60% effectiveness
        ]

        # Dynamic thresholds
        self.dynamic_thresholds = {
            "response_time_target": 2000.0,  # Start with 2 seconds target
            "success_rate_target": 0.85,  # 85% success rate target
            "cache_hit_target": 0.5,  # 50% cache hit rate target
        }

        # Monitoring state
        self.monitoring_active = False
        self.last_optimization_check = datetime.now(timezone.utc)
        self.optimization_interval = timedelta(minutes=15)  # Check every 15 minutes

        # Performance callbacks
        self.alert_callbacks: List[Callable] = []
        self.optimization_callbacks: List[Callable] = []

    def register_alert_callback(self, callback: Callable[[PerformanceAlert, Dict], None]):
        """Register callback for performance alerts"""
        self.alert_callbacks.append(callback)

    def register_optimization_callback(self, callback: Callable[[Dict], None]):
        """Register callback for optimization recommendations"""
        self.optimization_callbacks.append(callback)

    async def start_monitoring(self, interval_seconds: int = 30):
        """Start real-time performance monitoring"""
        self.monitoring_active = True
        logger.info("Starting dynamic performance monitoring")

        while self.monitoring_active:
            try:
                await self._collect_performance_metrics()
                await self._check_alert_conditions()
                await self._check_optimization_opportunities()

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Stopped dynamic performance monitoring")

    async def _collect_performance_metrics(self):
        """Collect current performance metrics from all components"""

        # Collect vector search metrics
        if hasattr(self.vector_search, "get_performance_stats"):
            vector_stats = self.vector_search.get_performance_stats()

            # Extract and record metrics
            search_metrics = vector_stats.get("search_metrics", {})
            embedding_metrics = vector_stats.get("embedding_metrics", {})

            if search_metrics.get("avg_query_time_ms"):
                self.performance_trends["search_response_time"].add_value(search_metrics["avg_query_time_ms"])

            if embedding_metrics.get("avg_embedding_time_ms"):
                self.performance_trends["embedding_generation_time"].add_value(
                    embedding_metrics["avg_embedding_time_ms"]
                )

            if embedding_metrics.get("cache_hit_rate") is not None:
                self.performance_trends["cache_hit_rate"].add_value(embedding_metrics["cache_hit_rate"])

        # Collect router performance metrics
        if hasattr(self.enhanced_router, "get_comprehensive_stats"):
            router_stats = self.enhanced_router.get_comprehensive_stats()

            strategy_performance = router_stats.get("strategy_performance", {})
            for strategy_name, perf in strategy_performance.items():
                if perf.get("success_rate") is not None:
                    self.performance_trends["search_success_rate"].add_value(perf["success_rate"])

                if perf.get("avg_response_time") is not None:
                    self.performance_trends["vector_search_time"].add_value(perf["avg_response_time"])

    async def _check_alert_conditions(self):
        """Check if any alert conditions are met"""

        for condition in self.alert_conditions:
            if not condition.enabled:
                continue

            alert_triggered = await self._evaluate_alert_condition(condition)
            if alert_triggered:
                await self._trigger_alert(condition)

    async def _evaluate_alert_condition(self, condition: AlertCondition) -> bool:
        """Evaluate if a specific alert condition is met"""

        if condition.alert_type == PerformanceAlert.HIGH_RESPONSE_TIME:
            trend = self.performance_trends.get("search_response_time")
            if trend and len(trend.values) >= condition.measurement_window:
                recent_values = list(trend.values)[-condition.measurement_window:]
                avg_time = statistics.mean(recent_values)
                return avg_time > condition.threshold

        elif condition.alert_type == PerformanceAlert.LOW_SUCCESS_RATE:
            trend = self.performance_trends.get("search_success_rate")
            if trend and len(trend.values) >= condition.measurement_window:
                recent_values = list(trend.values)[-condition.measurement_window:]
                avg_success = statistics.mean(recent_values)
                return avg_success < condition.threshold

        elif condition.alert_type == PerformanceAlert.CACHE_INEFFICIENCY:
            trend = self.performance_trends.get("cache_hit_rate")
            if trend and len(trend.values) >= condition.measurement_window:
                recent_values = list(trend.values)[-condition.measurement_window:]
                avg_hit_rate = statistics.mean(recent_values)
                return avg_hit_rate < condition.threshold

        return False

    async def _trigger_alert(self, condition: AlertCondition):
        """Trigger a performance alert"""

        alert_data = {
            "alert_type": condition.alert_type,
            "severity": condition.severity,
            "threshold": condition.threshold,
            "timestamp": datetime.now(timezone.utc),
            "current_metrics": self._get_current_metrics_summary(),
        }

        logger.warning(
            f"Performance alert triggered: {condition.alert_type.value} " f"(severity: {condition.severity})"
        )

        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(condition.alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error calling alert callback: {e}")

    async def _check_optimization_opportunities(self):
        """Check for performance optimization opportunities"""

        current_time = datetime.now(timezone.utc)
        if current_time - self.last_optimization_check < self.optimization_interval:
            return

        self.last_optimization_check = current_time

        recommendations = await self._generate_optimization_recommendations()
        if recommendations:
            logger.info(f"Generated {len(recommendations)} optimization recommendations")

            for callback in self.optimization_callbacks:
                try:
                    await callback(recommendations)
                except Exception as e:
                    logger.error(f"Error calling optimization callback: {e}")

    async def _generate_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate performance optimization recommendations"""

        recommendations = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": [],
            "current_performance": self._get_current_metrics_summary(),
            "trends": self._get_trend_analysis(),
        }

        # Analyze cache performance
        cache_trend = self.performance_trends.get("cache_hit_rate")
        if cache_trend and cache_trend.values:
            current_hit_rate = list(cache_trend.values)[-1]
            if current_hit_rate < 0.4:  # Less than 40% hit rate
                recommendations["recommendations"].append(
                    {
                        "type": "cache_optimization",
                        "priority": "high",
                        "description": "Cache hit rate is low. Consider increasing cache size or improving cache key generation.",
                        "current_value": current_hit_rate,
                        "target_value": 0.6,
                        "actions": [
                            "Increase embedding cache size",
                            "Implement query normalization for better cache hits",
                            "Add result caching for frequently requested queries",
                        ],
                    }
                )

        # Analyze response time performance
        response_trend = self.performance_trends.get("search_response_time")
        if response_trend and response_trend.values:
            current_response_time = list(response_trend.values)[-1]
            if current_response_time > self.dynamic_thresholds["response_time_target"]:
                recommendations["recommendations"].append(
                    {
                        "type": "response_time_optimization",
                        "priority": "medium",
                        "description": "Search response time exceeds target. Consider query optimization or parallel processing.",
                        "current_value": current_response_time,
                        "target_value": self.dynamic_thresholds["response_time_target"],
                        "actions": [
                            "Implement parallel vector and graph search execution",
                            "Optimize database indexes for frequent query patterns",
                            "Implement result streaming for faster perceived performance",
                        ],
                    }
                )

        # Analyze strategy effectiveness
        router_stats = self.enhanced_router.get_comprehensive_stats()
        strategy_performance = router_stats.get("strategy_performance", {})

        for strategy_name, perf in strategy_performance.items():
            if perf.get("total_executions", 0) >= 10:  # Enough data for analysis
                success_rate = perf.get("success_rate", 0.0)
                if success_rate < 0.7:  # Less than 70% success rate
                    recommendations["recommendations"].append(
                        {
                            "type": "strategy_optimization",
                            "priority": "medium",
                            "description": f"Strategy {strategy_name} has low success rate. Consider strategy tuning.",
                            "strategy": strategy_name,
                            "current_success_rate": success_rate,
                            "target_success_rate": 0.8,
                            "actions": [
                                f"Review {strategy_name} implementation for edge cases",
                                "Adjust strategy selection criteria",
                                "Implement better fallback mechanisms",
                            ],
                        }
                    )

        return recommendations

    def _get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current performance metrics"""

        summary = {}
        for trend_name, trend in self.performance_trends.items():
            if trend.values:
                summary[trend_name] = {
                    "current": list(trend.values)[-1],
                    "trend": trend.current_trend,
                    "sample_count": len(trend.values),
                }

        return summary

    def _get_trend_analysis(self) -> Dict[str, Any]:
        """Get detailed trend analysis"""

        analysis = {}
        for trend_name, trend in self.performance_trends.items():
            stats = trend.get_statistics()
            if stats:
                analysis[trend_name] = stats

        return analysis

    async def adjust_dynamic_thresholds(self):
        """Adjust performance thresholds based on historical performance"""

        # Adjust response time target based on recent performance
        response_trend = self.performance_trends.get("search_response_time")
        if response_trend and len(response_trend.values) >= 20:
            recent_response_times = list(response_trend.values)[-20:]
            p75_response_time = statistics.quantiles(recent_response_times, n=4)[2]  # 75th percentile

            # Set target to 75th percentile + 20% buffer
            new_target = p75_response_time * 1.2

            # Don't let target drift too high
            max_target = 5000.0  # 5 seconds max
            self.dynamic_thresholds["response_time_target"] = min(new_target, max_target)

            logger.info(f"Adjusted response time target to {self.dynamic_thresholds['response_time_target']:.0f}ms")

        # Adjust success rate target based on recent performance
        success_trend = self.performance_trends.get("search_success_rate")
        if success_trend and len(success_trend.values) >= 20:
            recent_success_rates = list(success_trend.values)[-20:]
            avg_success_rate = statistics.mean(recent_success_rates)

            # Set target to recent average - 5% (to maintain improvement pressure)
            new_target = max(0.8, avg_success_rate - 0.05)  # Minimum 80%
            self.dynamic_thresholds["success_rate_target"] = new_target

            logger.info(f"Adjusted success rate target to {self.dynamic_thresholds['success_rate_target']:.2%}")

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""

        dashboard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monitoring_active": self.monitoring_active,
            "dynamic_thresholds": self.dynamic_thresholds,
            "current_metrics": self._get_current_metrics_summary(),
            "trend_analysis": self._get_trend_analysis(),
            "alert_status": {
                "total_conditions": len(self.alert_conditions),
                "active_conditions": len([c for c in self.alert_conditions if c.enabled]),
            },
            "router_stats": (
                self.enhanced_router.get_comprehensive_stats()
                if hasattr(self.enhanced_router, "get_comprehensive_stats")
                else {}
            ),
            "vector_search_stats": (
                self.vector_search.get_performance_stats()
                if hasattr(self.vector_search, "get_performance_stats")
                else {}
            ),
        }

        return dashboard


# Example alert handlers and optimization callbacks
async def default_alert_handler(alert_type: PerformanceAlert, alert_data: Dict):
    """Default alert handler that logs alerts"""
    logger.warning(f"🚨 Performance Alert: {alert_type.value}")
    logger.warning(f"   Severity: {alert_data['severity']}")
    logger.warning(f"   Threshold: {alert_data['threshold']}")
    logger.warning(f"   Current metrics: {alert_data['current_metrics']}")


async def default_optimization_callback(recommendations: Dict):
    """Default optimization callback that logs recommendations"""
    logger.info("🔧 Performance Optimization Recommendations:")

    for i, rec in enumerate(recommendations["recommendations"], 1):
        logger.info(f"   {i}. {rec['type']} (Priority: {rec['priority']})")
        logger.info(f"      {rec['description']}")
        if "actions" in rec:
            for action in rec["actions"]:
                logger.info(f"      - {action}")


# Utility function for Phase 3 testing
async def test_dynamic_performance_monitoring():
    """Test dynamic performance monitoring functionality"""

    from ..analytics.search import search_analytics
    from .optimized_pipeline import OptimizedVectorSearch
    from .smart_search import FixedGraphSearch

    print("🎯 Phase 3: Dynamic Performance Monitoring Test")
    print("=" * 60)

    # Initialize components
    vector_search = OptimizedVectorSearchV2(None)  # Mock embedder for testing
    graph_search = FixedGraphSearch()
    enhanced_router = EnhancedSmartSearchRouter(
        vector_search=vector_search, graph_search=graph_search, analytics=search_analytics
    )

    # Create monitor
    monitor = DynamicPerformanceMonitor(enhanced_router, vector_search)

    # Register default handlers
    monitor.register_alert_callback(default_alert_handler)
    monitor.register_optimization_callback(default_optimization_callback)

    print("✅ Performance monitor created with default handlers")

    # Simulate some performance data
    for i in range(10):
        # Simulate varying performance
        response_time = 1500 + (i * 200)  # Gradually increasing
        success_rate = 0.9 - (i * 0.05)  # Gradually decreasing
        cache_hit_rate = 0.3 + (i * 0.02)  # Gradually improving

        monitor.performance_trends["search_response_time"].add_value(response_time)
        monitor.performance_trends["search_success_rate"].add_value(success_rate)
        monitor.performance_trends["cache_hit_rate"].add_value(cache_hit_rate)

        print(f"   Sample {i + 1}: Response {response_time}ms, Success {success_rate:.1%}, Cache {cache_hit_rate:.1%}")

    # Check for alerts
    print("\n🚨 Checking alert conditions...")
    await monitor._check_alert_conditions()

    # Generate optimization recommendations
    print("\n🔧 Generating optimization recommendations...")
    recommendations = await monitor._generate_optimization_recommendations()
    if recommendations["recommendations"]:
        await default_optimization_callback(recommendations)
    else:
        print("   No optimization recommendations at this time")

    # Show dashboard
    print("\n📊 Performance Dashboard:")
    dashboard = monitor.get_performance_dashboard()
    print(f"   Active monitoring: {dashboard['monitoring_active']}")
    print(f"   Alert conditions: {dashboard['alert_status']['active_conditions']}")
    print(f"   Metrics tracked: {len(dashboard['current_metrics'])}")


if __name__ == "__main__":
    asyncio.run(test_dynamic_performance_monitoring())
