"""Health check utilities for TrackRealties AI Platform."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from ..data.graph.graph_builder import GraphBuilder
from .config import Settings
from .database import db_manager, test_connection
from .graph import graph_manager
from .logging import LoggerMixin

logger = structlog.get_logger(__name__)


@dataclass
class HealthStatus:
    """Health status for a component."""

    healthy: bool
    message: str
    details: Dict[str, Any]
    response_time_ms: float


class HealthChecker(LoggerMixin):
    """Performs health checks on system components."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def check_database_health(self) -> HealthStatus:
        """Check PostgreSQL database health."""
        start_time = datetime.now(timezone.utc)

        try:
            is_healthy = await test_connection()

            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            if is_healthy:
                return HealthStatus(
                    healthy=True,
                    message="Database is healthy",
                    details={
                        "database_url": self.settings.DATABASE_URL.split("@")[0] + "@***",  # Hide credentials
                        "pool_size": self.settings.DB_POOL_SIZE,
                    },
                    response_time_ms=response_time,
                )
            else:
                return HealthStatus(
                    healthy=False, message="Database health check failed", details={}, response_time_ms=response_time
                )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            self.logger.error(f"Database health check failed: {str(e)}")
            return HealthStatus(
                healthy=False,
                message=f"Database error: {str(e)}",
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    async def check_neo4j_health(self) -> HealthStatus:
        """Check Neo4j database health."""
        start_time = datetime.now(timezone.utc)

        try:
            neo4j_manager = graph_manager
            await neo4j_manager.initialize()
            is_healthy = await neo4j_manager.test_connection()

            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            if is_healthy:
                return HealthStatus(
                    healthy=True,
                    message="Neo4j is healthy",
                    details={
                        "neo4j_uri": (
                            self.settings.neo4j_uri.split("@")[0] + "@***"
                            if "@" in self.settings.neo4j_uri
                            else self.settings.neo4j_uri
                        ),
                        "database": self.settings.neo4j_database,
                    },
                    response_time_ms=response_time,
                )
            else:
                return HealthStatus(
                    healthy=False, message="Neo4j health check failed", details={}, response_time_ms=response_time
                )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            self.logger.error(f"Neo4j health check failed: {str(e)}")
            return HealthStatus(
                healthy=False,
                message=f"Neo4j error: {str(e)}",
                details={"error": str(e)},
                response_time_ms=response_time,
            )

    async def check_llm_health(self) -> HealthStatus:
        """Check LLM provider health."""
        start_time = datetime.now(timezone.utc)

        try:
            # Simple test to check if LLM configuration is valid
            # In a real implementation, you might make a test API call

            if not self.settings.llm_api_key:
                return HealthStatus(
                    healthy=False,
                    message="LLM API key not configured",
                    details={"provider": self.settings.LLM_PROVIDER},
                    response_time_ms=0,
                )

            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            return HealthStatus(
                healthy=True,
                message="LLM configuration is valid",
                details={
                    "provider": self.settings.LLM_PROVIDER,
                    "model": self.settings.LLM_MODEL,
                },
                response_time_ms=response_time,
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            self.logger.error(f"LLM health check failed: {str(e)}")
            return HealthStatus(
                healthy=False, message=f"LLM error: {str(e)}", details={"error": str(e)}, response_time_ms=response_time
            )

    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        self.logger.info("Starting system health check")

        # Run all health checks concurrently
        import asyncio

        database_task = asyncio.create_task(self.check_database_health())
        neo4j_task = asyncio.create_task(self.check_neo4j_health())
        llm_task = asyncio.create_task(self.check_llm_health())

        database_health = await database_task
        neo4j_health = await neo4j_task
        llm_health = await llm_task

        # Determine overall system health
        all_healthy = all([database_health.healthy, neo4j_health.healthy, llm_health.healthy])

        # Calculate overall status
        if all_healthy:
            overall_status = "healthy"
        elif database_health.healthy:  # Core database is healthy
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        health_report = {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": self.settings.APP_VERSION,
            "environment": self.settings.APP_ENV,
            "components": {
                "database": {
                    "healthy": database_health.healthy,
                    "message": database_health.message,
                    "response_time_ms": database_health.response_time_ms,
                    "details": database_health.details,
                },
                "neo4j": {
                    "healthy": neo4j_health.healthy,
                    "message": neo4j_health.message,
                    "response_time_ms": neo4j_health.response_time_ms,
                    "details": neo4j_health.details,
                },
                "llm": {
                    "healthy": llm_health.healthy,
                    "message": llm_health.message,
                    "response_time_ms": llm_health.response_time_ms,
                    "details": llm_health.details,
                },
            },
            "summary": {
                "total_components": 3,
                "healthy_components": sum([database_health.healthy, neo4j_health.healthy, llm_health.healthy]),
                "average_response_time_ms": (
                    database_health.response_time_ms + neo4j_health.response_time_ms + llm_health.response_time_ms
                ) / 3,
            },
        }

        self.logger.info(
            "System health check completed",
            status=overall_status,
            healthy_components=health_report["summary"]["healthy_components"],
            total_components=health_report["summary"]["total_components"],
        )

        return health_report


async def get_health_checker(settings: Optional[Settings] = None) -> HealthChecker:
    """Get a health checker instance."""
    if settings is None:
        from .config import get_settings

        settings = get_settings()

    return HealthChecker(settings)
