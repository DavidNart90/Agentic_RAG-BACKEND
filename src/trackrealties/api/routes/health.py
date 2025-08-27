"""
API route for health checks.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ...core.config import get_settings
from ...core.health import HealthChecker, get_health_checker
from ...models.api import HealthStatus

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=HealthStatus)
async def health_check(health_checker: HealthChecker = Depends(get_health_checker)):
    """Health check endpoint."""
    try:
        # Use the comprehensive system health check
        health_report = await health_checker.check_system_health()

        return HealthStatus(
            status=health_report["status"],
            database=health_report["components"]["database"]["healthy"],
            graph_database=health_report["components"]["neo4j"]["healthy"],
            llm_connection=health_report["components"]["llm"]["healthy"],
            version=health_report["version"],
            timestamp=datetime.fromisoformat(health_report["timestamp"]),
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")
