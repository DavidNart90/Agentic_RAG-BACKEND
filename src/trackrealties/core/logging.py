"""Structured logging configuration for TrackRealties AI Platform."""

import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure structured logging with Rich formatting."""

    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=settings.debug,
                markup=True,
                rich_tracebacks=True,
            )
        ],
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level)),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add structured logging to any class."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(
    logger: structlog.BoundLogger,
    function_name: str,
    args: Optional[Dict[str, Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Log function call with parameters."""
    logger.debug(
        "Function called",
        function=function_name,
        args=args or {},
        kwargs=kwargs or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def log_performance(
    logger: structlog.BoundLogger,
    operation: str,
    duration_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log performance metrics."""
    logger.info(
        "Performance metric",
        operation=operation,
        duration_ms=duration_ms,
        metadata=metadata or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def log_error(
    logger: structlog.BoundLogger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log error with context."""
    logger.error(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
        exc_info=True,
    )


def log_validation_result(
    logger: structlog.BoundLogger,
    validation_type: str,
    is_valid: bool,
    confidence_score: float,
    issues: Optional[list] = None,
) -> None:
    """Log validation results."""
    logger.info(
        "Validation completed",
        validation_type=validation_type,
        is_valid=is_valid,
        confidence_score=confidence_score,
        issues_count=len(issues) if issues else 0,
        issues=issues or [],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def log_agent_interaction(
    logger: structlog.BoundLogger,
    agent_type: str,
    query: str,
    response_length: int,
    tools_used: list,
    processing_time_ms: float,
) -> None:
    """Log agent interaction details."""
    logger.info(
        "Agent interaction",
        agent_type=agent_type,
        query_length=len(query),
        response_length=response_length,
        tools_used=tools_used,
        processing_time_ms=processing_time_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def log_database_operation(
    logger: structlog.BoundLogger,
    operation: str,
    table: str,
    duration_ms: float,
    rows_affected: Optional[int] = None,
) -> None:
    """Log database operation details."""
    logger.debug(
        "Database operation",
        operation=operation,
        table=table,
        duration_ms=duration_ms,
        rows_affected=rows_affected,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def log_external_api_call(
    logger: structlog.BoundLogger,
    provider: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
    response_size: Optional[int] = None,
) -> None:
    """Log external API call details."""
    logger.info(
        "External API call",
        provider=provider,
        endpoint=endpoint,
        status_code=status_code,
        duration_ms=duration_ms,
        response_size=response_size,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
