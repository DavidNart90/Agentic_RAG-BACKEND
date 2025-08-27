"""
Custom middleware for the FastAPI application.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Callable

import redis
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(f"Response: {response.status_code} " f"processed in {process_time:.3f}s")

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.redis_client = None

        # Initialize Redis if available
        try:
            if settings.redis_url:
                self.redis_client = redis.from_url(settings.redis_url)
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting: {e}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        if not await self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                },
            )

        return await call_next(request)

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use IP address as default identifier
        client_ip = request.client.host if request.client else "unknown"

        # Could also use API key or user ID if available
        auth_header = request.headers.get("authorization")
        if auth_header:
            return f"auth:{auth_header[:20]}"  # Use first 20 chars of auth header

        return f"ip:{client_ip}"

    async def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        if not self.redis_client:
            # If Redis not available, allow all requests
            return True

        try:
            current_time = int(time.time())
            window_start = current_time - 60  # 1 minute window

            # Use Redis sorted set to track requests
            key = f"rate_limit:{client_id}"

            # Remove old entries
            # Use a pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, 60)
            pipe.zcard(key)
            results = await pipe.execute()

            current_requests = results[-1]

            return current_requests <= self.requests_per_minute

        except redis.exceptions.ConnectionError as e:
            logger.warning(f"Could not connect to Redis for rate limiting: {e}")
            return True  # Allow request if Redis is down

        except Exception as e:
            logger.error(f"An unexpected error occurred during rate limiting check: {e}")
            return True  # Allow request on other unexpected errors


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unhandled exception: {e}", exc_info=True)

            return JSONResponse(
                status_code=500, content={"error": "Internal server error", "message": "An unexpected error occurred"}
            )
