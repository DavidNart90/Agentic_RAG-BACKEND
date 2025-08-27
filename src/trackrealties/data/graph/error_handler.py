"""
Error handling for graph operations.

This module provides error handling functionality for graph operations,
including retry logic, logging, and fallback mechanisms.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

from ...core.exceptions import GraphSearchError

logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar("T")


class GraphErrorHandler:
    """
    Error handler for graph operations.

    This class provides error handling functionality for graph operations,
    including retry logic, logging, and fallback mechanisms.
    """

    def __init__(
        self, max_retries: int = 3, retry_delay: float = 1.0, exponential_backoff: bool = True, log_errors: bool = True
    ):
        """
        Initialize the GraphErrorHandler.

        Args:
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
            exponential_backoff: Whether to use exponential backoff for retries
            log_errors: Whether to log errors
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.log_errors = log_errors
        self.logger = logging.getLogger(__name__)

    async def with_retry(self, operation: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute an operation with retry logic.

        Args:
            operation: Async function to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            GraphSearchError: If the operation fails after all retries
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                retries += 1
                last_error = e

                if self.log_errors:
                    self.logger.warning(f"Graph operation failed (attempt {retries}/{self.max_retries}): {e}")

                if retries > self.max_retries:
                    break

                # Calculate delay with exponential backoff if enabled
                delay = self.retry_delay
                if self.exponential_backoff:
                    delay *= 2 ** (retries - 1)

                await asyncio.sleep(delay)

        if self.log_errors:
            self.logger.error(f"Graph operation failed after {self.max_retries} retries: {last_error}")

        raise GraphSearchError(f"Graph operation failed: {last_error}")

    def with_fallback(self, fallback_value: T) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
        """
        Decorator to add fallback for graph operations.

        Args:
            fallback_value: Value to return if the operation fails

        Returns:
            Decorator function
        """

        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if self.log_errors:
                        self.logger.error(f"Graph operation failed, using fallback: {e}")
                    return fallback_value

            return wrapper

        return decorator

    async def execute_batch(
        self, operations: List[Callable[[], Awaitable[T]]], batch_size: int = 10, continue_on_error: bool = True
    ) -> List[Optional[T]]:
        """
        Execute a batch of operations with error handling.

        Args:
            operations: List of async functions to execute
            batch_size: Number of operations to execute in parallel
            continue_on_error: Whether to continue executing operations after an error

        Returns:
            List of operation results or None for failed operations
        """
        results: List[Optional[T]] = []

        for i in range(0, len(operations), batch_size):
            batch = operations[i: i + batch_size]
            batch_results = []

            for operation in batch:
                try:
                    result = await self.with_retry(operation)
                    batch_results.append(result)
                except Exception as e:
                    if self.log_errors:
                        self.logger.error(f"Batch operation failed: {e}")

                    if continue_on_error:
                        batch_results.append(None)
                    else:
                        raise

            results.extend(batch_results)

        return results

    def format_error_response(
        self, error: Exception, operation_type: str, entity_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format an error response for graph operations.

        Args:
            error: Exception that occurred
            operation_type: Type of operation that failed
            entity_id: ID of the entity being operated on

        Returns:
            Dictionary with error information
        """
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "operation_type": operation_type,
            "entity_id": entity_id or "unknown",
            "timestamp": time.time(),
        }
