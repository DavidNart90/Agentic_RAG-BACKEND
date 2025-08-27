"""Database utilities for PostgreSQL connection and operations.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
from asyncpg.pool import Pool

from .config import settings

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages PostgreSQL connection pool."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database pool.

        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or settings.DATABASE_URL
        if not self.database_url:
            raise ValueError("DATABASE_URL not set in environment or config")

        self.pool: Optional[Pool] = None

    async def initialize(self):
        """Create connection pool."""
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url, min_size=5, max_size=20, max_inactive_connection_lifetime=300, command_timeout=60
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as connection:
            yield connection


class DatabaseManager:
    """
    Database manager that provides high-level database operations.
    This wraps the DatabasePool to provide a consistent interface.
    """

    def __init__(self, pool: Optional[DatabasePool] = None):
        """
        Initialize database manager.

        Args:
            pool: DatabasePool instance (optional)
        """
        self.pool = pool or db_pool
        self.logger = logging.getLogger(f"{__name__}.DatabaseManager")

    async def initialize(self):
        """Initialize the underlying connection pool."""
        await self.pool.initialize()

    async def close(self):
        """Close the underlying connection pool."""
        await self.pool.close()

    @asynccontextmanager
    async def get_connection(self):
        """
        Get a database connection from the pool.

        Yields:
            Database connection
        """
        async with self.pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args):
        """
        Execute a query that doesn't return results.

        Args:
            query: SQL query
            *args: Query parameters
        """
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)

    async def fetch_one(self, query: str, *args):
        """
        Execute a query and fetch a single row.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single row or None
        """
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch_all(self, query: str, *args):
        """
        Execute a query and fetch all rows.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of rows
        """
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)

    async def fetch_val(self, query: str, *args):
        """
        Execute a query and fetch a single value.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single value
        """
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)

    @asynccontextmanager
    async def transaction(self):
        """
        Create a database transaction context.

        Usage:
            async with db_manager.transaction():
                await db_manager.execute(query1)
                await db_manager.execute(query2)
        """
        async with self.get_connection() as conn:
            async with conn.transaction():
                # Temporarily store connection for nested operations
                self._transaction_conn = conn
                try:
                    yield
                finally:
                    self._transaction_conn = None


# Global instances
db_pool = DatabasePool()
db_manager = DatabaseManager(db_pool)


async def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        True if connection successful
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        logger.info("Database connection successful.")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
